#include "sampler/isotropic/stiefel_gaussian_sampler.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace sampler {

namespace math = isomorphism::math;
using Tensor   = isomorphism::Tensor;
using DType    = isomorphism::DType;

// =============================================================================
// Construction
// =============================================================================

StiefelGaussianSampler::StiefelGaussianSampler(Tensor x_hat, int n, int k, Config cfg)
    : x_hat_(x_hat), n_(n), k_(k), cfg_(cfg)
{
    assert(n >= 1 && k >= 1 && k < n);
    alpha_  = cfg_.alpha;
    is_fat_ = (n_ < 2 * k_);
    k_eff_  = is_fat_ ? (n_ - k_) : k_;

    // --- Precompute the dual consensus frame for fat frames ---
    if (is_fat_) {
        // x_hat_ is 2D [n, k], so transpose uses {1, 0}
        Tensor Z = math::random_normal({n_, k_eff_}, cfg_.dtype);
        Tensor ZtX = math::matmul(math::transpose(x_hat_, {1, 0}), Z);
        Tensor proj = math::matmul(x_hat_, ZtX);
        x_hat_prime_ = thin_qr(math::subtract(Z, proj));
    }

    rebuild_angle_sampler();
}

void StiefelGaussianSampler::rebuild_angle_sampler() {
    auto acfg  = cfg_.angle_cfg;
    acfg.n     = n_;
    acfg.k     = k_eff_;
    acfg.alpha = alpha_;

    angle_sampler_ = std::make_unique<PrincipalAngleSampler>(acfg);
}

void StiefelGaussianSampler::set_config(Config cfg) {
    cfg_   = cfg;
    alpha_ = cfg_.alpha;
    rebuild_angle_sampler();
}

void StiefelGaussianSampler::set_x_hat(isomorphism::Tensor x_hat) {
    x_hat_ = std::move(x_hat);
    if (is_fat_) {
        // Recompute the orthogonal complement for the new consensus frame.
        Tensor Z    = math::random_normal({n_, k_eff_}, cfg_.dtype);
        Tensor ZtX  = math::matmul(math::transpose(x_hat_, {1, 0}), Z);
        Tensor proj = math::matmul(x_hat_, ZtX);
        x_hat_prime_ = thin_qr(math::subtract(Z, proj));
    }
}

void StiefelGaussianSampler::update_alpha(double alpha, int burn_in_steps) {
    alpha_     = alpha;
    cfg_.alpha = alpha;
    angle_sampler_->set_alpha(alpha, burn_in_steps);
}

// =============================================================================
// Public sample() – Batched Tensor Generation
// =============================================================================

Tensor StiefelGaussianSampler::sample() {
    int N = cfg_.num_samples;
    if (N <= 0) N = 1;

    // 1. Fetch the N independent sets of principal angles as a flat array.
    //    High-α: tangent-space Wishart sampling.
    //    Otherwise: HMC/Splines on the Weyl chamber.
    std::vector<double> flat_theta = (alpha_ >= kHighAlphaThreshold)
        ? sample_angles_tangent()
        : angle_sampler_->sample_angles(N);

    Tensor sample_frame;

    // 2. Batched Geometric Mapping
    // We execute the O(k^3) and O(nk^2) math natively on tensors of shape [N, ...].
    // MLX will build a unified graph and execute it optimally on the GPU without
    // thread locks or race conditions.
    if (!is_fat_) {
        // ------------------------------------------------------------------
        // Tall-and-skinny (n >= 2k): O(k³) Spectral Lift
        // ------------------------------------------------------------------
        auto [Theta11, Theta21, Theta12, Theta22] = compute_shape_blocks(flat_theta, k_);

        // Draw U ∈ ℝ^{N×n×k}: batched random orthonormal k-frames in X_hat^⊥
        Tensor Z       = math::random_normal({N, n_, k_}, cfg_.dtype);
        Tensor x_hat_b = math::expand_dims(x_hat_, {0});                      // [1, n, k]
        // 3D transpose: {0, 2, 1} swaps row and col, leaves batch intact
        Tensor x_hat_T = math::transpose(x_hat_b, {0, 2, 1});                 // [1, k, n]

        // Implicit broadcasting: [1, k, n] @ [N, n, k] -> [N, k, k]
        Tensor ZtX     = math::matmul(x_hat_T, Z);
        Tensor proj    = math::matmul(x_hat_b, ZtX);                          // [N, n, k]
        Tensor U       = thin_qr(math::subtract(Z, proj));                    // [N, n, k]

        // Assemble final frame: X = X_hat * Θ₁₁ + U * Θ₂₁
        Tensor X_rot   = math::matmul(x_hat_b, Theta11);                      // [N, n, k]
        Tensor U_rot   = math::matmul(U, Theta21);                            // [N, n, k]
        sample_frame   = math::add(X_rot, U_rot);
    }
    else {
        // ------------------------------------------------------------------
        // Fat frame (n < 2k): O(n(k')^2) Geometric Duality
        // ------------------------------------------------------------------
        auto [Theta11, Theta21, Theta12, Theta22] = compute_shape_blocks(flat_theta, k_eff_);

        // Draw U ∈ ℝ^{N×n×k'}: batched random orthonormal k'-frames in X_hat_prime^⊥
        Tensor Z       = math::random_normal({N, n_, k_eff_}, cfg_.dtype);
        Tensor xp_b    = math::expand_dims(x_hat_prime_, {0});                // [1, n, k']
        Tensor xp_T    = math::transpose(xp_b, {0, 2, 1});                    // [1, k', n]

        Tensor ZtXp    = math::matmul(xp_T, Z);                               // [N, k', k']
        Tensor proj    = math::matmul(xp_b, ZtXp);                            // [N, n, k']
        Tensor U       = thin_qr(math::subtract(Z, proj));                    // [N, n, k']

        // Exact rotation back to primal space
        Tensor x_hat_b = math::expand_dims(x_hat_, {0});                      // [1, n, k]
        Tensor Ut      = math::transpose(U, {0, 2, 1});                       // [N, k', n]
        Tensor UtX     = math::matmul(Ut, x_hat_b);                           // [N, k', k]

        Tensor rot_Xp  = math::matmul(xp_b, Theta12);                         // [N, n, k']
        Tensor rot_U   = math::matmul(U, Theta22);                            // [N, n, k']
        Tensor sum     = math::add(rot_Xp, rot_U);
        Tensor diff    = math::subtract(sum, U);

        Tensor correction = math::matmul(diff, UtX);                          // [N, n, k]
        sample_frame      = math::add(x_hat_b, correction);
    }

    math::eval(sample_frame);

    // 3. Output Formatting
    if (N == 1) {
        // Strip the batch dimension for legacy [n, k] shape compatibility
        return math::slice(sample_frame, 0, 1, 0);
    }

    // High-performance batch path: Returns shape [N, n, k].
    return sample_frame;
}

Tensor StiefelGaussianSampler::draw_uniform() {
    int N = cfg_.num_samples > 0 ? cfg_.num_samples : 1;

    // 1. Draw N independent n x k standard Gaussian matrices
    Tensor G = math::random_normal({N, n_, k_}, cfg_.dtype);

    // 2. The thin QR decomposition of a standard Gaussian matrix
    //    yields a uniformly distributed orthogonal k-frame in R^n.
    return thin_qr(G);
}

// =============================================================================
// O(k³) spectral matrix exponential for the shape s (Batched Tensor Graph)
// =============================================================================

StiefelGaussianSampler::ShapeBlocks
    StiefelGaussianSampler::compute_shape_blocks(const std::vector<double>& flat_theta, int k)
{
    int N = cfg_.num_samples;
    if (N <= 0) N = 1;

    // Reshape the flat std::vector into a batched Tensor of shape [N, k]
    Tensor theta_vec = vec_to_tensor(flat_theta);
    Tensor theta     = math::reshape(theta_vec, {N, k});

    // --- Draw Ω_int ∈ so(k) (Batched [N, k, k]) ---
    Tensor omega = draw_omega_int(k);

    // --- Draw V_right ∈ O(k) (Batched [N, k, k]) ---
    Tensor vr = draw_v_right(k);

    // --- B = diag(θ) · V_right^T  ([N, k, k]) ---
    Tensor theta_col = math::expand_dims(theta, {2});               // [N, k] -> [N, k, 1]
    Tensor VrT       = math::transpose(vr, {0, 2, 1});              // [N, k, k]
    Tensor B         = math::multiply(theta_col, VrT);              // Broadcasts to [N, k, k]

    // --- V_active = [[Ω, -B^T], [B, 0]] ∈ so(2k) (Batched [N, 2k, 2k]) ---
    Tensor neg_BT  = math::multiply(math::transpose(B, {0, 2, 1}),
                                    Tensor(-1.0f, cfg_.dtype));
    Tensor zeros_k = math::full({N, k, k}, 0.0f, cfg_.dtype);

    // Concatenate along the column axis (2) then the row axis (1)
    Tensor top     = math::concatenate({omega, neg_BT}, 2);         // [N, k, 2k]
    Tensor bot     = math::concatenate({B, zeros_k}, 2);            // [N, k, 2k]
    Tensor V_act   = math::concatenate({top, bot}, 1);              // [N, 2k, 2k]

    // --- Θ_active = exp(V_active) ∈ SO(2k) ---
    // Computes the batched matrix exponential using a global norm to avoid GPU warp divergence.
    Tensor Theta_act = math::matrix_exp(V_act);

    // --- Extract sub-blocks across the batch ---
    // Axis 1 is rows, Axis 2 is cols for shape [N, 2k, 2k]
    Tensor Theta11 = math::slice(math::slice(Theta_act, 0, k, 1), 0, k, 2);
    Tensor Theta21 = math::slice(math::slice(Theta_act, k, 2 * k, 1), 0, k, 2);

    Tensor Theta12 = math::slice(math::slice(Theta_act, 0, k, 1), k, 2 * k, 2);
    Tensor Theta22 = math::slice(math::slice(Theta_act, k, 2 * k, 1), k, 2 * k, 2);

    return {Theta11, Theta21, Theta12, Theta22};
}

// =============================================================================
// Helpers (Batched Tensor Operations)
// =============================================================================

Tensor StiefelGaussianSampler::draw_omega_int(int k) {
    int N = cfg_.num_samples > 0 ? cfg_.num_samples : 1;
    const double sigma = 1.0 / std::sqrt(2.0 * alpha_);

    Tensor G    = math::random_normal({N, k, k}, cfg_.dtype);
    Tensor GT   = math::transpose(G, {0, 2, 1}); // 3D Transpose
    Tensor skew = math::subtract(G, GT);

    return math::multiply(skew, Tensor(static_cast<float>(sigma / std::sqrt(2.0)), cfg_.dtype));
}

Tensor StiefelGaussianSampler::draw_v_right(int k) {
    int N = cfg_.num_samples > 0 ? cfg_.num_samples : 1;

    Tensor G = math::random_normal({N, k, k}, cfg_.dtype);
    auto [Q, R] = math::qr(G);
    //math::eval(Q);
    //math::eval(R);

    // Backend-agnostic diagonal extraction: Mask R with a batched Identity matrix
    Tensor I = math::expand_dims(math::eye(k, cfg_.dtype), {0}); // [1, k, k]
    Tensor R_masked = math::multiply(R, I);                      // [N, k, k]

    // Summing across columns (axis 2) collapses it to the diagonals safely
    Tensor R_diag = math::sum(R_masked, {2});                    // [N, k]
    Tensor sgn    = math::sign(R_diag);                          // [N, k]

    // Expand dims to [N, 1, k] to broadcast column multiplication across Q [N, k, k]
    return math::multiply(Q, math::expand_dims(sgn, {1}));
}

Tensor StiefelGaussianSampler::thin_qr(const Tensor& a) {
    const int n_cols = a.shape().back();

    // Dynamically detect column axis (axis 1 for 2D, axis 2 for 3D tensors)
    int col_axis = a.shape().size() - 1;

    //math::set_default_device_cpu();
    auto [Q_full, R] = math::qr(a);
    //math::eval(Q_full);
    //math::eval(R);

    // Slice off the extra columns from the full Q matrices
    return math::slice(Q_full, 0, n_cols, col_axis);
}

Tensor StiefelGaussianSampler::vec_to_tensor(const std::vector<double>& v) {
    std::vector<float> f32(v.begin(), v.end());
    return math::array(f32, {static_cast<int>(f32.size())}, cfg_.dtype);
}

// =============================================================================
// High-α Phase I: Tangent-space principal angle sampler
// =============================================================================
//
// When α ≥ kHighAlphaThreshold, the stationary density concentrates so tightly
// around the consensus point (X̂) that the Stiefel manifold V(n, k) is locally
// flat. In this limit, the principal angles θᵢ are sufficiently small
// that we can apply the approximation sin(θ) ≈ θ.
//
// 1. Geometric Reduction:
//    The Weyl density kernel for V(n, k), which governs eigenvalue repulsion,
//    simplifies from a trigonometric product to an algebraic one:
//      w(θ) ≈ (∏ |θᵢ² - θⱼ²|) · ∏ |θᵢ|^(n-2k)
//
//    This simplified form is identically the joint eigenvalue density of a
//    Wishart matrix (Laguerre Orthogonal Ensemble)
//
// 2. Sampling Logic:
//    Rather than running HMC on the curved Weyl chamber, we sample directly
//    in the tangent space reductive complement m:
//
//    a. Draw G ∈ ℝ^{(n-k)×k}, where Gᵢⱼ ~ N(0, 1).
//       For "fat" frames (n < 2k), we utilize geometric duality to sample
//       G ∈ ℝ^{k_eff×(n-k_eff)} instead.
//
//    b. Form the symmetric PSD matrix M = GᵀG (or GGᵀ). The eigenvalues
//       of M represent the squared singular values of G.
//
//    c. The principal angles θ are extracted as the square roots of these
//       eigenvalues, scaled by the temperature variance (1/√(2α)).
//
// 3. Efficiency:
//    This method replaces O(G) numerical spline inversions with a single
//    O(k³) eigvalsh call per batch[cite: 1055, 1056]. It automatically
//    enforces the Dyson-gas repulsion and boundary interactions (n-2k)
//    through the inherent geometry of the Gaussian matrix.
//
std::vector<double> StiefelGaussianSampler::sample_angles_tangent() {
    int N = cfg_.num_samples > 0 ? cfg_.num_samples : 1;

    // The active off-diagonal block dimensions in the tangent space m
    int d_max = n_ - k_eff_;
    int d_min = k_eff_;

    // Step 1: Draw G ~ N(0, 1) of shape [N, d_max, d_min]
    Tensor G = math::random_normal({N, d_max, d_min}, cfg_.dtype);

    // Step 2: Compute M = G^T * G -> [N, d_min, d_min] symmetric PSD matrix
    Tensor GT = math::transpose(G, {0, 2, 1});
    Tensor M  = math::matmul(GT, G);

    // Step 3: Extract eigenvalues (squared singular values of G)
    // eigvalsh uses efficient tridiagonalization, much faster than full SVD
    Tensor eigvals = math::eigvalsh(M);
    math::eval(eigvals);

    // Step 4: Scale and extract roots
    float scale = static_cast<float>(1.0 / (2.0 * alpha_));
    std::vector<double> flat_theta(static_cast<std::size_t>(N * d_min));

    for (int n = 0; n < N; ++n) {
        Tensor En = math::slice(eigvals, n, n + 1, 0);              // [1, d_min]
        for (int j = 0; j < d_min; ++j) {
            Tensor v  = math::slice(En, j, j + 1, 1);               // [1, 1]
            double ev = math::to_double(v);

            // Protect against float precision dropping slightly below 0
            flat_theta[static_cast<std::size_t>(n * d_min + j)] =
                std::sqrt(ev > 0.0 ? ev * scale : 0.0);
        }
    }

    return flat_theta;
}

} // namespace sampler