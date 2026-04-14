#include "sampler/stiefel_gaussian_sampler.hpp"

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

    // Explicitly pass the top-level batch size (N) down to the underlying HMC sampler.
    acfg.num_chains = cfg_.num_samples;

    angle_sampler_ = std::make_unique<PrincipalAngleSampler>(acfg);
}

void StiefelGaussianSampler::set_config(Config cfg) {
    cfg_   = cfg;
    alpha_ = cfg_.alpha;
    rebuild_angle_sampler();
}

// =============================================================================
// Public sample() – Batched Tensor Generation
// =============================================================================

Tensor StiefelGaussianSampler::sample() {
    int N = cfg_.num_samples;
    if (N <= 0) N = 1;

    // 1. Fetch the N independent sets of principal angles as a flat array.
    std::vector<double> flat_theta = angle_sampler_->sample_angles();

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

    auto [Q_full, R] = math::qr(a);

    // Slice off the extra columns from the full Q matrices
    return math::slice(Q_full, 0, n_cols, col_axis);
}

Tensor StiefelGaussianSampler::vec_to_tensor(const std::vector<double>& v) {
    std::vector<float> f32(v.begin(), v.end());
    return math::array(f32, {static_cast<int>(f32.size())}, cfg_.dtype);
}

} // namespace sampler