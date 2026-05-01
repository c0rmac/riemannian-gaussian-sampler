#include "sampler/isotropic/so_gaussian_sampler.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace sampler {

namespace math = isomorphism::math;
using Tensor   = isomorphism::Tensor;
using DType    = isomorphism::DType;

// =============================================================================
// Construction
// =============================================================================

SOdGaussianSampler::SOdGaussianSampler(Tensor m_hat, int d, Config cfg)
    : m_hat_(m_hat), d_(d), m_(d / 2), cfg_(cfg)
{
    assert(d >= 2);
    alpha_ = cfg_.alpha;
    rebuild_angle_sampler();
}

void SOdGaussianSampler::rebuild_angle_sampler() {
    if (alpha_ >= kHighAlphaThreshold) {
        angle_sampler_.reset();   // tangent-space sampler needs no HMC chain
        return;
    }
    auto acfg        = cfg_.angle_cfg;
    acfg.d           = d_;
    acfg.alpha       = alpha_;
    angle_sampler_   = std::make_unique<SOdAngleSampler>(acfg);
}

void SOdGaussianSampler::set_config(Config cfg) {
    cfg_   = cfg;
    alpha_ = cfg_.alpha;
    rebuild_angle_sampler();
}

void SOdGaussianSampler::set_m_hat(isomorphism::Tensor m_hat) {
    m_hat_ = std::move(m_hat);
}

void SOdGaussianSampler::update_alpha(double alpha, int burn_in_steps) {
    alpha_     = alpha;
    cfg_.alpha = alpha;
    if (alpha_ >= kHighAlphaThreshold) {
        angle_sampler_.reset();
    } else if (!angle_sampler_) {
        rebuild_angle_sampler();   // transitioning down from high-α
    } else {
        angle_sampler_->set_alpha(alpha, burn_in_steps);
    }
}

// =============================================================================
// Public sample() – Batched Tensor Generation
// =============================================================================

Tensor SOdGaussianSampler::sample() {
    // Fast path for SO(2)
    if (d_ == 2) {
        return sample_so2();
    }

    int N = cfg_.num_samples;
    if (N <= 0) N = 1;

    auto t_start = std::chrono::high_resolution_clock::now();
    // 1. Phase I: fetch N independent angle vectors.
    //    High-α (≥ kHighAlphaThreshold): tangent-space Gaussian sampling.
    //    Otherwise: HMC on the Weyl chamber.
    std::vector<double> flat_theta = (alpha_ >= kHighAlphaThreshold)
        ? sample_angles_tangent()
        : angle_sampler_->sample_angles(N);

    auto t_end = std::chrono::high_resolution_clock::now();

    //std::cout << " Phase 1 time: " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << "\n";

    // 2. Phase II: O(d) spectral lift per sample.

    auto t_start_2 = std::chrono::high_resolution_clock::now();
    // Build the canonical block-diagonal rotation Θ [N, d, d].
    // This uses exact 2×2 Givens blocks — no matrix exponential required.
    Tensor Theta = build_canonical_rotation(flat_theta);             // [N, d, d]

    // Draw Q ∈ O(d) from Haar measure [N, d, d].
    // Note: det(Q Θ Qᵀ) = det(Q)² · det(Θ) = 1, so the result is in SO(d).
    Tensor Q  = draw_haar_od();                                       // [N, d, d]
    Tensor QT = math::transpose(Q, {0, 2, 1});                       // [N, d, d]

    // g_tilde = Q Θ Qᵀ   [N, d, d]
    Tensor g_tilde = math::matmul(math::matmul(Q, Theta), QT);

    // X = g_tilde · M̂   [N, d, d]  (broadcast M̂ over the batch)
    Tensor m_hat_b = m_hat_;                 // [1, d, d]
    //Tensor m_hat_b = math::expand_dims(m_hat_, {0});                 // [1, d, d]
    Tensor X       = math::matmul(g_tilde, m_hat_b);                 // [N, d, d]

    math::eval(X);

    //std::cout << m_hat_ << std::endl;
    //std::cout << X << std::endl;

    auto t_end_2 = std::chrono::high_resolution_clock::now();

    //std::cout << " Phase 2 time: " << std::chrono::duration<double, std::milli>(t_end_2 - t_start_2).count() << "\n";

    // 3. Output formatting
    if (N == 1)
        return math::slice(X, 0, 1, 0);   // strip batch dim → [d, d]
    return X;
}

// =============================================================================
// Phase II helpers
// =============================================================================

// Build the batched canonical block-diagonal rotation.
//
// For d = 2m (even):
//   Θ = diag(R(θ₁), R(θ₂), …, R(θₘ))
//
// For d = 2m+1 (odd):
//   Θ = diag(R(θ₁), R(θ₂), …, R(θₘ), 1)
//
// where R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]].
//
// The matrix is assembled row by row using concatenation.
// No math::cos/sin needed: cosines and sines are computed on the CPU
// and loaded as tensors.
Tensor SOdGaussianSampler::build_canonical_rotation(
    const std::vector<double>& flat_theta)
{
    int N = cfg_.num_samples;
    if (N <= 0) N = 1;

    const bool odd  = (d_ % 2 == 1);
    const int  d    = d_;
    const int  m    = m_;

    // Precompute cos and sin on the CPU from the flat angle array [N * m].
    std::vector<float> cos_vals(N * m), sin_vals(N * m);
    for (int i = 0; i < N * m; i++) {
        cos_vals[i] = static_cast<float>(std::cos(flat_theta[i]));
        sin_vals[i] = static_cast<float>(std::sin(flat_theta[i]));
    }
    // cos_t, sin_t: [N, m]
    Tensor cos_t = math::array(cos_vals, {N, m}, cfg_.dtype);
    Tensor sin_t = math::array(sin_vals, {N, m}, cfg_.dtype);

    // Assemble Θ row-pair by row-pair.
    // Row  2j  :  [0…0 | cos_j, −sin_j | 0…0 (| 0 if odd)]
    // Row 2j+1 :  [0…0 | sin_j,  cos_j | 0…0 (| 0 if odd)]
    std::vector<Tensor> all_rows;
    all_rows.reserve(d);

    for (int j = 0; j < m; j++) {
        // Extract [N, 1] slices for cos and sin of angle j.
        // slice(tensor, start, stop, axis)
        Tensor c_j  = math::slice(cos_t, j, j + 1, 1);                   // [N, 1]
        Tensor s_j  = math::slice(sin_t, j, j + 1, 1);                   // [N, 1]
        Tensor ns_j = math::multiply(s_j,
                          Tensor(-1.0f, cfg_.dtype));                     // −sin_j

        // Expand to [N, 1, 1] for row construction.
        Tensor c_e  = math::expand_dims(c_j,  {2});                       // [N, 1, 1]
        Tensor s_e  = math::expand_dims(s_j,  {2});                       // [N, 1, 1]
        Tensor ns_e = math::expand_dims(ns_j, {2});                       // [N, 1, 1]

        // Build the two rows of block j.
        // row0 = [...zeros..., c_j, −s_j, ...zeros..., (0 if odd)]
        // row1 = [...zeros..., s_j,  c_j, ...zeros..., (0 if odd)]

        // Parts are collected and concatenated along the column axis (2).
        std::vector<Tensor> row0_parts, row1_parts;

        // Left zero padding: 2j zeros
        if (2 * j > 0) {
            Tensor lz = math::full({N, 1, 2 * j}, 0.0f, cfg_.dtype);
            row0_parts.push_back(lz);
            row1_parts.push_back(lz);
        }

        // The 2×2 block entries
        row0_parts.push_back(c_e);
        row0_parts.push_back(ns_e);
        row1_parts.push_back(s_e);
        row1_parts.push_back(c_e);   // cos_j appears on both diagonals

        // Right zero padding: d − 2(j+1) zeros (excluding the odd column)
        const int right = 2 * (m - j - 1);
        if (right > 0) {
            Tensor rz = math::full({N, 1, right}, 0.0f, cfg_.dtype);
            row0_parts.push_back(rz);
            row1_parts.push_back(rz);
        }

        // For odd d: one extra zero column before the last-row 1.
        if (odd) {
            Tensor oz = math::full({N, 1, 1}, 0.0f, cfg_.dtype);
            row0_parts.push_back(oz);
            row1_parts.push_back(oz);
        }

        // Concatenate parts → [N, 1, d] rows
        Tensor row0 = math::concatenate(row0_parts, 2);
        Tensor row1 = math::concatenate(row1_parts, 2);

        all_rows.push_back(row0);
        all_rows.push_back(row1);
    }

    // For odd d: add the final row [0, …, 0, 1].
    if (odd) {
        std::vector<Tensor> last_parts;
        if (d - 1 > 0)
            last_parts.push_back(math::full({N, 1, d - 1}, 0.0f, cfg_.dtype));
        last_parts.push_back(math::full({N, 1, 1}, 1.0f, cfg_.dtype));
        all_rows.push_back(math::concatenate(last_parts, 2));
    }

    // Stack all d row tensors along axis 1 → [N, d, d].
    return math::concatenate(all_rows, 1);
}

// Draw N independent Haar-uniform O(d) matrices via QR decomposition.
// The output lies in O(d) (det = ±1); however det(Q Θ Qᵀ) = det(Q)² det(Θ) = 1,
// so the final SO(d) sample is correct regardless of the sign of det(Q).
    /*
Tensor SOdGaussianSampler::draw_haar_od() {
    int N = cfg_.num_samples > 0 ? cfg_.num_samples : 1;

    Tensor G = math::random_normal({N, d_, d_}, cfg_.dtype);
    auto [Q, R] = math::qr(G);

    // Column-sign correction so R has positive diagonal (standard QR sign fix).
    Tensor I       = math::expand_dims(math::eye(d_, cfg_.dtype), {0}); // [1,d,d]
    Tensor R_masked = math::multiply(R, I);                              // [N,d,d]
    Tensor R_diag  = math::sum(R_masked, {2});                           // [N,d]
    Tensor sgn     = math::sign(R_diag);                                 // [N,d]

    return math::multiply(Q, math::expand_dims(sgn, {1}));              // [N,d,d]
}*/
    Tensor SOdGaussianSampler::draw_haar_od() {
        int N = cfg_.num_samples > 0 ? cfg_.num_samples : 1;
        Tensor G = math::random_normal({N, d_, d_}, cfg_.dtype);

        // 1. Standard QR (O(d^3))
        auto [Q, R] = math::qr(G);

        // 2. Extract diagonal signs (O(d))
        Tensor I        = math::expand_dims(math::eye(d_, cfg_.dtype), {0});
        Tensor R_diag   = math::sum(math::multiply(R, I), {2});
        Tensor sgn      = math::sign(R_diag);

        // 3. Compute det(Q_od) in O(d) using Householder parity
        // For d x d, Householder uses d-1 reflections. Parity = (-1)^(d-1)
        float parity = (d_ % 2 == 0) ? -1.0f : 1.0f;
        Tensor prod_sgn = math::prod(sgn, {1}); // [N]
        Tensor det_Q    = math::multiply(prod_sgn, Tensor(parity)); // [N]

        // 4. Construct SO(d) matrix (O(d^2))
        Tensor Q_od = math::multiply(Q, math::expand_dims(sgn, {1}));

        // If det is -1, flip the last column to force det = 1
        Tensor ones           = math::full({N, d_ - 1}, 1.0f, cfg_.dtype);
        Tensor col_correction = math::concatenate({ones, math::expand_dims(det_Q, {1})}, 1);

        return math::multiply(Q_od, math::expand_dims(col_correction, {1}));
    }

Tensor SOdGaussianSampler::vec_to_tensor(const std::vector<double>& v) {
    std::vector<float> f32(v.begin(), v.end());
    return math::array(f32, {static_cast<int>(f32.size())}, cfg_.dtype);
}

// =============================================================================
// High-α Phase I: tangent-space spectral sampler
// =============================================================================
//
// When α ≥ kHighAlphaThreshold the stationary density concentrates so tightly
// around the consensus point that the SO(d) manifold is locally flat.  We can
// sample directly in the tangent space 𝔰𝔬(d) rather than running HMC:
//
//   1. Draw A ∈ ℝ^{d×d},  A_{ij} ~ N(0,1).
//   2. Form the skew-symmetric matrix  Ω = (A − Aᵀ) / √(2α).
//   3. The eigenvalues of a real skew-symmetric matrix are purely imaginary,
//      appearing in conjugate pairs ±iθⱼ.  Equivalently, its singular values
//      come in identical pairs (θⱼ, θⱼ) in decreasing order.
//   4. Extract the m = ⌊d/2⌋ principal angles as the even-indexed singular
//      values: θⱼ = S[2j],  j = 0, …, m−1.
//   5. Feed these angles into the existing Phase II pipeline unchanged.
//
// The Dyson-gas repulsion between eigenvalues is automatically enforced by the
// geometry of the Gaussian Orthogonal Ensemble — no explicit Weyl-chamber
// rejection is needed.
std::vector<double> SOdGaussianSampler::sample_angles_tangent() {
    int N = cfg_.num_samples > 0 ? cfg_.num_samples : 1;

    // Steps 1–2: generate [N, d, d] scaled skew-symmetric Gaussian matrix
    //   Ω = (A − Aᵀ) / √(2α),   A_ij ~ N(0,1)
    Tensor A     = math::random_normal({N, d_, d_}, cfg_.dtype);     // [N, d, d]
    float  scale = static_cast<float>(1.0 / std::sqrt(2.0 * alpha_));
    Tensor Omega = math::multiply(
        math::subtract(A, math::transpose(A, {0, 2, 1})),
        Tensor(scale, cfg_.dtype));                                    // [N, d, d]

    // Step 3: B = Ω·Ωᵀ = −Ω²   (symmetric PSD, eigenvalues are θⱼ²)
    // eigvalsh is ~3–4× cheaper than SVD: no singular vectors, uses
    // tridiagonalisation (O(d³/3) vs O(4d³/3) for full SVD).
    Tensor B      = math::matmul(Omega, math::transpose(Omega, {0, 2, 1})); // [N,d,d]
    Tensor eigvals = math::eigvalsh(B);                                      // [N,d] ascending
    math::eval(eigvals);

    // Step 4: extract m principal angles.
    // eigvalsh returns ascending eigenvalues; paired values θⱼ² sit at the end.
    // For j = 0…m−1: θⱼ = √(eigvals[n, d−1−2j])
    //   j=0 → index d−1 (largest pair → θ₁, the biggest angle)
    //   j=m−1 → index d−1−2(m−1) = d−2m+1 ≥ 1  ✓ for both even and odd d
    std::vector<double> flat_theta(static_cast<std::size_t>(N * m_));
    for (int n = 0; n < N; ++n) {
        Tensor En = math::slice(eigvals, n, n + 1, 0);              // [1, d]
        for (int j = 0; j < m_; ++j) {
            int    idx = d_ - 1 - 2 * j;
            Tensor v   = math::slice(En, idx, idx + 1, 1);          // [1, 1]
            double ev  = math::to_double(v);
            flat_theta[static_cast<std::size_t>(n * m_ + j)] =
                std::sqrt(ev > 0.0 ? ev : 0.0);
        }
    }
    return flat_theta;
}

Tensor SOdGaussianSampler::sample_so2() {
    int N = cfg_.num_samples;
    if (N <= 0) N = 1;

    // 1. Phase I: Draw angles directly from 1D Truncated Normal
    // The target density is exp(-alpha * theta^2), which corresponds to
    // a Gaussian with variance sigma^2 = 1 / (2 * alpha).
    double stddev = 1.0 / std::sqrt(2.0 * alpha_);
    std::vector<double> flat_theta(N);

    // Thread-local generator for safety and speed
    static thread_local std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, stddev);

    const double PI = std::acos(-1.0);

    for (int i = 0; i < N; ++i) {
        double val;
        do {
            val = dist(gen);
        } while (val < -PI || val > PI); // Truncate to injectivity radius
        flat_theta[i] = val;
    }

    // 2. Phase II: Direct geometric recombination
    // Because SO(2) is commutative, Q * Theta * Q^T = Theta.
    // We completely skip draw_haar_od() and matmuls!
    Tensor g_tilde = build_canonical_rotation(flat_theta); // [N, 2, 2]

    // 3. Apply to consensus point
    Tensor m_hat_b = m_hat_;                           // [1, 2, 2]
    Tensor X       = math::matmul(g_tilde, m_hat_b);   // [N, 2, 2]

    math::eval(X);

    // 4. Output formatting
    if (N == 1)
        return math::slice(X, 0, 1, 0);   // strip batch dim → [2, 2]
    return X;
}

} // namespace sampler
