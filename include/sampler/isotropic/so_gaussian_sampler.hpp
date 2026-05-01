#pragma once

#include "../sampler_base.hpp"
#include "so_angle_sampler.hpp"
#include <isomorphism/tensor.hpp>
#include <isomorphism/math.hpp>
#include <memory>
#include <vector>
#include <random>

namespace sampler {

// Samples exactly from the isotropic Gaussian distribution on SO(d) centred at
// a consensus rotation M̂ ∈ SO(d):
//
//   μ_∞(X) ∝ exp(−α · d_g²(X, M̂))   where α = λ/δ²
//
// This implements Algorithm 4.3 from the companion paper.
//
// --- Algorithm overview ---
//
// Phase I  (offline, via SOdAngleSampler):
//   Sample rotation angles θ = (θ₁, …, θₘ) ∈ ℝᵐ, m = ⌊d/2⌋, from the
//   SO(d) Weyl-chamber density using parallel HMC.
//
// Phase II (online, per sample):
//   1. Build the canonical block-diagonal rotation
//        Θ = diag(R(θ₁), …, R(θₘ), [1 if d odd])  ∈ SO(d)
//      using exact 2×2 Givens blocks (no matrix exponential needed).
//   2. Draw Q ∈ O(d) from the Haar measure via thin QR of a d×d Gaussian.
//      (det(Q·Θ·Qᵀ) = det(Q)² det(Θ) = 1, so the result is always in SO(d).)
//   3. Form the random conjugate rotation:  X = Q Θ Qᵀ M̂.
//
// The O(d) complexity of the manifest lift (vs. O(d³) for general matrix
// exponential) is the key computational advantage over the Stiefel case.
class SOdGaussianSampler : public SamplerBase {
public:
    struct Config {
        int    num_samples  = 1;
        double alpha        = 1.0;
        isomorphism::DType dtype = isomorphism::DType::Float32;

        // HMC configuration (d and alpha are set automatically; num_chains defaults to 8).
        SOdAngleSampler::Config angle_cfg;
    };

    // m_hat: d × d consensus rotation matrix, must satisfy det(M̂) = 1.
    SOdGaussianSampler(isomorphism::Tensor m_hat, int d, Config cfg);

    // Generates N samples.
    // If num_samples > 1: returns [N, d, d].
    // If num_samples == 1: returns [d, d].
    isomorphism::Tensor sample() override;

    void set_config(Config cfg);
    void set_m_hat(isomorphism::Tensor m_hat);
    void update_alpha(double alpha, int burn_in_steps = 500);
    void rebuild_angle_sampler();

    int    d()            const { return d_; }
    int    m()            const { return m_; }
    int    num_samples()  const { return cfg_.num_samples; }
    double alpha()        const { return alpha_; }
    double angle_acceptance_rate() const {
        return angle_sampler_ ? angle_sampler_->acceptance_rate() : -1.0;
    }

    // Draw N independent Haar-uniform O(d) matrices via QR.
    // Returns shape [N, d, d].
    isomorphism::Tensor draw_haar_od();

    // α ≥ this threshold activates the tangent-space sampler (replaces HMC Phase I).
    static constexpr double kHighAlphaThreshold = 1e9;
    //static constexpr double kHighAlphaThreshold = 0.0;

private:
    isomorphism::Tensor m_hat_;
    int    d_, m_;
    double alpha_;
    Config cfg_;

    std::unique_ptr<SOdAngleSampler> angle_sampler_;

    // Build the batched canonical block-diagonal rotation Θ ∈ SO(d)^N.
    // flat_theta: length N * m (one angle vector per sample).
    // Returns shape [N, d, d].
    isomorphism::Tensor build_canonical_rotation(const std::vector<double>& flat_theta);

    isomorphism::Tensor vec_to_tensor(const std::vector<double>& v);

    // High-α Phase I: tangent-space Gaussian sampling (replaces HMC when
    // alpha >= kHighAlphaThreshold).  Generates a d×d skew-symmetric Gaussian
    //   Ω = (A − Aᵀ) / √(2α),   A_ij ~ N(0,1)
    // then extracts the m = ⌊d/2⌋ principal angles from its singular values.
    // Singular values of a skew-symmetric matrix come in pairs (θⱼ, θⱼ), so
    // the m angles are S[n, 0], S[n, 2], …, S[n, 2(m−1)] for each sample n.
    // Returns flat layout flat_theta[n*m + j] = θⱼ for sample n, identical to
    // angle_sampler_->sample_angles().
    std::vector<double> sample_angles_tangent();

    isomorphism::Tensor sample_so2();
};

} // namespace sampler
