#pragma once

#include "sampler_base.hpp"
#include "so_angle_sampler.hpp"
#include <isomorphism/tensor.hpp>
#include <isomorphism/math.hpp>
#include <memory>
#include <vector>

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

        // HMC configuration (d, alpha, and num_chains are set automatically).
        SOdAngleSampler::Config angle_cfg;
    };

    // m_hat: d × d consensus rotation matrix, must satisfy det(M̂) = 1.
    SOdGaussianSampler(isomorphism::Tensor m_hat, int d, Config cfg);

    // Generates N samples.
    // If num_samples > 1: returns [N, d, d].
    // If num_samples == 1: returns [d, d].
    isomorphism::Tensor sample() override;

    void set_config(Config cfg);
    void rebuild_angle_sampler();

    int    d()            const { return d_; }
    int    m()            const { return m_; }
    int    num_samples()  const { return cfg_.num_samples; }
    double alpha()        const { return alpha_; }
    double angle_acceptance_rate() const { return angle_sampler_->acceptance_rate(); }

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

    // Draw N independent Haar-uniform O(d) matrices via QR.
    // Returns shape [N, d, d].
    isomorphism::Tensor draw_haar_od();

    isomorphism::Tensor vec_to_tensor(const std::vector<double>& v);
};

} // namespace sampler
