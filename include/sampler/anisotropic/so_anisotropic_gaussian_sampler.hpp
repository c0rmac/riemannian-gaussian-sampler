#pragma once

#include "so_anisotropic_angle_sampler.hpp"
#include "so_hypersphere_sampler_cpu.hpp"
#include "sampler/sampler_base.hpp"
#include <isomorphism/tensor.hpp>
#include <isomorphism/math.hpp>
#include <memory>
#include <vector>

namespace sampler {
namespace anisotropic {

// Algorithm 4.8: Exact Anisotropic Spectral Generative Sampling on SO(d)
// — full-precision (D×D Lie algebra) variant.
//
// Samples i.i.d. X ∈ SO(d) from
//   μ_∞(X) ∝ exp(−⟨Log_{M̂}(X), Γ Log_{M̂}(X)⟩_g)
//
// where M̂ ∈ SO(d) is the consensus rotation and Γ ≻ 0 is the D×D Lie algebra
// precision tensor (D = d*(d-1)/2).
//
// --- Pipeline ---
//
// Phase I  (SOdAnisotropicAngleSampler, HMC):
//   The D×D precision is projected to a d×d spatial diagonal surrogate to derive
//   γ_min and steer the HMC envelope sampler.
//
// Phase II (SOdHypersphereSamplerCPU):
//   The full D×D matrix is passed directly so that the Judge evaluates the exact
//   Lie algebra energy in the IMH acceptance ratio, guaranteeing zero-bias sampling.
//   The Guide still uses the least-squares spatial surrogate internally.
//
// Phase III (geometric recombination):
//   Build Θ = diag(R(θ₁),…,R(θₘ),[1 if d odd])  from the accepted angles.
//   Return  X = Q Θ Qᵀ M̂.
//
// For a lighter variant that avoids the D×D tensor entirely, see
// SOdAnisotropicSpatialGaussianSampler.
class SOdAnisotropicGaussianSampler : public SamplerBase {
public:
    struct Config {
        int    num_samples = 1;
        isomorphism::DType dtype = isomorphism::DType::Float32;

        SOdAnisotropicAngleSampler::Config angle_cfg;
        SOdHypersphereSamplerCPU::Config   cpu_cfg;
    };

    // m_hat    : d×d consensus rotation (det = +1).
    // gamma    : row-major D×D symmetric positive-definite Lie algebra precision Γ,
    //            where D = d*(d-1)/2.
    SOdAnisotropicGaussianSampler(isomorphism::Tensor        m_hat,
                                   int                        d,
                                   const std::vector<double>& gamma,
                                   Config                     cfg);

    // Generate num_samples draws from the current distribution.
    // Returns [N, d, d] when N > 1, [d, d] when N == 1.
    isomorphism::Tensor sample() override;

    void set_m_hat(isomorphism::Tensor m_hat);

    // gamma: row-major D×D Lie algebra precision (same layout as constructor).
    void update_gamma(const std::vector<double>& gamma);

    int    d()         const { return d_; }
    int    m()         const { return m_; }
    double gamma_min() const { return gamma_min_; }
    double angle_acceptance_rate() const;

private:
    isomorphism::Tensor m_hat_;
    int    d_, m_;
    std::vector<double> gamma_;   // D×D Lie algebra precision
    double gamma_min_;
    Config cfg_;

    std::unique_ptr<SOdAnisotropicAngleSampler> angle_sampler_;
    std::unique_ptr<SOdHypersphereSamplerCPU>   cpu_sampler_;

    void rebuild_samplers();

    isomorphism::Tensor build_canonical_rotation(const std::vector<double>& flat_theta,
                                                  int N) const;
};

} // namespace anisotropic
} // namespace sampler
