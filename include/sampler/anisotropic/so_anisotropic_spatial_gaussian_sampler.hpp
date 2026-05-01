#pragma once

#include "so_anisotropic_angle_sampler.hpp"
#include "so_hypersphere_sampler_cpu_spatial.hpp"
#include "sampler/sampler_base.hpp"
#include <isomorphism/tensor.hpp>
#include <isomorphism/math.hpp>
#include <memory>
#include <vector>

namespace sampler {
namespace anisotropic {

// Algorithm 4.8: Anisotropic Spectral Generative Sampling on SO(d)
// — spatial-precision variant.
//
// Identical to SOdAnisotropicGaussianSampler except that the precision input
// is the d×d spatial matrix Γ_spatial (only the diagonal matters), and Phase II
// uses SOdHypersphereSamplerCPUSpatial so no D×D Lie algebra tensor is required
// or constructed.
//
// Use this variant when:
//   • only the spatial d×d precision is available (e.g. CMA-ES path), or
//   • a lighter, faster Phase II is acceptable in exchange for slightly less
//     exact anisotropic rejection (Guide and Judge stay consistent with the
//     spatial approximation throughout).
//
// For the full-precision variant that constructs and uses the D×D Lie algebra
// tensor in Phase II, see SOdAnisotropicGaussianSampler.
class SOdAnisotropicSpatialGaussianSampler : public SamplerBase {
public:
    struct Config {
        int    num_samples = 1;
        isomorphism::DType dtype = isomorphism::DType::Float32;

        SOdAnisotropicAngleSampler::Config      angle_cfg;
        SOdHypersphereSamplerCPUSpatial::Config cpu_cfg;
    };

    // m_hat  : d×d consensus rotation (det = +1).
    // gamma  : row-major d×d symmetric positive-definite spatial precision Γ.
    //          Only the diagonal entries gamma[i*d+i] are consumed by Phase II;
    //          the full d×d matrix is forwarded to the Phase I angle sampler.
    SOdAnisotropicSpatialGaussianSampler(isomorphism::Tensor        m_hat,
                                         int                        d,
                                         const std::vector<double>& gamma,
                                         Config                     cfg);

    // Generate num_samples draws from the current distribution.
    // Returns [N, d, d] when N > 1, [d, d] when N == 1.
    isomorphism::Tensor sample() override;

    void set_m_hat(isomorphism::Tensor m_hat);

    // gamma: row-major d×d spatial precision (same layout as constructor).
    void update_gamma(const std::vector<double>& gamma);

    int    d()         const { return d_; }
    int    m()         const { return m_; }
    double gamma_min() const { return gamma_min_; }
    double angle_acceptance_rate() const;

private:
    isomorphism::Tensor m_hat_;
    int    d_, m_;
    std::vector<double> gamma_;     // d×d spatial, full matrix
    double gamma_min_;
    Config cfg_;

    std::unique_ptr<SOdAnisotropicAngleSampler>      angle_sampler_;
    std::unique_ptr<SOdHypersphereSamplerCPUSpatial> cpu_sampler_;

    void rebuild_samplers();

    isomorphism::Tensor build_canonical_rotation(const std::vector<double>& flat_theta,
                                                  int N) const;
};

} // namespace anisotropic
} // namespace sampler
