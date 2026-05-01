#pragma once

#include "sampler/isotropic/so_angle_sampler.hpp"
#include <isomorphism/tensor.hpp>
#include <isomorphism/math.hpp>
#include <memory>
#include <random>
#include <vector>

namespace sampler {
namespace anisotropic {

// Phase I of Algorithm 4.8: sample the marginal angle vector
//   θ = (θ₁, …, θₘ) ∈ ℝᵐ,  m = ⌊d/2⌋
// from the isotropic upper-bound envelope via HMC:
//
//   p_upper(θ) ∝ w(θ) exp(−γ_min ‖θ‖²)
//
// where w(θ) is the SO(d) Weyl density and γ_min is the minimum eigenvalue
// of the anisotropic precision tensor Γ ≻ 0.
//
// This density is identical to the isotropic SO(d) Gaussian angle density
// (SOdAngleSampler) with α = γ_min, so Phase I simply delegates to it.
//
// Phase II (SOdHypersphereSamplerCPU or SOdHypersphereSamplerGPU) corrects
// the envelope samples to the true anisotropic target via column-level
// rejection on shrinking hyperspheres.
class SOdAnisotropicAngleSampler {
public:
    struct Config {
        int      d;
        int      num_chains     = 8;
        int      num_threads    = 1;
        int      burn_in        = 2000;
        int      leapfrog_steps = 5;
        double   init_epsilon   = 1e-4;
        double   target_accept  = 0.65;
        double   mass_high      = 1.0;
        double   mass_low       = 0.05;
        int      thinning       = 1;
        uint64_t seed           = static_cast<uint64_t>(std::random_device{}());
    };

    // gamma_flat: row-major d×d symmetric positive-definite precision matrix Γ.
    SOdAnisotropicAngleSampler(const std::vector<double>& gamma_flat, Config cfg);

    // Draw num_samples angle vectors.  Returns flat layout [num_samples * m]
    // where sample n occupies indices [n*m, (n+1)*m).
    std::vector<double> sample_angles(int num_samples);

    double gamma_min()       const { return gamma_min_; }
    double acceptance_rate() const;
    int    d()               const { return d_; }
    int    m()               const { return m_; }

    void update_gamma(const std::vector<double>& gamma_flat, int burn_in_steps = 500);

private:
    int    d_, m_;
    double gamma_min_;
    Config cfg_;
    std::vector<std::unique_ptr<SOdAngleSampler>> inners_;

    double compute_gamma_min(const std::vector<double>& gamma_flat) const;
    void   rebuild(const std::vector<double>& gamma_flat, int burn_in_steps);

    SOdAngleSampler::Config make_inner_config(int burn_in_steps) const;
};

} // namespace anisotropic
} // namespace sampler
