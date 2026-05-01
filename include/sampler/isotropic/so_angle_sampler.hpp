#pragma once

#include "../angle_sampler_hmc.hpp"
#include <isomorphism/tensor.hpp>
#include <isomorphism/math.hpp>
#include <vector>

namespace sampler {

// Samples the rotation angles θ = (θ₁, …, θₘ) from the SO(d) Weyl chamber
//
//   π > θ₁ > θ₂ > … > θₘ > 0,   m = ⌊d/2⌋
//
// under the exact joint invariant density for the rotation synchronization
// process with concentration α = λ/δ²:
//
//   p(θ) ∝ ∏_{i<j} sin²((θᵢ−θⱼ)/2) sin²((θᵢ+θⱼ)/2)
//         × [∏ᵢ sin²(θᵢ/2)]^{d mod 2}
//         × exp(−α Σᵢ θᵢ²)
//
// Equivalently, using the identity
//   sin((a−b)/2) sin((a+b)/2) = (cos b − cos a)/2:
//
//   p(θ) ∝ ∏_{i<j} (cos θⱼ − cos θᵢ)²
//         × [∏ᵢ (1 − cos θᵢ)/2]^{d mod 2}
//         × exp(−α Σᵢ θᵢ²)
//
// This is Phase I of Algorithm 4.3.  The full SO(d) Gaussian sampler
// (Phase I + Phase II) is implemented in SOdGaussianSampler.
//
// Inherits the full parallel HMC machinery from AngleSamplerHMC<>.
// Only compute_log_p_grad() is specialised here.
class SOdAngleSampler : public AngleSamplerHMC<SOdAngleSampler> {
    using Base = AngleSamplerHMC<SOdAngleSampler>;

public:
    // Extends the shared HMC config with d (full SO(d) dimension).
    struct Config {
        int      d;                       // Full dimension (m = d/2 angles)
        double   alpha        = 1.0;
        int      num_chains   = 1;
        int      num_threads  = 1;        // per-sampler thread budget (Layer 1)
        int      burn_in      = 2000;
        int      leapfrog_steps = 5;
        double   init_epsilon = 1e-4;
        double   target_accept = 0.65;
        double   mass_high    = 1.0;
        double   mass_low     = 0.05;
        int      thinning     = 1;
        uint64_t seed         = 0;
    };

    explicit SOdAngleSampler(Config cfg);

    int d() const { return d_; }
    // m() is inherited from AngleSamplerHMC base

    // SO(d)-specific log p(θ) and its gradient (called by the base HMC engine).
    // scratch1[i] = cos θᵢ,   scratch2[i] = sin θᵢ.
    double compute_log_p_grad(const double* theta,
                              double*       grad,
                              double*       scratch1,
                              double*       scratch2,
                              bool          calc_log_p) const;

private:
    int  d_;
    bool odd_;   // d % 2 == 1 (adds per-angle boundary term)

    static Base::Config to_base_config(const Config& c) {
        Base::Config b;
        b.m              = c.d / 2;
        b.alpha          = c.alpha;
        b.num_chains     = c.num_chains;
        b.num_threads    = c.num_threads;
        b.burn_in        = c.burn_in;
        b.leapfrog_steps = c.leapfrog_steps;
        b.init_epsilon   = c.init_epsilon;
        b.target_accept  = c.target_accept;
        b.mass_high      = c.mass_high;
        b.mass_low       = c.mass_low;
        b.thinning       = c.thinning;
        b.seed           = c.seed;
        return b;
    }
};

} // namespace sampler
