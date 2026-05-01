#pragma once

#include "../angle_sampler_hmc.hpp"
#include <isomorphism/tensor.hpp>
#include <isomorphism/math.hpp>
#include <vector>

namespace sampler {

// Samples the principal angles θ = (θ₁, …, θ_k) from the Stiefel Weyl chamber
//
//   π > θ₁ > θ₂ > … > θ_k > 0
//
// under the exact joint density for V(n, k) with concentration α = λ/δ²:
//
//   p(θ) ∝ ∏ᵢ |sin θᵢ|^(n−2k)  ×  ∏_{i<j} |sin²θᵢ − sin²θⱼ|
//         ×  exp(−α ‖θ‖²)
//
// Requires n ≥ 2k (tall-and-skinny / dual tall-and-skinny regime).
//
// Inherits the full parallel HMC machinery from AngleSamplerHMC<>.
// Only compute_log_p_grad() is specialised here.
class PrincipalAngleSampler : public AngleSamplerHMC<PrincipalAngleSampler> {
    using Base = AngleSamplerHMC<PrincipalAngleSampler>;

public:
    // Extends the shared HMC config with the Stiefel-specific ambient dimension n.
    struct Config {
        int      n;                       // Ambient dimension (must satisfy n >= 2k)
        int      k;                       // Number of principal angles to sample
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

    explicit PrincipalAngleSampler(Config cfg);

    // --- forwarded accessors -------------------------------------------------
    int n() const { return n_; }
    int k() const { return m(); }   // delegates to base m()

    // Stiefel-specific log p(θ) and gradient (called by the base HMC engine).
    // scratch1 = sin²(θ),  scratch2 = sin(2θ).
    double compute_log_p_grad(const double* theta,
                              double*       grad,
                              double*       scratch1,
                              double*       scratch2,
                              bool          calc_log_p) const;

    // Kept for interface compatibility (no-op in batched mode).
    void hmc_step() {}

private:
    int    n_;    // ambient dimension
    double bnd_;  // n − 2k  (exponent of the per-angle boundary term)

    bool is_valid(const double* theta) const;

    // Translate the domain-specific Config into the base HMC Config.
    static Base::Config to_base_config(const Config& c) {
        Base::Config b;
        b.m              = c.k;
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
