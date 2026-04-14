#pragma once

#include "sampler_base.hpp"
#include "thread_config.hpp"
#include "xoshiro.hpp"
#include <isomorphism/tensor.hpp>
#include <isomorphism/math.hpp>
#include <cstdint>
#include <random>
#include <vector>

namespace sampler {

// =============================================================================
// CRTP base class – Hamiltonian Monte Carlo over the Weyl chamber
//
//   π > θ₁ > θ₂ > … > θₘ > 0
//
// Implements the full parallel HMC machinery (burn-in, leapfrog, dual-average
// step-size adaptation, batched sampling).  The only thing a derived class
// must provide is the target log-density and its gradient:
//
//   double compute_log_p_grad(
//       const double* theta,    // current angles [m]
//       double*       grad,     // output gradient [m]
//       double*       scratch1, // scratch buffer [m]  (use as needed)
//       double*       scratch2, // scratch buffer [m]
//       bool          calc_log_p) const;
//
// Construction is two-phase so that derived-class data is fully initialised
// before any CRTP dispatch occurs:
//
//   Derived::Derived(Config cfg)
//       : AngleSamplerHMC<Derived>(to_base_config(cfg))  // allocate buffers
//       , my_data_(...)                                   // initialise derived
//   { this->start_sampler(); }                           // init chains + burn-in
//
// Template method bodies and explicit instantiations live in
// src/angle_sampler_hmc.cpp, keeping this header declaration-only.
// =============================================================================

template <typename Derived>
class AngleSamplerHMC : public SamplerBase {
public:
    struct Config {
        int      m             = 1;
        double   alpha         = 1.0;
        int      num_chains    = 1;
        int      num_threads   = 1;   // per-sampler thread budget (Layer 1)
        int      burn_in       = 2000;
        int      leapfrog_steps = 5;
        double   init_epsilon  = 1e-4;
        double   target_accept = 0.65;
        double   mass_high     = 1.0;
        double   mass_low      = 0.05;
        int      thinning      = 1;
        uint64_t seed          = 0;
    };

    int    m()            const;
    int    num_chains()   const;
    bool   is_warmed_up() const;
    double acceptance_rate() const;

    std::vector<double>  sample_angles();
    isomorphism::Tensor  sample() override;

protected:
    Config cfg_;
    int    num_chains_    = 0;
    bool   warmed_up_     = false;

    std::vector<double> mass_;
    std::vector<double> inv_mass_;
    size_t              stride_ = 0;
    std::vector<double> flat_thetas_;
    std::vector<double> flat_grads_;

    struct alignas(64) ChainState {
        double*            theta          = nullptr;
        double*            grad           = nullptr;
        double             log_p          = 0.0;
        double             epsilon        = 0.0;
        int                step_count     = 0;
        int                accepted_count = 0;
        Xoshiro256PlusPlus rng;
    };

    std::vector<ChainState> chains_;

    explicit AngleSamplerHMC(Config cfg);
    void start_sampler();

private:
    void init_chains();
    void run_burn_in();

    bool hmc_trajectory(
        int c,
        std::vector<double>& p,
        std::vector<double>& theta_p,
        std::vector<double>& p_p,
        std::vector<double>& final_grad,
        std::vector<double>& prop_grad,
        std::vector<double>& s1,
        std::vector<double>& s2,
        std::normal_distribution<double>&       norm01,
        std::uniform_real_distribution<double>& unif01);
};

} // namespace sampler
