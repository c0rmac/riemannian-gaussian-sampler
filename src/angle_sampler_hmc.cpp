// Include derived class headers first so the compiler knows their full
// definitions (specifically their compute_log_p_grad signatures) before
// the explicit instantiations at the bottom of this file.
#include "sampler/principal_angle_sampler.hpp"
#include "sampler/so_angle_sampler.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>

namespace sampler {

// =============================================================================
// Phase-1 constructor – allocates shared buffers, no CRTP dispatch.
// =============================================================================

template <typename Derived>
AngleSamplerHMC<Derived>::AngleSamplerHMC(Config cfg) : cfg_(cfg) {
    assert(cfg_.m >= 1);
    assert(cfg_.alpha > 0.0);

    num_chains_ = std::max(1, cfg_.num_chains);
    stride_     = static_cast<size_t>(((cfg_.m + 7) / 8) * 8);

    flat_thetas_.assign(num_chains_ * stride_, 0.0);
    flat_grads_.assign(num_chains_ * stride_,  0.0);

    mass_.resize(cfg_.m);
    inv_mass_.resize(cfg_.m);
    for (int i = 0; i < cfg_.m; i++) {
        const double t = (cfg_.m == 1) ? 0.0
                       : static_cast<double>(i) / (cfg_.m - 1);
        mass_[i]     = cfg_.mass_high + t * (cfg_.mass_low - cfg_.mass_high);
        inv_mass_[i] = 1.0 / mass_[i];
    }

    chains_.resize(num_chains_);
}

// =============================================================================
// Accessors
// =============================================================================

template <typename Derived>
int AngleSamplerHMC<Derived>::m() const { return cfg_.m; }

template <typename Derived>
int AngleSamplerHMC<Derived>::num_chains() const { return num_chains_; }

template <typename Derived>
bool AngleSamplerHMC<Derived>::is_warmed_up() const { return warmed_up_; }

template <typename Derived>
double AngleSamplerHMC<Derived>::acceptance_rate() const {
    double total = 0.0;
    for (int c = 0; c < num_chains_; c++)
        if (chains_[c].step_count > 0)
            total += static_cast<double>(chains_[c].accepted_count)
                   / chains_[c].step_count;
    return num_chains_ > 0 ? (total / num_chains_) : 0.0;
}

// =============================================================================
// Phase-3 entry point – call from derived constructor after derived data is set.
// =============================================================================

template <typename Derived>
void AngleSamplerHMC<Derived>::start_sampler() {
    init_chains();
    run_burn_in();
}

// =============================================================================
// Chain initialisation – sets starting angles, computes initial log_p/grad.
// =============================================================================

template <typename Derived>
void AngleSamplerHMC<Derived>::init_chains() {
    const int m  = cfg_.m;
    const int nt = effective_num_threads(cfg_.num_threads);

    #pragma omp parallel num_threads(nt)
    {
        std::vector<double> ig(m), s1(m), s2(m);

        #pragma omp for schedule(guided)
        for (int c = 0; c < num_chains_; c++) {
            chains_[c].rng.seed(cfg_.seed + static_cast<uint64_t>(c));
            chains_[c].theta = &flat_thetas_[c * stride_];
            chains_[c].grad  = &flat_grads_[c * stride_];

            // Linearly spaced in (0.05, π/2 − 0.05), strictly decreasing.
            const double hi = M_PI / 2.0 - 0.05, lo = 0.05;
            for (int i = 0; i < m; i++) {
                const double t = (m == 1) ? 0.5
                               : static_cast<double>(i) / (m - 1);
                chains_[c].theta[i] = hi + t * (lo - hi);
            }

            chains_[c].epsilon        = cfg_.init_epsilon;
            chains_[c].step_count     = 0;
            chains_[c].accepted_count = 0;

            chains_[c].log_p = static_cast<const Derived*>(this)
                ->compute_log_p_grad(chains_[c].theta, chains_[c].grad,
                                     s1.data(), s2.data(), true);
        }
    }
}

// =============================================================================
// Burn-in with dual-average step-size adaptation
// =============================================================================

template <typename Derived>
void AngleSamplerHMC<Derived>::run_burn_in() {
    const int m  = cfg_.m;
    const int nt = effective_num_threads(cfg_.num_threads);
    std::cout << ">> Starting Parallel Burn-in (" << num_chains_
              << " chains, " << nt << " thread(s))..." << std::endl;

    std::atomic<int> done{0};
    const int upd = std::max(1, num_chains_ / 50);

    #pragma omp parallel num_threads(nt)
    {
        std::vector<double> p(m), tp(m), pp(m), fg(m), pg(m), s1(m), s2(m);
        std::normal_distribution<double>       norm01(0.0, 1.0);
        std::uniform_real_distribution<double> unif01(0.0, 1.0);

        #pragma omp for schedule(guided)
        for (int c = 0; c < num_chains_; c++) {
            for (int step = 0; step < cfg_.burn_in; step++) {
                chains_[c].step_count++;
                const bool acc = hmc_trajectory(c, p, tp, pp, fg, pg, s1, s2,
                                                norm01, unif01);

                // Dual-averaging step-size adaptation
                const double gamma = 1.0 / std::pow(chains_[c].step_count + 50.0, 0.6);
                chains_[c].epsilon *= std::exp(gamma * ((acc ? 1.0 : 0.0)
                                                        - cfg_.target_accept));
                chains_[c].epsilon  = std::clamp(chains_[c].epsilon, 1e-6, 1e-2);
            }

            const int n_done = ++done;
            if (n_done % upd == 0 || n_done == num_chains_) {
                #pragma omp critical
                {
                    const float pct = static_cast<float>(n_done) / num_chains_ * 100.f;
                    const int bw = 40, pos = static_cast<int>(bw * pct / 100.f);
                    std::cout << "\r   [";
                    for (int i = 0; i < bw; ++i)
                        std::cout << (i < pos ? '=' : (i == pos ? '>' : ' '));
                    std::cout << "] " << std::fixed << std::setprecision(1)
                              << pct << "% " << std::flush;
                }
            }
        }
    }

    std::cout << "\n   Burn-in Complete." << std::endl;
    warmed_up_ = true;
    for (int c = 0; c < num_chains_; c++) {
        chains_[c].step_count     = 0;
        chains_[c].accepted_count = 0;
    }
}

// =============================================================================
// Public sampling interface
// =============================================================================

template <typename Derived>
std::vector<double> AngleSamplerHMC<Derived>::sample_angles() {
    const int m  = cfg_.m;
    const int nt = effective_num_threads(cfg_.num_threads);

    #pragma omp parallel num_threads(nt)
    {
        std::vector<double> p(m), tp(m), pp(m), fg(m), pg(m), s1(m), s2(m);
        std::normal_distribution<double>       norm01(0.0, 1.0);
        std::uniform_real_distribution<double> unif01(0.0, 1.0);

        #pragma omp for schedule(guided)
        for (int c = 0; c < num_chains_; c++) {
            for (int t = 0; t < cfg_.thinning; t++) {
                const bool acc = hmc_trajectory(c, p, tp, pp, fg, pg, s1, s2,
                                                norm01, unif01);
                chains_[c].step_count++;
                if (acc) chains_[c].accepted_count++;
            }
        }
    }

    std::vector<double> out(num_chains_ * m);
    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int c = 0; c < num_chains_; c++)
        std::copy(chains_[c].theta, chains_[c].theta + m,
                  out.begin() + c * m);
    return out;
}

template <typename Derived>
isomorphism::Tensor AngleSamplerHMC<Derived>::sample() {
    auto v = sample_angles();
    std::vector<float> f32(v.begin(), v.end());
    if (num_chains_ > 1)
        return isomorphism::math::array(f32, {num_chains_, cfg_.m},
                                        isomorphism::DType::Float32);
    return isomorphism::math::array(f32, {cfg_.m}, isomorphism::DType::Float32);
}

// =============================================================================
// Core leapfrog + Metropolis-Hastings step
// =============================================================================

template <typename Derived>
bool AngleSamplerHMC<Derived>::hmc_trajectory(
    int c,
    std::vector<double>& p,
    std::vector<double>& theta_p,
    std::vector<double>& p_p,
    std::vector<double>& final_grad,
    std::vector<double>& prop_grad,
    std::vector<double>& s1,
    std::vector<double>& s2,
    std::normal_distribution<double>&       norm01,
    std::uniform_real_distribution<double>& unif01)
{
    const int    m   = cfg_.m;
    const double eps = chains_[c].epsilon;

    // Sample momentum
    double kinetic_curr = 0.0;
    for (int i = 0; i < m; i++) {
        p[i] = norm01(chains_[c].rng) * std::sqrt(mass_[i]);
        kinetic_curr += p[i] * p[i] * inv_mass_[i];
    }
    const double H_curr = -chains_[c].log_p + 0.5 * kinetic_curr;

    std::copy(chains_[c].theta, chains_[c].theta + m, theta_p.begin());
    std::copy(chains_[c].grad,  chains_[c].grad  + m, final_grad.begin());
    std::copy(p.begin(), p.end(), p_p.begin());

    bool   valid       = true;
    double final_log_p = chains_[c].log_p;

    // Leapfrog
    for (int i = 0; i < m; i++)
        p_p[i] += 0.5 * eps * final_grad[i];

    for (int l = 0; l < cfg_.leapfrog_steps; l++) {
        for (int i = 0; i < m; i++)
            theta_p[i] += eps * p_p[i] * inv_mass_[i];

        const bool   is_last = (l == cfg_.leapfrog_steps - 1);
        const double plp     = static_cast<const Derived*>(this)
            ->compute_log_p_grad(theta_p.data(), prop_grad.data(),
                                 s1.data(), s2.data(), is_last);

        if (!std::isfinite(plp)) { valid = false; break; }
        if (is_last) final_log_p = plp;

        std::copy(prop_grad.begin(), prop_grad.end(), final_grad.begin());
        if (!is_last)
            for (int i = 0; i < m; i++)
                p_p[i] += eps * final_grad[i];
    }

    if (!valid) return false;

    // MH accept / reject
    for (int i = 0; i < m; i++)
        p_p[i] += 0.5 * eps * final_grad[i];

    double kinetic_p = 0.0;
    for (int i = 0; i < m; i++)
        kinetic_p += p_p[i] * p_p[i] * inv_mass_[i];
    const double H_prop = -final_log_p + 0.5 * kinetic_p;

    if (std::log(unif01(chains_[c].rng)) < (H_curr - H_prop)) {
        std::copy(theta_p.begin(),    theta_p.end(),    chains_[c].theta);
        std::copy(final_grad.begin(), final_grad.end(), chains_[c].grad);
        chains_[c].log_p = final_log_p;
        return true;
    }
    return false;
}

// =============================================================================
// Explicit instantiations – the compiler generates code for exactly these two
// specialisations and links them into the library.  Users of the library need
// only include the clean declaration header.
// =============================================================================

template class AngleSamplerHMC<PrincipalAngleSampler>;
template class AngleSamplerHMC<SOdAngleSampler>;

} // namespace sampler
