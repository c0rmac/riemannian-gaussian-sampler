#include "sampler/anisotropic/so_anisotropic_angle_sampler.hpp"

#include <cassert>
#include <cmath>
#include <isomorphism/math.hpp>
#include <omp.h>
#include <vector>

namespace sampler {
namespace anisotropic {

namespace math = isomorphism::math;

// =============================================================================
// Construction
// =============================================================================

SOdAnisotropicAngleSampler::SOdAnisotropicAngleSampler(
    const std::vector<double>& gamma_flat, Config cfg)
    : d_(cfg.d)
    , m_(cfg.d / 2)
    , gamma_min_(0.0)
    , cfg_(cfg)
{
    assert(cfg.d >= 2);
    assert(static_cast<int>(gamma_flat.size()) == cfg.d * cfg.d);
    rebuild(gamma_flat, cfg.burn_in);
}

// =============================================================================
// Helpers
// =============================================================================

SOdAngleSampler::Config SOdAnisotropicAngleSampler::make_inner_config(
    int burn_in_steps) const
{
    SOdAngleSampler::Config c;
    // We do NOT set c.d or c.alpha here; the rebuild loop handles that per-channel.
    c.num_chains     = cfg_.num_chains;
    c.num_threads    = cfg_.num_threads;
    c.burn_in        = (burn_in_steps > 0) ? burn_in_steps : cfg_.burn_in;
    c.leapfrog_steps = cfg_.leapfrog_steps;
    c.init_epsilon   = cfg_.init_epsilon;
    c.target_accept  = cfg_.target_accept;
    c.mass_high      = cfg_.mass_high;
    c.mass_low       = cfg_.mass_low;
    c.thinning       = cfg_.thinning;
    c.seed           = cfg_.seed;
    return c;
}

void SOdAnisotropicAngleSampler::rebuild(const std::vector<double>& gamma_flat, int burn_in_steps) {
    // -------------------------------------------------------------------------
    // NUMERICAL STABILIZATION BLOCK
    // -------------------------------------------------------------------------
    std::vector<double> G_safe = gamma_flat;

    for (int i = 0; i < d_; i++) {
        // 1. Explicit Symmetrization (Guarantees bit-for-bit symmetry even after float cast)
        for (int j = i + 1; j < d_; j++) {
            double avg = (G_safe[i * d_ + j] + G_safe[j * d_ + i]) * 0.5;
            G_safe[i * d_ + j] = avg;
            G_safe[j * d_ + i] = avg;
        }
        // 2. Diagonal Jitter (Prevents solver collapse on ill-conditioned / zero spectra)
        G_safe[i * d_ + i] += 1e-8;
    }

    // Upload stabilized Γ.
    auto G    = math::array(G_safe, {d_, d_}, isomorphism::DType::Float64);
    auto eigs = math::eigvalsh(G);     // [d] ascending order
    eigs = math::clamp(eigs, -1e8, 1e30);
    math::eval(eigs);
    std::vector<double> eig_vals = math::to_double_vector(eigs);

    // Build new samplers into a fresh local vector then swap.
    // This avoids the clear()+resize() pattern on the member vector which triggers
    // ASAN container-overflow false positives when rebuild() is called from within
    // a nested OpenMP parallel region (the outer update_gammas parallel-for).
    std::vector<std::unique_ptr<SOdAngleSampler>> new_inners(m_);

    // Each rotational plane k is a fully independent 2D HMC sampler (including
    // its burn-in phase), so construction is embarrassingly parallel.
    #pragma omp parallel for schedule(static) num_threads(cfg_.num_threads)
    for (int k = 0; k < m_; k++) {
        // Group eigenvalues into rotational pairs: (0,1), (2,3), (4,5)
        double a1 = eig_vals[2 * k];
        double a2 = (2 * k + 1 < d_) ? eig_vals[2 * k + 1] : a1;
        double alpha_k = (a1 + a2) / 2.0;

        // Prevent strictly zero precision which breaks HMC
        if (alpha_k < 1e-6) alpha_k = 1e-6;

        SOdAngleSampler::Config c = make_inner_config(burn_in_steps);
        c.d     = 2;         // 2D matrix generates exactly 1 angle
        c.alpha = alpha_k;

        // Shift seed slightly for each channel so they don't draw identical HMC noise
        c.seed  = cfg_.seed + k * 1337;

        new_inners[k] = std::make_unique<SOdAngleSampler>(c);
    }

    // Single-threaded swap: destroys old samplers, installs new ones.
    inners_ = std::move(new_inners);
}

// =============================================================================
// Public interface
// =============================================================================

std::vector<double> SOdAnisotropicAngleSampler::sample_angles(int num_samples) {
    std::vector<double> out(static_cast<size_t>(num_samples) * m_, 0.0);

    // Pre-allocate one result buffer per channel so threads never share state.
    std::vector<std::vector<double>> channel_results(m_);

    // The m independent HMC chains share no data — run them in parallel.
    #pragma omp parallel for schedule(static) num_threads(cfg_.num_threads)
    for (int k = 0; k < m_; k++) {
        channel_results[k] = inners_[k]->sample_angles(num_samples);
    }

    // Interleave: channel k maps to column k of the [num_samples × m] layout.
    for (int k = 0; k < m_; k++)
        for (int n = 0; n < num_samples; n++)
            out[n * m_ + k] = channel_results[k][n];

    return out;
}

double SOdAnisotropicAngleSampler::acceptance_rate() const {
    if (inners_.empty()) return -1.0;
    double sum = 0.0;
    for (const auto& sampler : inners_) {
        sum += sampler->acceptance_rate();
    }
    return sum / inners_.size(); // Return average acceptance across all channels
}

void SOdAnisotropicAngleSampler::update_gamma(
    const std::vector<double>& gamma_flat, int burn_in_steps)
{
    assert(static_cast<int>(gamma_flat.size()) == d_ * d_);
    rebuild(gamma_flat, burn_in_steps);
}

} // namespace anisotropic
} // namespace sampler