#include "sampler/so_angle_sampler.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

namespace sampler {

// =============================================================================
// Construction
// =============================================================================

SOdAngleSampler::SOdAngleSampler(Config cfg)
    : AngleSamplerHMC<SOdAngleSampler>(to_base_config(cfg))
    , d_(cfg.d)
    , odd_(cfg.d % 2 == 1)
{
    assert(cfg.d >= 2 && "SOdAngleSampler requires d >= 2");
    assert(cfg.alpha > 0.0);

    // Derived data fully initialised: safe to call CRTP dispatch.
    start_sampler();
}

// =============================================================================
// Log-probability and gradient  (SO(d) density)
//
// Using the cosine-difference representation:
//
//   log p(θ) = −α Σᵢ θᵢ²
//            + 2 [d odd] Σᵢ log sin(θᵢ/2)
//            + 2 Σ_{i<j} log(cos θⱼ − cos θᵢ)
//
// Gradients:
//   ∂/∂θᵢ = −2α θᵢ
//           + [d odd] cot(θᵢ/2)   ≡ (1 + cos θᵢ)/sin θᵢ
//           + 2 sin θᵢ Σ_{j>i} 1/(cos θⱼ − cos θᵢ)
//           − 2 sin θᵢ Σ_{j<i} 1/(cos θᵢ − cos θⱼ)
//
// scratch1[i] = cos θᵢ,   scratch2[i] = sin θᵢ   (precomputed once per call)
// =============================================================================

double SOdAngleSampler::compute_log_p_grad(
    const double* theta,
    double*       grad,
    double*       scratch1,   // cos θ
    double*       scratch2,   // sin θ
    bool          calc_log_p) const
{
    const int    m       = cfg_.m;
    const double alp     = cfg_.alpha;
    const double neg_inf = -std::numeric_limits<double>::infinity();

    std::fill(grad, grad + m, 0.0);

    // --- Weyl chamber boundary check -----------------------------------------
    if (theta[0] >= M_PI || theta[m - 1] <= 0.0) return neg_inf;
    for (int i = 0; i < m - 1; i++)
        if (theta[i + 1] >= theta[i]) return neg_inf;

    double log_p = 0.0;

    // --- O(m) precomputation -------------------------------------------------
    for (int i = 0; i < m; i++) {
        const double c = std::cos(theta[i]);
        const double s = std::sin(theta[i]);

        scratch1[i] = c;
        scratch2[i] = s;

        // Gaussian potential
        grad[i] -= 2.0 * alp * theta[i];
        if (calc_log_p) log_p -= alp * theta[i] * theta[i];

        // Odd-d boundary term: 2 log sin(θᵢ/2)
        // gradient: cot(θᵢ/2) = (1 + cos θᵢ) / sin θᵢ
        if (odd_) {
            if (s <= 0.0) return neg_inf;
            grad[i] += (1.0 + c) / s;
            if (calc_log_p) {
                const double s_half = std::sin(0.5 * theta[i]);
                if (s_half <= 0.0) return neg_inf;
                log_p += 2.0 * std::log(s_half);
            }
        }
    }

    // --- O(m²) Dyson repulsion via cosine differences (SIMD-friendly) --------
    // For pair (i, j) with i < j:   θᵢ > θⱼ ⟹ cos θⱼ > cos θᵢ ⟹ diff > 0.
    // Contribution to log p: 2 log(cos θⱼ − cos θᵢ)
    // grad[i] += 2 sin θᵢ / diff
    // grad[j] -= 2 sin θⱼ / diff
    double log_p_sum = 0.0;
    bool   invalid   = false;

    for (int i = 0; i < m; i++) {
        double g_i = grad[i];
        const double s_i = scratch2[i];
        const double c_i = scratch1[i];

        #pragma omp simd reduction(+:log_p_sum) reduction(|:invalid)
        for (int j = i + 1; j < m; j++) {
            const double diff = scratch1[j] - c_i;   // cos θⱼ − cos θᵢ

            if (diff <= 0.0) invalid = true;
            const double safe = (diff <= 0.0) ? 1.0 : diff;
            const double inv  = 1.0 / safe;

            g_i       += 2.0 * s_i      * inv;
            grad[j]   -= 2.0 * scratch2[j] * inv;
            if (calc_log_p) log_p_sum += 2.0 * std::log(safe);
        }
        grad[i] = g_i;
    }

    if (invalid) return neg_inf;
    if (calc_log_p) log_p += log_p_sum;

    return calc_log_p ? log_p : 0.0;
}

} // namespace sampler
