#include "sampler/principal_angle_sampler.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

namespace sampler {

// =============================================================================
// Construction
// =============================================================================

PrincipalAngleSampler::PrincipalAngleSampler(Config cfg)
    : AngleSamplerHMC<PrincipalAngleSampler>(to_base_config(cfg))
    , n_(cfg.n)
    , bnd_(static_cast<double>(cfg.n - 2 * cfg.k))
{
    assert(cfg.n >= 2 * cfg.k && "PrincipalAngleSampler requires n >= 2k");
    assert(cfg.k >= 1);
    assert(cfg.alpha > 0.0);

    // Derived data is fully initialised: safe to call CRTP dispatch.
    start_sampler();
}

// =============================================================================
// Log-probability and Dyson drift  (Stiefel density)
//
//   log p(θ) = −α ‖θ‖² + (n−2k) Σᵢ log|sin θᵢ| + Σ_{i<j} log|sin²θᵢ − sin²θⱼ|
//
// scratch1[i] = sin²(θᵢ),   scratch2[i] = sin(2θᵢ)  (precomputed once per call)
// =============================================================================

double PrincipalAngleSampler::compute_log_p_grad(
    const double* theta,
    double*       grad,
    double*       scratch1,   // sin²
    double*       scratch2,   // sin 2θ
    bool          calc_log_p) const
{
    const int    k       = cfg_.m;
    const double alp     = cfg_.alpha;
    const double neg_inf = -std::numeric_limits<double>::infinity();

    std::fill(grad, grad + k, 0.0);

    // --- Weyl chamber boundary check -----------------------------------------
    if (theta[0] >= M_PI || theta[k - 1] <= 0.0) return neg_inf;
    for (int i = 0; i < k - 1; i++)
        if (theta[i + 1] >= theta[i]) return neg_inf;

    double log_p = 0.0;

    // --- O(k) precomputation -------------------------------------------------
    for (int i = 0; i < k; i++) {
        const double s = std::sin(theta[i]);
        const double c = std::cos(theta[i]);

        scratch1[i] = s * s;
        scratch2[i] = 2.0 * s * c;

        grad[i] -= 2.0 * alp * theta[i];
        if (calc_log_p) log_p -= alp * theta[i] * theta[i];

        if (bnd_ > 0.0) {
            if (s <= 0.0) return neg_inf;
            grad[i] += bnd_ * c / s;
            if (calc_log_p) log_p += bnd_ * std::log(s);
        }
    }

    // --- O(k²) Dyson repulsion (SIMD-friendly) --------------------------------
    double log_p_sum = 0.0;
    bool   invalid   = false;

    for (int i = 0; i < k; i++) {
        double g_i = grad[i];

        #pragma omp simd reduction(+:log_p_sum) reduction(|:invalid)
        for (int j = i + 1; j < k; j++) {
            const double diff = scratch1[i] - scratch1[j];

            if (diff <= 0.0) invalid = true;
            const double safe = (diff <= 0.0) ? 1.0 : diff;
            const double inv  = 1.0 / safe;

            g_i       += scratch2[i] * inv;
            grad[j]   -= scratch2[j] * inv;
            if (calc_log_p) log_p_sum += std::log(safe);
        }
        grad[i] = g_i;
    }

    if (invalid) return neg_inf;
    if (calc_log_p) log_p += log_p_sum;

    return calc_log_p ? log_p : 0.0;
}

// =============================================================================
// Validity check (used externally)
// =============================================================================

bool PrincipalAngleSampler::is_valid(const double* theta) const {
    std::vector<double> g(cfg_.m), s1(cfg_.m), s2(cfg_.m);
    const double lp = compute_log_p_grad(theta, g.data(), s1.data(), s2.data(), true);
    return std::isfinite(lp);
}

} // namespace sampler
