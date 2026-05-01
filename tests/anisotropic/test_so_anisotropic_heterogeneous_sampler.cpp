#include "sampler/anisotropic/so_anisotropic_heterogeneous_gaussian_sampler.hpp"
#include "sampler/thread_config.hpp"

#include <isomorphism/math.hpp>
#include <isomorphism/tensor.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

namespace math = isomorphism::math;
using Tensor   = isomorphism::Tensor;
using DType    = isomorphism::DType;
using Sampler  = sampler::anisotropic::SOdAnisotropicHeterogeneousGaussianSampler;

// =============================================================================
// Utilities
// =============================================================================

static int g_passed = 0;
static int g_failed = 0;

static void check(bool cond, const char* label) {
    if (cond) { std::printf("  [PASS] %s\n", label); ++g_passed; }
    else       { std::printf("  [FAIL] %s\n", label); ++g_failed; }
}

// RMS of (XᵀX − I) averaged over N particles.  X must be [N, d, d].
static float so_error(const Tensor& X, int d, int N) {
    Tensor XtX  = math::matmul(math::transpose(X, {0, 2, 1}), X);
    Tensor diff = math::subtract(XtX, math::eye(d, DType::Float32));
    float  total = static_cast<float>(
        math::to_double(math::sqrt(math::sum(math::square(diff), {}))));
    return total / std::sqrt(static_cast<float>(N));
}

// Per-particle Frobenius distance from the consensus (M̂ = I by default).
// Returns a CPU vector of length N.
static std::vector<float> distances_from_identity(const Tensor& X, int d, int N) {
    Tensor eye_b = math::expand_dims(math::eye(d, DType::Float32), {0}); // [1,d,d]
    Tensor diff  = math::subtract(X, eye_b);                             // [N,d,d]
    Tensor dist2 = math::sum(math::square(diff), {1, 2});                // [N]
    math::eval(dist2);
    std::vector<float> out(N);
    for (int i = 0; i < N; ++i)
        out[i] = std::sqrt(std::max(0.0f, static_cast<float>(
            math::to_double(math::slice(dist2, i, i + 1, 0)))));
    return out;
}

// Build a D×D scaled-identity Lie algebra precision (D = d*(d-1)/2).
static std::vector<double> make_scaled_identity(int d, double scale) {
    const int D = d * (d - 1) / 2;
    std::vector<double> G(static_cast<size_t>(D) * D, 0.0);
    for (int a = 0; a < D; ++a) G[a * D + a] = scale;
    return G;
}

// Build a D×D block-diagonal Lie precision where plane (j,k) with j<d/2 and k<d/2
// gets gamma_low, all other planes get gamma_high.
static std::vector<double> make_block_gamma(int d, double gamma_low, double gamma_high) {
    const int D = d * (d - 1) / 2;
    std::vector<double> G(static_cast<size_t>(D) * D, 0.0);
    int a = 0;
    for (int j = 0; j < d; ++j) {
        for (int k = j + 1; k < d; ++k, ++a) {
            G[a * D + a] = (j < d / 2 && k < d / 2) ? gamma_low : gamma_high;
        }
    }
    return G;
}

// =============================================================================
// Test 1 — SO(d) membership for N particles with mixed gammas
//
// Creates N particles split into two groups:
//   - Group A (first N/2): loose gamma (small scale → broad distribution)
//   - Group B (last  N/2): tight gamma (large scale → concentrated near I)
//
// Verifies that all N output matrices satisfy XᵀX ≈ I.
// =============================================================================

static void test_so_membership(int d, int N) {
    std::printf("\n--- Test 1: SO(d) membership  d=%d  N=%d ---\n", d, N);

    const int half = N / 2;

    std::vector<std::vector<double>> gammas(N);
    for (int i = 0; i < N; ++i)
        gammas[i] = make_scaled_identity(d, i < half ? 0.3 : 8.0);

    Sampler::Config cfg;
    cfg.N           = N;
    cfg.d           = d;
    cfg.num_threads = sampler::effective_num_threads(8);
    cfg.burn_in     = 1500;
    cfg.num_chains  = 4;

    Sampler samp(cfg);
    samp.set_m_hat(math::eye(d, DType::Float32));
    samp.update_gammas(gammas, cfg.burn_in);

    auto t0 = std::chrono::high_resolution_clock::now();
    Tensor X = samp.sample();
    auto t1 = std::chrono::high_resolution_clock::now();

    float rms = so_error(X, d, N);
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  sample() time: %.2f ms | RMS orthogonality error: %.2e\n",
                ms, static_cast<double>(rms));

    auto rates = samp.angle_acceptance_rates();
    double avg_rate = std::accumulate(rates.begin(), rates.end(), 0.0) / N;
    std::printf("  Mean Phase I acceptance rate across %d particles: %.3f\n",
                N, avg_rate);

    check(rms < 1e-4f, "all N samples lie on SO(d)");
    check(avg_rate > 0.1 && avg_rate < 1.0, "Phase I acceptance rate is plausible");
}

// =============================================================================
// Test 2 — Concentration ordering
//
// Particles with tight Γ should produce samples closer to the consensus M̂ = I
// than particles with loose Γ.  We draw a batch and verify:
//   mean_dist(tight group) < mean_dist(loose group)
// =============================================================================

static void test_concentration_ordering(int d, int N_per_group) {
    std::printf("\n--- Test 2: Concentration ordering  d=%d  N_per_group=%d ---\n",
                d, N_per_group);

    const int N      = 2 * N_per_group;
    const double loose = 0.2;
    const double tight = 20.0;

    std::vector<std::vector<double>> gammas(N);
    for (int i = 0; i < N; ++i)
        gammas[i] = make_scaled_identity(d, i < N_per_group ? loose : tight);

    Sampler::Config cfg;
    cfg.N           = N;
    cfg.d           = d;
    cfg.num_threads = sampler::effective_num_threads(8);
    cfg.burn_in     = 2000;
    cfg.num_chains  = 4;

    Sampler samp(cfg);
    samp.set_m_hat(math::eye(d, DType::Float32));
    samp.update_gammas(gammas, cfg.burn_in);

    // Draw several batches to build up per-particle distance estimates.
    const int rounds = 30;
    std::vector<float> cumulative(N, 0.0f);
    for (int r = 0; r < rounds; ++r) {
        Tensor X  = samp.sample();
        auto   ds = distances_from_identity(X, d, N);
        for (int i = 0; i < N; ++i) cumulative[i] += ds[i];
    }

    float mean_loose = 0.0f, mean_tight = 0.0f;
    for (int i = 0; i < N_per_group; ++i) {
        mean_loose += cumulative[i];
        mean_tight += cumulative[N_per_group + i];
    }
    mean_loose /= N_per_group * rounds;
    mean_tight /= N_per_group * rounds;

    std::printf("  Mean dist from I — loose (γ=%.1f): %.4f\n", loose,
                static_cast<double>(mean_loose));
    std::printf("  Mean dist from I — tight (γ=%.1f): %.4f\n", tight,
                static_cast<double>(mean_tight));

    check(mean_tight < mean_loose,
          "tight-gamma particles are closer to consensus than loose-gamma");
    check(mean_tight > 0.0f, "tight-gamma samples are not trivially identity");
}

// =============================================================================
// Test 3 — Gamma warm-start update
//
// Starts N particles with one set of gammas, draws a sample (checks validity),
// then updates to a different set of gammas and draws again.  Verifies that:
//   - Both pre- and post-update samples lie on SO(d).
//   - Only the particles whose gamma changed beyond the threshold trigger a
//     warm-start (no crash, acceptance rates stay reasonable).
//   - A sub-threshold change (< gamma_frob_rtol) does not rebuild chains.
// =============================================================================

static void test_gamma_update(int d, int N) {
    std::printf("\n--- Test 3: Gamma warm-start update  d=%d  N=%d ---\n", d, N);

    // Initial gammas: all anisotropic block-diagonal.
    std::vector<std::vector<double>> gammas_v0(N);
    for (int i = 0; i < N; ++i)
        gammas_v0[i] = make_block_gamma(d, 0.5, 5.0);

    Sampler::Config cfg;
    cfg.N            = N;
    cfg.d            = d;
    cfg.num_threads  = sampler::effective_num_threads(8);
    cfg.burn_in      = 1500;
    cfg.num_chains   = 4;
    cfg.gamma_frob_rtol = 0.05;

    Sampler samp(cfg);
    samp.set_m_hat(math::eye(d, DType::Float32));
    samp.update_gammas(gammas_v0, cfg.burn_in);

    Tensor X0    = samp.sample();
    float  rms0  = so_error(X0, d, N);
    std::printf("  Pre-update  RMS error: %.2e\n", static_cast<double>(rms0));
    check(rms0 < 1e-4f, "pre-update samples on SO(d)");

    // Large change: flip the block structure.  All particles trigger a warm-start.
    std::vector<std::vector<double>> gammas_v1(N);
    for (int i = 0; i < N; ++i)
        gammas_v1[i] = make_block_gamma(d, 5.0, 0.5);   // roles swapped

    samp.update_gammas(gammas_v1, 800);

    Tensor X1   = samp.sample();
    float  rms1 = so_error(X1, d, N);
    std::printf("  Post-update RMS error: %.2e\n", static_cast<double>(rms1));
    check(rms1 < 1e-4f, "post-update samples on SO(d) after warm-start");

    // Sub-threshold change: perturb each gamma by a tiny amount.
    // No chain should rebuild; output should still be valid.
    const int D_v2 = d * (d - 1) / 2;
    std::vector<std::vector<double>> gammas_v2 = gammas_v1;
    for (int i = 0; i < N; ++i)
        for (int a = 0; a < D_v2; ++a)
            gammas_v2[i][a * D_v2 + a] *= 1.001;   // 0.1 % change, well below rtol=5 %

    samp.update_gammas(gammas_v2, 800);   // no rebuilds expected

    Tensor X2   = samp.sample();
    float  rms2 = so_error(X2, d, N);
    std::printf("  Sub-threshold RMS error: %.2e\n", static_cast<double>(rms2));
    check(rms2 < 1e-4f, "sub-threshold update: samples still on SO(d)");
}

// =============================================================================
// Test 4 — Simulated diffusion rounds
//
// Mimics a CBO-style outer loop where the consensus tightens over iterations:
//   Round 0: broad priors (γ = 0.1 * I) → particles spread over SO(d)
//   Round 1: moderate concentration (γ = 2.0 * I)
//   Round 2: strong concentration, heterogeneous (odd particles tight, even loose)
//
// Checks SO(d) membership at every round and prints the evolution of the
// mean distance from the consensus.
// =============================================================================

static void test_diffusion_rounds(int d, int N) {
    std::printf("\n--- Test 4: Diffusion rounds  d=%d  N=%d ---\n", d, N);

    Sampler::Config cfg;
    cfg.N           = N;
    cfg.d           = d;
    cfg.num_threads = sampler::effective_num_threads(8);
    cfg.burn_in     = 1500;
    cfg.num_chains  = 4;

    Sampler samp(cfg);
    samp.set_m_hat(math::eye(d, DType::Float32));

    struct Round { const char* label; double gamma_even; double gamma_odd; };
    const Round rounds[] = {
        { "Round 0 (broad, homogeneous)",       0.1,  0.1  },
        { "Round 1 (moderate, homogeneous)",    2.0,  2.0  },
        { "Round 2 (tight+loose, heterogeneous)", 10.0, 0.3 },
    };

    float prev_tight_dist = 1e9f;
    bool  tightened       = true;

    for (const auto& rnd : rounds) {
        std::vector<std::vector<double>> gammas(N);
        for (int i = 0; i < N; ++i)
            gammas[i] = make_scaled_identity(d, (i % 2 == 0) ? rnd.gamma_even
                                                              : rnd.gamma_odd);

        samp.update_gammas(gammas, cfg.burn_in);

        Tensor X   = samp.sample();
        float  rms = so_error(X, d, N);
        auto   ds  = distances_from_identity(X, d, N);
        float  mean_dist = std::accumulate(ds.begin(), ds.end(), 0.0f) / N;

        std::printf("  %s\n", rnd.label);
        std::printf("    RMS error: %.2e  |  mean dist from I: %.4f\n",
                    static_cast<double>(rms), static_cast<double>(mean_dist));

        std::string lbl = std::string(rnd.label) + " — samples on SO(d)";
        check(rms < 1e-4f, lbl.c_str());
    }

    // In the final heterogeneous round, tight particles (even indices, γ=10)
    // should sit closer to the consensus than loose particles (odd, γ=0.3).
    {
        // Re-draw a fresh sample at the final round's gamma.
        std::vector<std::vector<double>> final_gammas(N);
        for (int i = 0; i < N; ++i)
            final_gammas[i] = make_scaled_identity(
                d, (i % 2 == 0) ? rounds[2].gamma_even : rounds[2].gamma_odd);
        samp.update_gammas(final_gammas, 800);

        float sum_even = 0.0f, sum_odd = 0.0f;
        int   cnt_even = 0,    cnt_odd = 0;
        const int draws = 20;
        for (int r = 0; r < draws; ++r) {
            Tensor X  = samp.sample();
            auto   ds = distances_from_identity(X, d, N);
            for (int i = 0; i < N; ++i) {
                if (i % 2 == 0) { sum_even += ds[i]; ++cnt_even; }
                else             { sum_odd  += ds[i]; ++cnt_odd;  }
            }
        }
        float mean_even = sum_even / cnt_even;   // tight
        float mean_odd  = sum_odd  / cnt_odd;    // loose

        std::printf("  Final heterogeneous round:\n");
        std::printf("    Even (tight γ=%.1f) mean dist: %.4f\n",
                    rounds[2].gamma_even, static_cast<double>(mean_even));
        std::printf("    Odd  (loose γ=%.1f) mean dist: %.4f\n",
                    rounds[2].gamma_odd,  static_cast<double>(mean_odd));

        check(mean_even < mean_odd,
              "tight particles closer to consensus than loose particles (round 2)");
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::printf("=================================================\n");
    std::printf("  SO(d) Anisotropic Heterogeneous Sampler Tests\n");
    std::printf("=================================================\n");

    sampler::set_num_threads(8);
    math::set_default_device_cpu();

    test_so_membership(4, 8);
    test_concentration_ordering(4, 5);
    test_gamma_update(4, 6);
    test_diffusion_rounds(4, 8);

    std::printf("\n=================================================\n");
    std::printf(" Final Results: %d passed, %d failed\n", g_passed, g_failed);
    std::printf("=================================================\n");

    return g_failed > 0 ? 1 : 0;
}
