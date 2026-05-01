#include "sampler/anisotropic/so_anisotropic_angle_sampler.hpp"
#include "sampler/anisotropic/so_hypersphere_sampler_cpu.hpp"
#include "sampler/thread_config.hpp"

#include <isomorphism/math.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

namespace math = isomorphism::math;
using AngleSampler  = sampler::anisotropic::SOdAnisotropicAngleSampler;
using HyperSampler  = sampler::anisotropic::SOdHypersphereSamplerCPU;
using ColumnStats   = HyperSampler::ColumnStats;

// =============================================================================
// Helpers
// =============================================================================

// Build a d×d block-diagonal spatial gamma (for Phase I angle sampler).
static std::vector<double> make_block_gamma_spatial(int d,
                                                     double gamma_low,
                                                     double gamma_high)
{
    std::vector<double> G(d * d, 0.0);
    for (int i = 0; i < d; i++)
        G[i * d + i] = (i < d / 2) ? gamma_low : gamma_high;
    return G;
}

// Build the corresponding D×D Lie algebra precision for Phase II.
// Plane (j,k): gets gamma_low if both j,k < d/2, else gamma_high.
static std::vector<double> make_block_gamma_lie(int d,
                                                 double gamma_low,
                                                 double gamma_high)
{
    const int D = d * (d - 1) / 2;
    std::vector<double> G(static_cast<size_t>(D) * D, 0.0);
    int a = 0;
    for (int j = 0; j < d; j++) {
        for (int k = j + 1; k < d; k++, a++) {
            G[a * D + a] = (j < d / 2 && k < d / 2) ? gamma_low : gamma_high;
        }
    }
    return G;
}

// Run Phase I + Phase II for a given (d, gamma_low, gamma_high) and print
// per-column acceptance statistics.
static void analyse(int d, double gamma_low, double gamma_high, int N) {
    std::printf("\n--- d=%d  gamma_low=%.2f  gamma_high=%.2f  N=%d ---\n",
                d, gamma_low, gamma_high, N);

    auto gamma_spatial = make_block_gamma_spatial(d, gamma_low, gamma_high);
    auto gamma_lie     = make_block_gamma_lie(d, gamma_low, gamma_high);

    // Phase I: sample angles from the isotropic envelope (needs d×d spatial gamma).
    AngleSampler::Config acfg;
    acfg.d          = d;
    acfg.num_chains = 8;
    acfg.burn_in    = 1000;
    AngleSampler angle_samp(gamma_spatial, acfg);
    std::printf("  Phase I acceptance rate: %.3f\n", angle_samp.acceptance_rate());

    auto flat_theta = angle_samp.sample_angles(N);

    // Phase II: build orientation frames with D×D Lie gamma, collecting stats.
    HyperSampler::Config hcfg;
    hcfg.num_threads = sampler::effective_num_threads(8);
    HyperSampler hyper(d, gamma_lie, angle_samp.gamma_min(), hcfg);

    std::vector<ColumnStats> stats;
    hyper.build_orientation_frames_with_stats(flat_theta, N, stats);

    std::printf("  Phase II per-column acceptance (summed over %d samples):\n", N);
    std::printf("  %-6s  %-10s  %-10s  %-10s\n",
                "col", "attempts", "accepted", "rate");
    for (int col = 0; col < static_cast<int>(stats.size()); col++) {
        std::printf("  %-6d  %-10lld  %-10lld  %.4f\n",
                    col,
                    static_cast<long long>(stats[col].attempts),
                    static_cast<long long>(stats[col].accepted),
                    stats[col].acceptance_rate());
    }
}

// =============================================================================
// main
// =============================================================================

int main() {
    std::printf("=================================================\n");
    std::printf("  Phase II — Acceptance Rate Analysis\n");
    std::printf("=================================================\n");

    sampler::set_num_threads(8);
    math::set_default_device_cpu();

    const int N = 500;

    // Isotropic baseline (gamma_low == gamma_high → ratio 1×)
    /*analyse(4, 1.0, 1.0, N);
    analyse(6, 1.0, 1.0, N);

    // Mild anisotropy (5×)
    analyse(4, 1.0, 5.0, N);
    analyse(6, 1.0, 5.0, N);

    // Strong anisotropy (50×)
    analyse(4, 0.1, 5.0, N);
    analyse(6, 0.1, 5.0, N);*/

    analyse(3, 0.1, 5.0, N);

    std::printf("\n=================================================\n");
    return 0;
}
