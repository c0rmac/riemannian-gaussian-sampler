#include "sampler/isotropic/so_gaussian_sampler.hpp"

#include <isomorphism/math.hpp>
#include <isomorphism/tensor.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace math = isomorphism::math;
using Tensor   = isomorphism::Tensor;
using DType    = isomorphism::DType;

// =============================================================================
// Helpers
// =============================================================================

// ||X^T X − I_d||_F  (X is always [N, d, d] or [1, d, d] — never bare 2D)
// For a single sample (N=1) this is just the per-sample Frobenius error.
// For a batch (N>1) this returns the RMS error across all N samples.
static float so_error(const Tensor& X, int d, int N = 1) {
    Tensor XtX  = math::matmul(math::transpose(X, {0, 2, 1}), X);
    Tensor diff = math::subtract(XtX, math::eye(d, DType::Float32));
    float  total = static_cast<float>(math::to_double(
        math::sqrt(math::sum(math::square(diff), {}))));
    return (N > 1) ? total / std::sqrt(static_cast<float>(N)) : total;
}

// ||A||_F
static float frob(const Tensor& A) {
    return static_cast<float>(math::to_double(
        math::sqrt(math::sum(math::square(A), {}))));
}

// =============================================================================
// Test utilities
// =============================================================================

static int g_passed = 0;
static int g_failed = 0;

static void check(bool cond, const char* label) {
    if (cond) { std::printf("  [PASS] %s\n", label); ++g_passed; }
    else       { std::printf("  [FAIL] %s\n", label); ++g_failed; }
}

// =============================================================================
// Tests
// =============================================================================

// -----------------------------------------------------------------------------
// 1. Manifold membership: X^T X ≈ I_d for every sample.
//    Tests both even d (d=4) and odd d (d=3).
// -----------------------------------------------------------------------------
static void test_manifold_membership(int d, int n_samples) {
    std::printf("\n--- Manifold membership  SO(%d)  [%d samples] ---\n",
                d, n_samples);

    Tensor m_hat = math::eye(d, DType::Float32);

    sampler::SOdGaussianSampler::Config cfg;
    cfg.alpha                  = 1.0;
    cfg.num_samples            = 1;
    //cfg.angle_cfg.burn_in      = 500;

    sampler::SOdGaussianSampler samp(m_hat, d, cfg);

    float max_err  = 0.f;
    float mean_err = 0.f;
    for (int i = 0; i < n_samples; ++i) {
        Tensor X = samp.sample();
        float  e = so_error(X, d);
        if (e > max_err) max_err = e;
        mean_err += e;
    }
    mean_err /= n_samples;

    std::printf("  max  ||X^T X − I|| = %.2e\n", static_cast<double>(max_err));
    std::printf("  mean ||X^T X − I|| = %.2e\n", static_cast<double>(mean_err));

    check(max_err  < 1e-4f, "max  SO(d) orthogonality error < 1e-4");
    check(mean_err < 1e-5f, "mean SO(d) orthogonality error < 1e-5");
}

// -----------------------------------------------------------------------------
// 2. Sample shape: unbatched (N=1) → [d, d],  batched (N>1) → [N, d, d].
// -----------------------------------------------------------------------------
static void test_sample_shape(int d) {
    std::printf("\n--- Sample shape  SO(%d) ---\n", d);

    Tensor m_hat = math::eye(d, DType::Float32);

    // --- unbatched ---
    {
        sampler::SOdGaussianSampler::Config cfg;
        cfg.alpha       = 10.0;
        cfg.num_samples = 1;
        sampler::SOdGaussianSampler samp(m_hat, d, cfg);

        Tensor X     = samp.sample();
        auto   shape = X.shape();
        std::printf("  unbatched shape: [%d, %d]\n", shape[0], shape[1]);
        check(shape.size() == 2 && shape[0] == d && shape[1] == d,
              "unbatched shape == [d, d]");
    }

    // --- batched ---
    {
        const int N = 8;
        sampler::SOdGaussianSampler::Config cfg;
        cfg.alpha       = 10.0;
        cfg.num_samples = N;
        sampler::SOdGaussianSampler samp(m_hat, d, cfg);

        Tensor X     = samp.sample();
        auto   shape = X.shape();
        std::printf("  batched   shape: [%d, %d, %d]\n",
                    shape[0], shape[1], shape[2]);
        check(shape.size() == 3 && shape[0] == N
                               && shape[1] == d
                               && shape[2] == d,
              "batched shape == [N, d, d]");
    }
}

// -----------------------------------------------------------------------------
// 3. Concentration: higher α ⟹ samples closer to M_hat.
// -----------------------------------------------------------------------------
static void test_concentration(int d, int n_samples) {
    std::printf("\n--- Concentration  SO(%d)  [%d samples] ---\n",
                d, n_samples);

    Tensor m_hat = math::eye(d, DType::Float32);

    auto mean_dist = [&](double alpha) -> float {
        sampler::SOdGaussianSampler::Config cfg;
        cfg.alpha             = alpha;
        cfg.num_samples       = 1;
        //cfg.angle_cfg.burn_in = 500;
        sampler::SOdGaussianSampler samp(m_hat, d, cfg);

        float total = 0.f;
        for (int i = 0; i < n_samples; ++i) {
            Tensor X = samp.sample();
            total += frob(math::subtract(X, m_hat));
        }
        return total / n_samples;
    };

    float d_low  = mean_dist(0.1);
    float d_high = mean_dist(5.0);

    std::printf("  mean dist from M_hat  (α=0.1): %.4f\n",
                static_cast<double>(d_low));
    std::printf("  mean dist from M_hat  (α=5.0): %.4f\n",
                static_cast<double>(d_high));

    check(d_high < d_low, "higher alpha => samples closer to M_hat");
}

// -----------------------------------------------------------------------------
// 4. Batched manifold membership: N samples in one call all on the manifold.
// -----------------------------------------------------------------------------
static void test_batch_membership(int d, int N) {
    std::printf("\n--- Batched membership  SO(%d)  N=%d ---\n", d, N);

    Tensor m_hat = math::eye(d, DType::Float32);

    sampler::SOdGaussianSampler::Config cfg;
    cfg.alpha             = 1.0;
    cfg.num_samples       = N;
    //cfg.angle_cfg.burn_in = 500;
    sampler::SOdGaussianSampler samp(m_hat, d, cfg);

    Tensor X      = samp.sample();
    float  rms_err = so_error(X, d, N);

    std::printf("  RMS ||X^T X − I|| over %d samples = %.2e\n",
                N, static_cast<double>(rms_err));
    check(rms_err < 1e-4f, "batched samples on SO(d) manifold (RMS error < 1e-4)");
}

// -----------------------------------------------------------------------------
// 5. HMC acceptance rate in a reasonable range after warm-up.
// -----------------------------------------------------------------------------
static void test_hmc_acceptance(int d) {
    std::printf("\n--- HMC acceptance rate  SO(%d) ---\n", d);

    Tensor m_hat = math::eye(d, DType::Float32);

    sampler::SOdGaussianSampler::Config cfg;
    cfg.alpha             = 1.0;
    cfg.num_samples       = 1;
    //cfg.angle_cfg.burn_in = 2000;
    sampler::SOdGaussianSampler samp(m_hat, d, cfg);

    for (int i = 0; i < 200; ++i) samp.sample();

    double acc = samp.angle_acceptance_rate();
    std::printf("  acceptance rate = %.3f  (target ~0.65)\n", acc);
    check(acc > 0.30 && acc < 0.95, "acceptance rate in [0.30, 0.95]");
}

// -----------------------------------------------------------------------------
// 6. Even / odd d parity: both root systems (D_m and B_m) work correctly.
// -----------------------------------------------------------------------------
static void test_even_odd_parity() {
    std::printf("\n--- Even/odd d parity ---\n");

    for (int d : {4, 3, 6, 5}) {
        Tensor m_hat = math::eye(d, DType::Float32);

        sampler::SOdGaussianSampler::Config cfg;
        cfg.alpha             = 1.0;
        cfg.num_samples       = 1;
        //cfg.angle_cfg.burn_in = 300;
        sampler::SOdGaussianSampler samp(m_hat, d, cfg);

        float err = so_error(samp.sample(), d);
        std::printf("  SO(%d) [%s]:  ||X^T X − I|| = %.2e\n",
                    d, (d % 2 == 0 ? "even" : "odd "),
                    static_cast<double>(err));

        char label[64];
        std::snprintf(label, sizeof(label),
                      "SO(%d) sample on manifold (err < 1e-4)", d);
        check(err < 1e-4f, label);
    }
}

// =============================================================================
// Benchmark
// =============================================================================

static void run_so_benchmark(int K, int d, int N) {
    using namespace std::chrono;

    std::cout << "\n--- SO(" << d << ") Benchmark  N=" << N
              << "  K=" << K << " batches ---\n";

    Tensor m_hat = math::eye(d, DType::Float32);

    sampler::SOdGaussianSampler::Config cfg;
    cfg.alpha             = 1.0;
    cfg.num_samples       = N;
    //cfg.angle_cfg.burn_in = 500;

    auto t0 = high_resolution_clock::now();
    sampler::SOdGaussianSampler samp(m_hat, d, cfg);
    auto t1 = high_resolution_clock::now();
    std::cout << ">> Burn-in: "
              << duration<double, std::milli>(t1 - t0).count() << " ms\n";

    std::vector<double> timings;
    timings.reserve(K);

    for (int i = 0; i < K; ++i) {
        auto ts = high_resolution_clock::now();
        Tensor batch = samp.sample();
        auto te = high_resolution_clock::now();
        timings.push_back(duration<double, std::milli>(te - ts).count());
        std::cout << "   Batch [" << i + 1 << "/" << K << "]: "
                  << std::setprecision(3) << timings.back() << " ms\n";
    }

    double avg = std::accumulate(timings.begin(), timings.end(), 0.0) / K;
    double sq  = std::inner_product(timings.begin(), timings.end(),
                                    timings.begin(), 0.0);
    double std = std::sqrt(sq / K - avg * avg);

    std::cout << "\n  Avg " << std::setprecision(3) << avg << " ms  "
              << "± " << std << " ms  "
              << "(" << std::setprecision(0) << N * 1000.0 / avg
              << " samples/sec)\n";
}

// -----------------------------------------------------------------------------
// Print a single SO(d) sample to stdout.
// -----------------------------------------------------------------------------
static void test_print_sample(int d) {
    std::printf("\n--- Single sample  SO(%d) ---\n", d);

    Tensor m_hat = math::eye(d, DType::Float32);

    double delta = 0.0000001;
    sampler::SOdGaussianSampler::Config cfg;
    //cfg.alpha             = 1.0 / (delta * delta);
    cfg.alpha             = 1e11;
    cfg.num_samples       = 1;
    //cfg.angle_cfg.burn_in = 2000;

    sampler::SOdGaussianSampler samp(m_hat, d, cfg);
    Tensor X = samp.sample();

    std::cout << "  X =\n" << X << "\n";
    std::printf("  ||X^T X − I|| = %.2e\n",
                static_cast<double>(so_error(X, d)));
}

// =============================================================================
// main
// =============================================================================

int main() {
    using namespace std::chrono;

    std::printf("========================================\n");
    std::printf("   SO(d) Gaussian Sampler — Test Suite  \n");
    std::printf("========================================\n");
    sampler::set_num_threads(8);

    math::set_default_device_cpu();
    auto t_start = high_resolution_clock::now();

    //test_print_sample(10);
    test_print_sample(5);

    /*
    // --- Correctness tests ---
    test_manifold_membership(4, 20);   // even d
    test_manifold_membership(3, 20);   // odd d
    test_manifold_membership(6, 10);
    test_manifold_membership(5, 10);

    test_sample_shape(4);
    test_sample_shape(3);

    test_concentration(4, 30);
    test_concentration(3, 30);

    test_batch_membership(4, 32);
    test_batch_membership(3, 32);

    test_hmc_acceptance(4);
    test_hmc_acceptance(3);

    test_even_odd_parity();
*/
    // --- Benchmark ---
    //run_so_benchmark(20, 1000, 8);
    //run_so_benchmark(20, 128, 5000);

    auto t_end = high_resolution_clock::now();

    std::printf("\n========================================\n");
    std::printf(" Results: %d passed, %d failed\n", g_passed, g_failed);
    std::printf(" Total time: %.3f s\n",
                duration<double>(t_end - t_start).count());
    std::printf("========================================\n");

    return g_failed > 0 ? 1 : 0;
}
