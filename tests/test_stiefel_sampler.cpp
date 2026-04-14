#include "sampler/stiefel_gaussian_sampler.hpp"

#include <isomorphism/math.hpp>
#include <isomorphism/tensor.hpp>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>
#include <iostream>
#include <numeric>
#include <iomanip>


namespace math = isomorphism::math;
using Tensor   = isomorphism::Tensor;
using DType    = isomorphism::DType;

// =============================================================================
// Helpers
// =============================================================================

// Returns the Frobenius norm of (A^T A - I_k) as a measure of how far A is
// from being a Stiefel matrix.  Perfect orthonormality gives 0.
static float stiefel_error(const Tensor& X, int k) {
    Tensor XtX  = math::matmul(math::transpose(X, {0, 2, 1}), X);   // k × k
    Tensor diff = math::subtract(XtX, math::eye(k, DType::Float32));
    Tensor err  = math::sqrt(math::sum(math::square(diff), {}));
    return static_cast<float>(math::to_double(err));
}

// Returns ||A||_F (Frobenius norm).
static float frob(const Tensor& A) {
    return static_cast<float>(math::to_double(math::sqrt(math::sum(math::square(A), {}))));
}

// Builds an n × k Stiefel frame by thin-QR of a random Gaussian matrix.
static Tensor random_stiefel(int n, int k) {
    Tensor G = math::random_normal({n, k}, DType::Float32);
    auto [Q, R] = math::qr(G);
    return math::slice(Q, 0, k, 1);   // first k columns of full Q
}

// =============================================================================
// Test utilities
// =============================================================================

static int g_passed = 0;
static int g_failed = 0;

static void check(bool cond, const char* label) {
    if (cond) {
        std::printf("  [PASS] %s\n", label);
        ++g_passed;
    } else {
        std::printf("  [FAIL] %s\n", label);
        ++g_failed;
    }
}

// =============================================================================
// Tests
// =============================================================================

// -----------------------------------------------------------------------------
// 1. Manifold membership: X^T X ≈ I_k for every sample.
// -----------------------------------------------------------------------------
static void test_manifold_membership(int n, int k, int n_samples) {
    std::printf("\n--- Manifold membership  V(%d, %d)  [%d samples] ---\n",
                n, k, n_samples);

    Tensor x_hat = random_stiefel(n, k);

    sampler::StiefelGaussianSampler::Config cfg;
    cfg.alpha = 1.0;
    //cfg.angle_cfg.burn_in = 500;

    sampler::StiefelGaussianSampler samp(x_hat, n, k, cfg);

    float max_err  = 0.f;
    float mean_err = 0.f;
    for (int i = 0; i < n_samples; ++i) {
        Tensor X = samp.sample();
        float  e = stiefel_error(X, samp.is_fat_frame() ? k : k);
        if (e > max_err) max_err = e;
        mean_err += e;
    }
    mean_err /= n_samples;

    std::printf("  max  ||X^T X - I|| = %.2e\n", static_cast<double>(max_err));
    std::printf("  mean ||X^T X - I|| = %.2e\n", static_cast<double>(mean_err));

    check(max_err  < 1e-4f, "max  Stiefel error < 1e-4");
    check(mean_err < 1e-5f, "mean Stiefel error < 1e-5");
}

/**
 * @brief Manually constructs the Stiefel identity matrix (canonical origin).
 * Bypasses the math API's square 'eye' limitation by using a CPU vector.
 */
isomorphism::Tensor stiefel_identity(int n, int k, isomorphism::DType dtype = isomorphism::DType::Float32) {
    // 1. Initialize a flat CPU vector of size n * k with zeros.
    // This represents the entire [n, k] matrix in row-major order.
    std::vector<float> data(n * k, 0.0f);

    // 2. Populate the diagonal of the upper k x k block.
    // In row-major format, the index for (row i, column i) is (i * k + i).
    for (int i = 0; i < k; ++i) {
        data[i * k + i] = 1.0f;
    }

    // 3. Translate the CPU vector into a Tensor using the routing layer.
    // This moves the data to the active backend (GPU/MKL).
    return isomorphism::math::array(data, {n, k}, dtype);
}

/**
 * @brief Benchmarks the Stiefel manifold sampler.
 * @param K The number of batches to draw.
 * @param x_hat The consensus frame [n, k].
 * @param n Ambient dimension.
 * @param k Manifold dimension.
 * @param cfg The sampler configuration (including num_samples).
 */
void run_stiefel_benchmark(int K, isomorphism::Tensor x_hat, int n, int k, sampler::StiefelGaussianSampler::Config cfg) {
    using namespace std::chrono;

    std::cout << "--- Starting Stiefel Sampler Benchmark (N=" << cfg.num_samples << ", K=" << K << ") ---" << std::endl;

    // 1. Initialisation & Burn-in
    // The constructor triggers 'run_burn_in', which uses coarse-grained OpenMP.
    auto start_init = high_resolution_clock::now();
    sampler::StiefelGaussianSampler samp(x_hat, n, k, cfg);
    auto end_init = high_resolution_clock::now();

    duration<double, std::milli> init_ms = end_init - start_init;
    std::cout << ">> Initialisation (Burn-in) Time: " << std::fixed << std::setprecision(2)
              << init_ms.count() << " ms" << std::endl;

    // 2. Sampling Loop
    std::vector<double> timings;
    timings.reserve(K);

    for (int i = 0; i < K; ++i) {
        auto start_sample = high_resolution_clock::now();

        // Generate the batch
        isomorphism::Tensor batch = samp.sample();

        // CRITICAL: Force MLX evaluation. Without this, you are only measuring
        // the time to build the computation graph, not the GPU execution time.
        isomorphism::math::eval(batch);

        auto end_sample = high_resolution_clock::now();

        duration<double, std::milli> sample_ms = end_sample - start_sample;
        timings.push_back(sample_ms.count());

        std::cout << "   Batch [" << i + 1 << "/" << K << "]: "
                  << std::setprecision(3) << sample_ms.count() << " ms" << std::endl;
    }

    // 3. Statistical Analysis
    double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    double avg = sum / K;

    double sq_sum = std::inner_product(timings.begin(), timings.end(), timings.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / K - avg * avg);

    // Throughput calculation
    double samples_per_sec = (cfg.num_samples * 1000.0) / avg;

    std::cout << "\n--- Final Performance Analysis ---" << std::endl;
    std::cout << "Average Time per Batch: " << std::setprecision(3) << avg << " ms" << std::endl;
    std::cout << "Standard Deviation:     " << std::setprecision(3) << stdev << " ms" << std::endl;
    std::cout << "Effective Throughput:   " << std::setprecision(0) << samples_per_sec << " samples/sec" << std::endl;
    std::cout << "----------------------------------" << std::endl;
}

// -----------------------------------------------------------------------------
// 2. Shape: samples have the right dimensions.
// -----------------------------------------------------------------------------
static void test_sample_shape(int n, int k) {
    std::printf("\n--- Sample shape  V(%d, %d) ---\n", n, k);

    //Tensor x_hat = random_stiefel(n, k);
    Tensor x_hat = stiefel_identity(n, k);
    sampler::StiefelGaussianSampler::Config cfg = {.alpha = 100.0, .num_samples = 1};
    sampler::StiefelGaussianSampler samp(x_hat, n, k, cfg);

    Tensor X     = samp.sample();
    auto   shape = X.shape();

    std::printf("  sample shape: [%d, %d]\n", shape[0], shape[1]);
    std::cout << "  sample =\n" << X << "\n";
    check(shape.size() == 3,    "ndim == 2");
    check(shape[1] == n,        "rows == n");
    check(shape[2] == k,        "cols == k");
}

// -----------------------------------------------------------------------------
// 3. Concentration: higher α = λ/δ² ⟹ samples closer to X_hat.
// -----------------------------------------------------------------------------
static void test_concentration(int n, int k, int n_samples) {
    std::printf("\n--- Concentration  V(%d, %d)  [%d samples] ---\n",
                n, k, n_samples);

    Tensor x_hat = random_stiefel(n, k);

    auto mean_dist = [&](double alpha) -> float {
        sampler::StiefelGaussianSampler::Config cfg;
        cfg.alpha = alpha;
        //cfg.angle_cfg.burn_in = 500;
        sampler::StiefelGaussianSampler samp(x_hat, n, k, cfg);

        float total = 0.f;
        for (int i = 0; i < n_samples; ++i) {
            Tensor X   = samp.sample();
            Tensor dif = math::subtract(X, x_hat);
            total += frob(dif);
        }
        return total / n_samples;
    };

    // Low concentration (small α = 0.1): samples spread out.
    float d_low  = mean_dist(0.1);
    // High concentration (large α = 5.0): samples close to X_hat.
    float d_high = mean_dist(5.0);

    std::printf("  mean dist from X_hat  (α=0.1): %.4f\n",
                static_cast<double>(d_low));
    std::printf("  mean dist from X_hat  (α=5.0): %.4f\n",
                static_cast<double>(d_high));

    check(d_high < d_low, "higher alpha => samples closer to X_hat");
}

// -----------------------------------------------------------------------------
// 4. Fat-frame dispatch: n < 2k triggers geometric duality.
// -----------------------------------------------------------------------------
static void test_fat_frame(int n, int k, int n_samples) {
    std::printf("\n--- Fat frame  V(%d, %d)  n < 2k=%d  [%d samples] ---\n",
                n, k, 2 * k, n_samples);

    Tensor x_hat = random_stiefel(n, k);
    sampler::StiefelGaussianSampler::Config cfg;
    //cfg.angle_cfg.burn_in = 500;
    sampler::StiefelGaussianSampler samp(x_hat, n, k, cfg);

    check(samp.is_fat_frame(), "sampler detected fat frame");

    float max_err = 0.f;
    for (int i = 0; i < n_samples; ++i) {
        Tensor X = samp.sample();
        float  e = stiefel_error(X, k);
        if (e > max_err) max_err = e;
    }
    std::printf("  max  ||X^T X - I|| = %.2e\n", static_cast<double>(max_err));
    check(max_err < 1e-4f, "fat-frame samples on manifold");
}

// -----------------------------------------------------------------------------
// 5. HMC diagnostics: acceptance rate in a reasonable range after warm-up.
// -----------------------------------------------------------------------------
static void test_hmc_acceptance(int n, int k) {
    std::printf("\n--- HMC acceptance rate  V(%d, %d) ---\n", n, k);

    Tensor x_hat = random_stiefel(n, k);
    sampler::StiefelGaussianSampler::Config cfg;
    cfg.alpha             = 1.0;
    //cfg.angle_cfg.burn_in = 2000;
    sampler::StiefelGaussianSampler samp(x_hat, n, k, cfg);

    // Draw enough samples for acceptance rate to stabilise.
    for (int i = 0; i < 200; ++i) samp.sample();

    double acc = samp.angle_acceptance_rate();
    std::printf("  acceptance rate = %.3f  (target ~0.65)\n", acc);
    check(acc > 0.30 && acc < 0.95, "acceptance rate in [0.30, 0.95]");
}

// -----------------------------------------------------------------------------
// Print a single V(n, k) sample to stdout.
// -----------------------------------------------------------------------------
static void test_print_sample(int n, int k) {
    std::printf("\n--- Single sample  V(%d, %d) ---\n", n, k);

    Tensor x_hat = stiefel_identity(n, k);

    sampler::StiefelGaussianSampler::Config cfg;
    cfg.alpha             = 1.0;
    cfg.num_samples       = 1;
    //cfg.angle_cfg.burn_in = 500;

    sampler::StiefelGaussianSampler samp(x_hat, n, k, cfg);
    Tensor X = samp.sample();

    std::cout << "  X =\n" << X << "\n";
    std::printf("  ||X^T X − I_k|| = %.2e\n",
                static_cast<double>(stiefel_error(X, k)));
}

// =============================================================================
// main
// =============================================================================

int main() {
    using namespace std::chrono;

    std::printf("========================================\n");
    std::printf(" Stiefel Gaussian Sampler — Test Suite  \n");
    std::printf("========================================\n");

    math::set_default_device_gpu();
    auto start_total = high_resolution_clock::now();

    //test_print_sample(6, 3);

    //run_stiefel_benchmark(500, stiefel_identity(500, 50), 500, 50, {.num_samples = 500});

    // --- Tall-and-skinny regime (n >= 2k) ---
    //test_sample_shape(50, 40);
/*
    test_manifold_membership(50,  5,  20);
    test_manifold_membership(100, 10, 20);
    test_manifold_membership(200, 20, 10);
    test_concentration(50, 5, 30);
    test_hmc_acceptance(50, 5);

    // --- Fat-frame regime (n < 2k) ---
    test_fat_frame(10, 7, 20);   // k' = n-k = 3,  n=10 >= 2*3=6
    test_fat_frame(20, 14, 15);  // k' = 6,         n=20 >= 12
*/
    auto end_total = high_resolution_clock::now();
    duration<double> elapsed = end_total - start_total;

    // --- Summary ---
    std::printf("\n========================================\n");
    std::printf(" Results: %d passed, %d failed\n", g_passed, g_failed);
    std::printf(" Total Execution Time: %.3f seconds\n", elapsed.count());
    std::printf("========================================\n");

    return g_failed > 0 ? 1 : 0;
}