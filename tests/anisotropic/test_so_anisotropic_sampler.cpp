#include "sampler/anisotropic/so_anisotropic_spatial_gaussian_sampler.hpp"

#include <isomorphism/math.hpp>
#include <isomorphism/tensor.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>

namespace math = isomorphism::math;
using Tensor   = isomorphism::Tensor;
using DType    = isomorphism::DType;
using Sampler  = sampler::anisotropic::SOdAnisotropicSpatialGaussianSampler;

// =============================================================================
// Validation Helpers
// =============================================================================

static float so_error(const Tensor& X, int d, int N = 1) {
    Tensor XtX  = math::matmul(math::transpose(X, {0, 2, 1}), X);
    Tensor diff = math::subtract(XtX, math::eye(d, DType::Float32));
    float  total = static_cast<float>(math::to_double(
        math::sqrt(math::sum(math::square(diff), {}))));
    return (N > 1) ? total / std::sqrt(static_cast<float>(N)) : total;
}

static int g_passed = 0;
static int g_failed = 0;

static void check(bool cond, const char* label) {
    if (cond) { std::printf("  [PASS] %s\n", label); ++g_passed; }
    else       { std::printf("  [FAIL] %s\n", label); ++g_failed; }
}

// =============================================================================
// Generalized Test Suite
// =============================================================================

static void run_test_case(int d, int N, const std::string& name, const std::vector<double>& G) {
    std::printf("\n--- %s  SO(%d)  N=%d ---\n", name.c_str(), d, N);

    Sampler::Config cfg;
    cfg.num_samples       = N;
    cfg.angle_cfg.d       = d;
    cfg.angle_cfg.burn_in = 2000;

    Sampler samp(math::eye(d, DType::Float32), d, G, cfg);

    auto t0 = std::chrono::high_resolution_clock::now();
    Tensor X = samp.sample();
    auto t1 = std::chrono::high_resolution_clock::now();

    float rms = so_error(X, d, N);
    std::printf("  sample() time: %.2f ms | RMS Error: %.2e\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                static_cast<double>(rms));

    // Print first sample for visual inspection
    //if (d <= 6) std::cout << "  first sample =\n" << math::slice(X, 0, 1, 0) << "\n";
    std::cout << X << std::endl;

    check(rms < 1e-4f, (name + " manifold integrity").c_str());
}

// =============================================================================
// Precision Matrix Generators
// =============================================================================

static std::vector<double> make_needle(int d, double high = 5000.0, double low = 0.1) {
    std::vector<double> G(static_cast<size_t>(d) * d, 0.0);
    for (int i = 0; i < d; i++) G[i * d + i] = (i == d - 1) ? high : low;
    return G;
}

static std::vector<double> make_exponential(int d, double base = 0.1, double factor = 10.0) {
    std::vector<double> G(static_cast<size_t>(d) * d, 0.0);
    for (int i = 0; i < d; i++) G[i * d + i] = base * std::pow(factor, i);
    return G;
}

static std::vector<double> make_tiers(int d) {
    std::vector<double> G(static_cast<size_t>(d) * d, 0.0);
    for (int i = 0; i < d; i++) {
        if (i < d / 3)          G[i * d + i] = 0.1;    // Tier 1: Loose
        else if (i < 2 * d / 3) G[i * d + i] = 100.0;  // Tier 2: Medium
        else                    G[i * d + i] = 5000.0; // Tier 3: Strict
    }
    return G;
}

static std::vector<double> make_rotated_needle(int d, double high = 5000.0) {
    std::vector<double> G(static_cast<size_t>(d) * d, 0.0);
    // Diagonal needle in D
    std::vector<double> D(static_cast<size_t>(d) * d, 0.0);
    for(int i = 0; i < d; i++) D[i * d + i] = (i == 0) ? high : 0.1;

    // 45-degree rotation in the first plane (0,1)
    std::vector<double> R(static_cast<size_t>(d) * d, 0.0);
    for(int i = 0; i < d; i++) R[i * d + i] = 1.0;
    double c = std::cos(M_PI / 4.0), s = std::sin(M_PI / 4.0);
    R[0] = c; R[1] = -s; R[d] = s; R[d+1] = c;

    // G = R * D * R^T
    for(int i = 0; i < d; i++)
        for(int j = 0; j < d; j++)
            for(int k = 0; k < d; k++)
                for(int l = 0; l < d; l++)
                    G[i*d + j] += R[i*d + k] * D[k*d + l] * R[j*d + l];
    return G;
}

static void test_fully_correlated_anisotropy(int d, int N) {
    std::printf("\n--- Test 7: Fully Correlated (Dense Gamma)  SO(%d)  N=%d ---\n", d, N);

    // 1. Create an extremely anisotropic diagonal spectrum D
    // We want a mix of very loose and very tight constraints.
    std::vector<double> D(static_cast<size_t>(d) * d, 0.0);
    for (int i = 0; i < d; i++) {
        D[i * d + i] = (i < d / 2) ? 0.1 : 5000.0;
    }

    // 2. Create a dense, "scrambling" rotation matrix R.
    // We'll use a sequence of Givens rotations to ensure every entry is non-zero.
    std::vector<double> R(static_cast<size_t>(d) * d, 0.0);
    for (int i = 0; i < d; i++) R[i * d + i] = 1.0;

    auto apply_givens = [&](int i, int j, double theta) {
        double c = std::cos(theta), s = std::sin(theta);
        for (int k = 0; k < d; k++) {
            double r_ik = R[i * d + k], r_jk = R[j * d + k];
            R[i * d + k] = c * r_ik - s * r_jk;
            R[j * d + k] = s * r_ik + c * r_jk;
        }
    };

    // Scramble the basis!
    for (int i = 0; i < d - 1; i++) apply_givens(i, i + 1, 0.5 + i * 0.1);

    // 3. Compute G = R * D * R^T (Full Dense Matrix)
    std::vector<double> G(static_cast<size_t>(d) * d, 0.0);
    std::vector<double> tmp(static_cast<size_t>(d) * d, 0.0);

    // tmp = D * R^T
    for(int i=0; i<d; i++)
        for(int j=0; j<d; j++)
            tmp[i*d + j] = D[i*d + i] * R[j*d + i]; // R^T means index swap

    // G = R * tmp
    for(int i=0; i<d; i++)
        for(int j=0; j<d; j++)
            for(int k=0; k<d; k++)
                G[i*d + j] += R[i*d + k] * tmp[k*d + j];

    // 4. Sample!
    Sampler::Config cfg;
    cfg.num_samples = N; cfg.angle_cfg.d = d; cfg.angle_cfg.burn_in = 2500;
    Sampler samp(math::eye(d, DType::Float32), d, G, cfg);

    auto t0 = std::chrono::high_resolution_clock::now();
    Tensor X = samp.sample();
    auto t1 = std::chrono::high_resolution_clock::now();

    float rms = so_error(X, d, N);
    std::printf("  sample() time: %.2f ms | RMS Error: %.2e\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                static_cast<double>(rms));

    //if (d <= 6) std::cout << "  First Sample (Dense Gamma Result):\n" << math::slice(X, 0, 1, 0) << "\n";

    std::cout << X << std::endl;

    check(rms < 1e-4f, "Fully correlated samples remain orthogonal");
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::printf("=================================================\n");
    std::printf("  SO(d) Universal Anisotropic Sampler Tests\n");
    std::printf("=================================================\n");

    sampler::set_num_threads(8);
    math::set_default_device_cpu();

    const int N = 1;
    std::vector<int> test_dims = {3, 4, 6, 8}; // Stress test various d
    //std::vector<int> test_dims = {50}; // Stress test various d

    test_fully_correlated_anisotropy(3, 8);
/*
    for (int d : test_dims) {
        std::printf("\n>>> TESTING DIMENSION d = %d <<<\n", d);
        run_test_case(d, N, "Needle",      make_needle(d));
        run_test_case(d, N, "Exponential", make_exponential(d));
        run_test_case(d, N, "3-Tier",      make_tiers(d));
        run_test_case(d, N, "Rotated",     make_rotated_needle(d));
    }
*/
    std::printf("\n=================================================\n");
    std::printf(" Final Results: %d passed, %d failed\n", g_passed, g_failed);
    std::printf("=================================================\n");

    return g_failed > 0 ? 1 : 0;
}