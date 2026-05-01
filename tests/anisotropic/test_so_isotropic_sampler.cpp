#include "sampler/anisotropic/so_anisotropic_spatial_gaussian_sampler.hpp"

#include <isomorphism/math.hpp>
#include <isomorphism/tensor.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

namespace math = isomorphism::math;
using Tensor   = isomorphism::Tensor;
using DType    = isomorphism::DType;
using Sampler  = sampler::anisotropic::SOdAnisotropicSpatialGaussianSampler;

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

static std::vector<double> make_isotropic_gamma(int d, double alpha) {
    std::vector<double> G(d * d, 0.0);
    for (int i = 0; i < d; i++) G[i * d + i] = alpha;
    return G;
}

static void test_isotropic(int d, int N) {
    std::printf("\n--- Isotropic  SO(%d)  N=%d ---\n", d, N);

    const double alpha = 2.0;
    Tensor m_hat = math::eye(d, DType::Float32);

    Sampler::Config cfg;
    cfg.num_samples       = N;
    cfg.angle_cfg.d       = d;
    cfg.angle_cfg.burn_in = 1000;

    auto gamma = make_isotropic_gamma(d, alpha);
    Sampler samp(m_hat, d, gamma, cfg);

    std::printf("  gamma_min = %.4f  (expected %.4f)\n", samp.gamma_min(), alpha);

    auto t0 = std::chrono::high_resolution_clock::now();
    Tensor X = samp.sample();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  sample() time: %.2f ms\n", ms);

    float rms = so_error(X, d, N);
    std::printf("  RMS ||X^T X - I|| = %.2e\n", static_cast<double>(rms));
    std::cout << "  samples =\n" << X << "\n";

    check(std::abs(samp.gamma_min() - alpha) < 1e-4,
          "gamma_min == alpha for isotropic Gamma");
    check(rms < 1e-4f, "samples lie on SO(d) manifold");
}

int main() {
    std::printf("=================================================\n");
    std::printf("  SO(d) Isotropic Gaussian Sampler — Tests\n");
    std::printf("=================================================\n");

    sampler::set_num_threads(8);
    math::set_default_device_gpu();

    const int N = 8;

    test_isotropic(3, N);
    test_isotropic(4, N);

    std::printf("\n=================================================\n");
    std::printf(" Results: %d passed, %d failed\n", g_passed, g_failed);
    std::printf("=================================================\n");

    return g_failed > 0 ? 1 : 0;
}
