#pragma once

#include "sampler/xoshiro.hpp"
#include <isomorphism/tensor.hpp>
#include <isomorphism/math.hpp>
#include <random>
#include <vector>

namespace sampler {
namespace anisotropic {

// Phase II of Algorithm 4.8 — GPU implementation (isomorphism backend).
//
// Builds orientation frames Q ∈ SO(d) using batched GPU tensor operations via
// the isomorphism framework.  The sequential column loop runs on CPU (d−1
// iterations), but within each iteration all N sample proposals are evaluated
// in parallel on the GPU.
//
// Per-column rejection strategy:
//   1. Draw proposals [N, d] on GPU and apply batched Gram-Schmidt.
//   2. Compute acceptance weights q^T Γ q for all N in one GPU pass.
//   3. Read back the N weight values to CPU; decide acceptance with
//      a fast Xoshiro RNG (no GPU-side branching required).
//   4. Repeat for any samples that did not accept, up to max_retry_loops.
//
// The last column is determined on CPU by orthogonality + det = +1.
class SOdHypersphereSamplerGPU {
public:
    struct Config {
        int    max_retry_loops = 64;
        isomorphism::DType dtype = isomorphism::DType::Float32;
        uint64_t seed            = static_cast<uint64_t>(std::random_device{}());
    };

    // gamma_flat: row-major d×d precision matrix Γ (symmetric, ≻ 0).
    // gamma_min:  smallest eigenvalue of Γ.
    SOdHypersphereSamplerGPU(int d,
                              const std::vector<double>& gamma_flat,
                              double gamma_min,
                              Config cfg);

    // Build orientation frames for N samples.
    //
    // flat_theta : length N*m, sample n occupies [n*m, (n+1)*m).
    // Returns    : Tensor of shape [N, d, d] (or [d, d] when N == 1).
    isomorphism::Tensor build_orientation_frames(const std::vector<double>& flat_theta,
                                                  int N);

    void update_gamma(const std::vector<double>& gamma_flat, double gamma_min);

private:
    int    d_, m_;
    double gamma_min_;
    Config cfg_;

    isomorphism::Tensor gamma_t_;   // [d, d] float32

    // Upload a flat double vector as a float32 tensor with the given shape.
    isomorphism::Tensor upload(const std::vector<double>& v,
                                std::initializer_list<int> shape) const;

    // Read back all N float values from a [N] tensor into a CPU vector.
    std::vector<float> readback_1d(const isomorphism::Tensor& t, int N) const;

    // Read back an [N, d] tensor into a flat CPU vector of length N*d.
    std::vector<float> readback_2d(const isomorphism::Tensor& t, int N, int d) const;
};

} // namespace anisotropic
} // namespace sampler
