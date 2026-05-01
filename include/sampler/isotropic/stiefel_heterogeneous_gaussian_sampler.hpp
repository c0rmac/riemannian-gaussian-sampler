#pragma once

#include "principal_angle_sampler.hpp"
#include <isomorphism/tensor.hpp>
#include <isomorphism/math.hpp>
#include <memory>
#include <vector>

namespace sampler {

// Samples N particles simultaneously from the isotropic Stiefel Gibbs measure,
// where each particle i has its own concentration alpha_i:
//
//   X_i ~ μ_∞(X) ∝ exp(−alpha_i · d²(X, X̂))
//
// All particles share the same consensus frame X̂ ∈ V(n, k).
//
// --- Typical usage ---
//
//   StiefelHeterogeneousGaussianSampler s(cfg);
//   s.update_alphas(alphas, burn_in);   // initialise / update concentrations
//   s.set_x_hat(consensus);             // update consensus (cheap for tall frames;
//                                       // recomputes dual for fat frames)
//   Tensor X = s.sample();              // draw [N, n, k]
//
// update_alphas() rebuilds chains (in parallel) only for particles whose alpha
// changed by more than cfg.alpha_rtol, then runs the requested burn-in.
// Particles at or above kHighAlphaThreshold skip HMC entirely and use the
// tangent-space Wishart sampler.
// sample() does no alpha logic: Phase I HMC continues from current chain states;
// Phase II builds the batched spectral lift.
//
// --- Fat frames (n < 2k) ---
// Geometric duality is applied automatically: particles are sampled on V(n, n-k)
// near the dual consensus X̂^⊥ and projected back to V(n, k).
class StiefelHeterogeneousGaussianSampler {
public:
    struct Config {
        int    N;
        int    n, k;
        isomorphism::DType dtype  = isomorphism::DType::Float32;
        int    leapfrog_steps     = 5;
        int    num_threads        = 8;
        double alpha_rtol         = 0.01;
    };

    explicit StiefelHeterogeneousGaussianSampler(Config cfg);

    // Update per-particle concentrations and run burn-in on changed chains.
    // alphas must contain exactly cfg.N values.
    // Particles at or above kHighAlphaThreshold skip HMC management.
    void update_alphas(const std::vector<double>& alphas, int burn_in_steps);

    // Update the shared consensus frame.
    // For fat frames (n < 2k), this also recomputes the dual frame X̂^⊥.
    void set_x_hat(const isomorphism::Tensor& x_hat);

    // Draw N samples from the current chain states and consensus.
    // Returns [N, n, k]. Must be called after at least one update_alphas().
    isomorphism::Tensor sample();

    static constexpr double kHighAlphaThreshold = 1e6;

private:
    Config cfg_;
    int n_, k_, k_eff_, N_;
    bool is_fat_;

    isomorphism::Tensor x_hat_;
    isomorphism::Tensor x_hat_prime_;   // orthogonal complement of x_hat (fat frames only)

    std::vector<double> alphas_;
    std::vector<std::unique_ptr<PrincipalAngleSampler>> angle_samplers_;

    // Phase I helpers
    void sample_hmc_parallel(const std::vector<int>& hmc_indices,
                             std::vector<double>&    flat_theta);

    void sample_tangent_batch(const std::vector<int>& tan_indices,
                              std::vector<double>&    flat_theta);

    // Phase II helpers
    struct ShapeBlocks {
        isomorphism::Tensor theta11, theta21, theta12, theta22;
    };
    ShapeBlocks compute_shape_blocks(const std::vector<double>& flat_theta, int k);

    // Per-particle Ω_int: skew-symmetric [N, k, k] scaled by per-particle alpha.
    isomorphism::Tensor draw_omega_int(int k);

    // Uniform O(k) matrix [N, k, k] via QR with diagonal-sign correction.
    isomorphism::Tensor draw_v_right(int k);

    // Thin QR: returns the first n_cols columns of the full Q factor.
    isomorphism::Tensor thin_qr(const isomorphism::Tensor& a);

    // Recompute x_hat_prime_ from the current x_hat_ (fat frames only).
    void update_x_hat_prime();

    isomorphism::Tensor vec_to_tensor(const std::vector<double>& v);
};

} // namespace sampler
