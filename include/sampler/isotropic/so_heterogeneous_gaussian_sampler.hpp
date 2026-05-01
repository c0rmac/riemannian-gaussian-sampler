#pragma once

#include "so_angle_sampler.hpp"
#include <isomorphism/tensor.hpp>
#include <isomorphism/math.hpp>
#include <memory>
#include <vector>

namespace sampler {

// Samples N particles simultaneously from the isotropic SO(d) Gibbs measure,
// where each particle i has its own concentration alpha_i:
//
//   X_i ~ μ_∞(X) ∝ exp(−alpha_i · d_g²(X, M̂))
//
// All particles share the same consensus point M̂ ∈ SO(d).
//
// --- Typical usage ---
//
//   SOdHeterogeneousGaussianSampler s(cfg);
//   s.update_alphas(alphas, burn_in);   // initialise / update concentrations
//   s.set_m_hat(consensus);             // update consensus (cheap)
//   Tensor X = s.sample();              // draw [N, d, d]
//
// --- Chain pooling ---
//
// Particles are clustered by alpha value (within cfg.alpha_rtol tolerance).
// Each distinct-alpha cluster shares ONE SOdAngleSampler with cfg.num_chains
// internal HMC chains.  sample_angles(M) draws M angle sets from those chains,
// amortising burn-in and chain state across the M particles in the cluster.
//
// This means K components each of size N require only K chains (not K*N),
// matching the cost of the original single-alpha SOdGaussianSampler.
//
// Old chains are reused across update_alphas() calls when their representative
// alpha is stable; a new chain (with fresh burn-in) is only created when a
// cluster's alpha moves outside cfg.alpha_rtol of the stored chain alpha.
//
// --- Pipeline ---
//
// Phase I  (CPU, parallel OpenMP over groups):
//   Each group's shared chain draws M angle sets via sample_angles(M).
//   High-alpha particles (alpha_i ≥ kHighAlphaThreshold) bypass HMC and use
//   the tangent-space batch sampler instead.
//
// Phase II (GPU, batched over all N particles):
//   1. build_canonical_rotation  → Θ  [N, d, d]
//   2. draw_haar_od              → Q  [N, d, d]
//   3. X = Q Θ Qᵀ M̂            → [N, d, d]
class SOdHeterogeneousGaussianSampler {
public:
    struct Config {
        int    N;
        int    d;
        isomorphism::DType dtype  = isomorphism::DType::Float32;
        int    leapfrog_steps     = 5;
        int    num_threads        = 8;    // OpenMP threads for Phase I group loop
        int    num_chains         = 8;    // HMC chains per alpha group
        double alpha_rtol         = 0.01; // tolerance for grouping and chain reuse
    };

    explicit SOdHeterogeneousGaussianSampler(Config cfg);

    // Update per-particle concentrations.
    // Particles are re-clustered by alpha; chains are reused when the cluster's
    // representative alpha is stable, or rebuilt (with burn_in_steps warm-up) when it moves.
    // Particles at or above kHighAlphaThreshold are excluded from HMC management.
    void update_alphas(const std::vector<double>& alphas, int burn_in_steps);

    // Update the shared consensus point (no HMC work, just stores the tensor).
    void set_m_hat(const isomorphism::Tensor& m_hat);

    // Draw N samples from the current chain states and consensus.
    // Returns [N, d, d]. Must be called after at least one update_alphas().
    isomorphism::Tensor sample();

    static constexpr double kHighAlphaThreshold = 1e9;

private:
    Config cfg_;
    int d_, m_, N_;

    isomorphism::Tensor m_hat_;
    std::vector<double> alphas_;  // per-particle alphas (for Phase II tangent routing)

    // Each group clusters particles with similar alpha and owns one shared chain.
    struct AlphaGroup {
        double                           alpha;        // representative alpha
        std::vector<int>                 particle_idx; // particle indices in this group
        std::unique_ptr<SOdAngleSampler> chain;        // shared multi-chain HMC sampler
    };
    std::vector<AlphaGroup> groups_;

    // Phase I helpers
    void sample_hmc_groups(std::vector<double>& flat_theta);

    void sample_tangent_batch(const std::vector<int>& tan_indices,
                              std::vector<double>&    flat_theta);

    // Phase II helpers
    isomorphism::Tensor build_canonical_rotation(const std::vector<double>& flat_theta,
                                                 int N);
    isomorphism::Tensor draw_haar_od(int N);
};

} // namespace sampler
