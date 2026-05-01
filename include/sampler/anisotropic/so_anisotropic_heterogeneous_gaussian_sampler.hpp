#pragma once

#include "so_anisotropic_angle_sampler.hpp"
#include "so_hypersphere_sampler_cpu.hpp"
#include <isomorphism/tensor.hpp>
#include <isomorphism/math.hpp>
#include <memory>
#include <vector>

namespace sampler {
namespace anisotropic {

// Algorithm 4.8 (heterogeneous batch): N independent anisotropic SO(d)
// Gaussians sampled simultaneously, each with its own precision matrix Γ_i.
//
// Each particle i is drawn from
//
//   μ_∞,i(X) ∝ exp(−⟨Log_{M̂}(X), Γ_i Log_{M̂}(X)⟩_g)
//
// where M̂ ∈ SO(d) is the shared consensus rotation.
//
// --- Pipeline ---
//
// Phase I  (parallel HMC, one SOdAnisotropicAngleSampler per particle):
//   Each particle draws θ_i from its own isotropic envelope
//     p_upper,i(θ) ∝ w(θ) exp(−γ_min,i ‖θ‖²)
//   using m = ⌊d/2⌋ independent 2D HMC chains.  Particles are processed in
//   parallel across threads; each particle's m chains run serially within it
//   to avoid oversubscription with the outer parallel loop.
//
// Phase II (parallel CPU, one SOdHypersphereSamplerCPU per particle):
//   Each particle builds its orientation frame Q_i ∈ SO(d) from (θ_i, Γ_i)
//   via the column-level vMF/IMH sampler.  Because Γ_i differs per particle,
//   the GPU batching available in SOdHypersphereSamplerGPU does not apply here;
//   this phase is CPU-only and parallelised across particles.
//
// Phase III (batched GPU tensor operations):
//   Θ_i = diag(R(θ₁),…,R(θₘ),[1 if d odd])  for each i  →  [N, d, d]
//   X_i = Q_i Θ_i Q_iᵀ M̂                                  →  [N, d, d]
//
// --- Chain management ---
//
// update_gammas() classifies each particle into one of three buckets:
//   NEW     — no chain yet: construct and burn in from scratch.
//   CHANGED — Γ changed by more than cfg.gamma_frob_rtol (relative Frobenius):
//             warm-start the existing chain in place via update_gamma().
//   SAME    — skip entirely; chain advances freely in sample().
// Construction and warm-starts both run in parallel across particles.
class SOdAnisotropicHeterogeneousGaussianSampler {
public:
    struct Config {
        int N;
        int d;
        int num_threads = 8;
        isomorphism::DType dtype = isomorphism::DType::Float32;

        // Relative Frobenius-norm change in Γ that triggers a chain warm-start.
        double gamma_frob_rtol = 0.01;

        // Per-particle angle-sampler HMC settings.
        int    num_chains      = 1;
        int    burn_in         = 2000;
        int    leapfrog_steps  = 5;
        double init_epsilon    = 1e-4;
        double target_accept   = 0.65;

        // Phase II CPU hypersphere config shared across particles
        // (num_threads and seed are overridden per particle).
        SOdHypersphereSamplerCPU::Config cpu_cfg;
    };

    explicit SOdAnisotropicHeterogeneousGaussianSampler(Config cfg);

    // Update per-particle precision matrices.
    //
    // gammas[i] must be a flat row-major D×D symmetric positive-definite matrix
    // in the Lie algebra so(d), where D = d*(d-1)/2.
    // Chains are rebuilt in parallel only for particles whose Γ changed beyond
    // cfg.gamma_frob_rtol.  burn_in_steps controls the warm-start length.
    void update_gammas(const std::vector<std::vector<double>>& gammas,
                       int burn_in_steps);

    // Update the shared consensus rotation.  No HMC work — just stores the tensor.
    void set_m_hat(const isomorphism::Tensor& m_hat);

    // Draw one sample per particle from the current chain states.
    // Returns shape [N, d, d].
    // Must be called after at least one update_gammas().
    isomorphism::Tensor sample();

    int N() const { return N_; }
    int d() const { return d_; }

    // Per-particle Phase I acceptance rates (average over m channels each).
    std::vector<double> angle_acceptance_rates() const;

private:
    Config cfg_;
    int d_, m_, N_;

    isomorphism::Tensor m_hat_;

    // Per-particle state.
    std::vector<std::vector<double>>                         gammas_;
    std::vector<double>                                      gamma_mins_;
    std::vector<std::unique_ptr<SOdAnisotropicAngleSampler>> angle_samplers_;
    std::vector<std::unique_ptr<SOdHypersphereSamplerCPU>>   phase2_samplers_;

    // Returns true if Γ_new differs from the stored Γ_i by more than
    // cfg_.gamma_frob_rtol in relative Frobenius norm.
    bool gamma_changed(int i, const std::vector<double>& new_gamma) const;

    // Phase I: advance each particle's angle chain by one sample step.
    void sample_angles_parallel(std::vector<double>& flat_theta);

    // Phase II: build one orientation frame per particle using its own Γ.
    // Returns flat [N * d * d] in row-major order.
    std::vector<double> build_frames_parallel(const std::vector<double>& flat_theta);

    // Phase III: construct batched block-diagonal rotation [N, d, d] from angles.
    isomorphism::Tensor build_canonical_rotation(const std::vector<double>& flat_theta) const;
};

} // namespace anisotropic
} // namespace sampler
