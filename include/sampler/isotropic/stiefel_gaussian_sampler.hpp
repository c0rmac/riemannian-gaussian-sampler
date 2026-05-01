#pragma once

#include "../sampler_base.hpp"
#include "principal_angle_sampler.hpp"
#include <isomorphism/tensor.hpp>
#include <isomorphism/math.hpp>
#include <memory>
#include <vector>

namespace sampler {

// Samples exactly from the isotropic Gaussian distribution on the Stiefel
// manifold V(n, k) centred at a consensus frame X_hat ∈ V(n, k):
//
//   μ_∞(X) ∝ exp(-α · d²(X, X_hat))    where α = λ / δ²
//
// This implements Algorithm 4.2 from the companion paper with FOUR major
// efficiency improvements:
//
//   1. O(k³) spectral matrix exponential
//      The Lie algebra element V(s) ∈ so(n) has non-zero entries confined to a
//      2k × 2k active core.  The matrix exponential is therefore computed on
//      this core only, then embedded into an n × n identity structure.
//
//   2. O(nk²) Phase 2 via implicit subspace projections (tall-and-skinny, n ≥ 2k)
//      Instead of constructing a full (n-k) × (n-k) stabiliser frame and an
//      n × n transport basis Q, we:
//        - Draw a thin (n-k) × k Gaussian and thin-QR to get a random k-frame
//          in the stabiliser (replaces the Haar draw from SO(n-k)).
//        - Project random Gaussian noise into X_hat^⊥ and thin-QR (replaces the
//          full orthogonal completion Q).
//      Final assembly:  X = X_hat · Θ₁₁ + U · Θ₂₁
//
//   3. Geometric duality for "fat" frames (n < 2k)
//      Direct CBO dynamics diverge in this regime (Dyson drift → -∞).  We map
//      to the dual space V(n, k') where k' = n - k < k, guaranteeing n ≥ 2k'.
//      A sample X' ∈ V(n, k') is generated near X_hat^⊥, and the final sample
//      X ∈ V(n, k) is recovered as (X')^⊥.
//
//   4. PARALLEL BATCH GENERATION (OpenMP)
//      The sampler spawns N independent Markov chains to sample N principal
//      angle vectors simultaneously. The O(k³) spectral lifts and subspace
//      projections are then mapped back to the manifold in parallel, outputting
//      a stacked tensor of shape [N, n, k].
class StiefelGaussianSampler : public SamplerBase {
public:
    struct Config {
        int    num_samples  = 1;                    // The batch size 'N' (Number of parallel samples to draw)
        double alpha        = 1.0;                  // Concentration α = λ/δ²
        isomorphism::DType dtype = isomorphism::DType::Float32;

        // HMC configuration (n, k, and alpha are set automatically; num_chains defaults to 8).
        PrincipalAngleSampler::Config angle_cfg;
    };

    // x_hat: n × k consensus (mean) frame, must satisfy X_hat^T X_hat = I_k.
    StiefelGaussianSampler(isomorphism::Tensor x_hat, int n, int k, Config cfg);

    // Generates N samples across parallel OpenMP threads.
    // If num_samples > 1: Returns a batched Tensor of shape [N, n, k].
    // If num_samples == 1: Returns an unbatched Tensor of shape [n, k].
    isomorphism::Tensor sample() override;

    isomorphism::Tensor draw_uniform();

    // Update the configuration. If alpha or angle_cfg changes, the principal
    // angle sampler is rebuilt (including a fresh parallel burn-in).
    void set_config(Config cfg);
    void set_x_hat(isomorphism::Tensor x_hat);
    void update_alpha(double alpha, int burn_in_steps = 500);
    void rebuild_angle_sampler();

    int  n()             const { return n_; }
    int  k()             const { return k_; }
    int  num_samples()   const { return cfg_.num_samples; }
    bool is_fat_frame()  const { return is_fat_; }
    double alpha()       const { return alpha_; }
    double angle_acceptance_rate() const { return angle_sampler_->acceptance_rate(); }

    // α ≥ this threshold activates the tangent-space sampler (replaces HMC Phase I).
    static constexpr double kHighAlphaThreshold = 1e6;

private:
    isomorphism::Tensor x_hat_;
    isomorphism::Tensor x_hat_prime_;   // Orthogonal complement of X_hat (fat case only, shared across threads)
    int    n_, k_;
    bool   is_fat_;
    int    k_eff_;      // k for tall-thin,  n-k for fat
    double alpha_;      // λ / δ²
    Config cfg_;

    std::unique_ptr<PrincipalAngleSampler> angle_sampler_;

    // -------------------------------------------------------------------------
    // Thread-Safe Math Helpers (Executed concurrently in OpenMP loop)
    // -------------------------------------------------------------------------

    // O(k³) spectral lift:  build V_active ∈ so(2k) from shape parameters
    // (Ω_int, θ, V_right), exponentiate, and return the two key sub-blocks:
    //   Θ₁₁ = exp(V_active)[0:k, 0:k]
    //   Θ₂₁ = exp(V_active)[k:2k, 0:k]
    struct ShapeBlocks {
        isomorphism::Tensor theta11;
        isomorphism::Tensor theta21;
        isomorphism::Tensor theta12;
        isomorphism::Tensor theta22;
    };
    ShapeBlocks compute_shape_blocks(const std::vector<double>& theta, int k);

    // Draw Ω_int ∈ so(k): skew-symmetric with strictly lower-triangular entries
    // distributed as N(0, δ²/(2λ)).
    isomorphism::Tensor draw_omega_int(int k);

    // Draw V_right ∈ O(k) uniformly via QR with column-sign correction.
    isomorphism::Tensor draw_v_right(int k);

    // Thin QR: given an m × n matrix A (m ≥ n), return Q ∈ R^{m×n} with Q^T Q = I.
    isomorphism::Tensor thin_qr(const isomorphism::Tensor& a);

    // Convert a std::vector<double> to a Float32 Tensor of shape [len].
    isomorphism::Tensor vec_to_tensor(const std::vector<double>& v);

    // High-α Phase I: Tangent-space principal angle sampler.
    //
    // When α is very large, the manifold is locally flat. The trigonometric
    // Weyl density simplifies to an algebraic form:
    //    w(θ) ≈ (∏ |θᵢ² - θⱼ²|) · ∏ |θᵢ|^(n-2k)
    //
    // This allows us to bypass HMC/Splines and sample angles as singular
    // values of a scaled Gaussian matrix in the reductive complement 𝔪:
    //
    // 1. Draw G ∈ ℝ^{(n-k)×k} with G_ij ~ N(0, 1).
    // 2. Compute the eigenvalues of M = GᵀG (or GGᵀ for fat frames).
    // 3. Extract θⱼ = sqrt(eig_j / (2α)).
    //
    // This provides exact Dyson repulsion and boundary interaction behavior
    // with O(k³) complexity per batch.
    std::vector<double> sample_angles_tangent();
};

} // namespace sampler