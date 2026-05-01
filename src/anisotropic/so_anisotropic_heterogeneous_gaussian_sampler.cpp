#include "sampler/anisotropic/so_anisotropic_heterogeneous_gaussian_sampler.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <omp.h>
#include <vector>

namespace sampler {
namespace anisotropic {

namespace math = isomorphism::math;
using Tensor   = isomorphism::Tensor;

// Convert a D×D Lie algebra precision matrix to the d×d spatial diagonal
// surrogate used by the Phase I angle sampler.  Mirrors compute_surrogate()
// in SOdHypersphereSamplerCPU but operates on a locally supplied matrix.
static std::vector<double> lie_to_spatial_diagonal(
    const std::vector<double>& gamma_DxD, int d)
{
    const int D = d * (d - 1) / 2;
    std::vector<double> spatial(static_cast<size_t>(d) * d, 0.0);

    if (d == 2) {
        const double val = gamma_DxD[0];  // only one basis element
        spatial[0] = val;
        spatial[3] = val;
        return spatial;
    }

    std::vector<double> gamma_plane(D);
    for (int a = 0; a < D; a++)
        gamma_plane[a] = gamma_DxD[a * D + a];

    std::vector<double> G(d, 0.0);
    double G_tot = 0.0;
    int a = 0;
    for (int j = 0; j < d; j++) {
        for (int k = j + 1; k < d; k++, a++) {
            G[j]  += gamma_plane[a];
            G[k]  += gamma_plane[a];
            G_tot += gamma_plane[a];
        }
    }

    const double S = 2.0 * G_tot / static_cast<double>(d - 1);
    for (int j = 0; j < d; j++) {
        const double c_j = std::max(1e-8, (2.0 * G[j] - S) / static_cast<double>(d - 2));
        spatial[j * d + j] = c_j;
    }
    return spatial;
}

// =============================================================================
// Construction
// =============================================================================

SOdAnisotropicHeterogeneousGaussianSampler::SOdAnisotropicHeterogeneousGaussianSampler(
    Config cfg)
    : cfg_(cfg), d_(cfg.d), m_(cfg.d / 2), N_(cfg.N)
{
    assert(d_ >= 2);
    assert(N_ >= 1);

    // All per-particle state is built lazily on the first update_gammas() call.
    angle_samplers_.resize(N_);
    phase2_samplers_.resize(N_);
    gammas_.resize(N_);              // empty vectors — sentinel "not yet set"
    gamma_mins_.assign(N_, -1.0);
}

// =============================================================================
// gamma_changed() — relative Frobenius test
// =============================================================================

bool SOdAnisotropicHeterogeneousGaussianSampler::gamma_changed(
    int i, const std::vector<double>& new_gamma) const
{
    const auto& old = gammas_[i];
    if (old.empty()) return true;   // never set before

    const int D  = d_ * (d_ - 1) / 2;
    const int DD = D * D;
    double diff2 = 0.0, norm2 = 0.0;
    for (int k = 0; k < DD; ++k) {
        const double e = new_gamma[k] - old[k];
        diff2 += e * e;
        norm2 += old[k] * old[k];
    }
    return (diff2 / (norm2 + 1e-14)) >
           cfg_.gamma_frob_rtol * cfg_.gamma_frob_rtol;
}

// =============================================================================
// update_gammas()
// =============================================================================

void SOdAnisotropicHeterogeneousGaussianSampler::update_gammas(
    const std::vector<std::vector<double>>& gammas,
    int burn_in_steps)
{
    assert(static_cast<int>(gammas.size()) == N_);
    const int D = d_ * (d_ - 1) / 2;

    // Classify particles before updating stored gammas.
    std::vector<int> init_idx;
    std::vector<int> update_idx;

    for (int i = 0; i < N_; ++i) {
        assert(static_cast<int>(gammas[i].size()) == D * D);

        const bool is_new     = !angle_samplers_[i];
        //const bool has_change = !is_new && gamma_changed(i, gammas[i]);
        const bool has_change = !is_new;

        // Store new gamma now so parallel loops below can read gammas_[i].
        gammas_[i] = gammas[i];

        if (is_new)          init_idx.push_back(i);
        else if (has_change) update_idx.push_back(i);
    }

    // -------------------------------------------------------------------------
    // First-time construction — build and burn in from scratch (parallel).
    //
    // num_threads is set to 1 for both inner samplers so that the outer
    // parallel-for across particles is the sole source of thread-level
    // parallelism.  Nested threading would oversubscribe the CPU.
    // -------------------------------------------------------------------------
    const int n_init = static_cast<int>(init_idx.size());
    //#pragma omp parallel for schedule(static) num_threads(cfg_.num_threads)
    for (int ii = 0; ii < n_init; ++ii) {
        const int i = init_idx[ii];

        // Phase I needs a d×d spatial gamma; derive it from the D×D Lie gamma.
        std::vector<double> gamma_spatial = lie_to_spatial_diagonal(gammas_[i], d_);

        // --- Phase I sampler ---
        SOdAnisotropicAngleSampler::Config acfg;
        acfg.d              = d_;
        acfg.num_chains     = cfg_.num_chains;
        acfg.num_threads    = 1;    // outer loop owns the threads
        acfg.burn_in        = burn_in_steps;
        acfg.leapfrog_steps = cfg_.leapfrog_steps;
        acfg.init_epsilon   = cfg_.init_epsilon;
        acfg.target_accept  = cfg_.target_accept;
        acfg.seed = static_cast<uint64_t>(i) * 6364136223846793005ULL + 1;

        angle_samplers_[i] = std::make_unique<SOdAnisotropicAngleSampler>(
            gamma_spatial, acfg);
        gamma_mins_[i] = angle_samplers_[i]->gamma_min();

        // --- Phase II sampler (receives full D×D Lie gamma) ---
        SOdHypersphereSamplerCPU::Config pcfg = cfg_.cpu_cfg;
        pcfg.num_threads = 1;
        pcfg.seed        = static_cast<uint64_t>(i) * 1234567891011ULL + 42;

        phase2_samplers_[i] = std::make_unique<SOdHypersphereSamplerCPU>(
            d_, gammas_[i], gamma_mins_[i], pcfg);
    }

    // -------------------------------------------------------------------------
    // Warm-start for changed gammas — update in-place without reconstruction.
    // -------------------------------------------------------------------------
    const int n_upd = static_cast<int>(update_idx.size());
    //#pragma omp parallel for schedule(static) num_threads(cfg_.num_threads)
    for (int ii = 0; ii < n_upd; ++ii) {
        const int i = update_idx[ii];
        std::vector<double> gamma_spatial = lie_to_spatial_diagonal(gammas_[i], d_);
        angle_samplers_[i]->update_gamma(gamma_spatial, burn_in_steps);
        gamma_mins_[i] = angle_samplers_[i]->gamma_min();
        phase2_samplers_[i]->update_gamma(gammas_[i], gamma_mins_[i]);
    }
}

// =============================================================================
// set_m_hat()
// =============================================================================

void SOdAnisotropicHeterogeneousGaussianSampler::set_m_hat(const Tensor& m_hat) {
    m_hat_ = m_hat;
}

// =============================================================================
// sample()
// =============================================================================

Tensor SOdAnisotropicHeterogeneousGaussianSampler::sample() {
    // Phase I: draw one angle vector per particle.
    std::vector<double> flat_theta(static_cast<size_t>(N_) * m_, 0.0);
    sample_angles_parallel(flat_theta);

    // Phase II: build one orientation frame per particle using its own Γ.
    std::vector<double> flat_q = build_frames_parallel(flat_theta);

    // Upload Q [N, d, d].
    std::vector<float> flat_q_f32(flat_q.begin(), flat_q.end());
    Tensor Q = math::array(flat_q_f32, {N_, d_, d_}, cfg_.dtype);

    // Phase III: X_i = Q_i Θ_i Q_iᵀ M̂  (batched GPU ops).
    Tensor Theta   = build_canonical_rotation(flat_theta);       // [N, d, d]
    Tensor QT      = math::transpose(Q, {0, 2, 1});              // [N, d, d]
    Tensor g_tilde = math::matmul(math::matmul(Q, Theta), QT);  // [N, d, d]
    Tensor m_hat_b = math::expand_dims(m_hat_, {0});             // [1, d, d]
    Tensor X       = math::matmul(g_tilde, m_hat_b);             // [N, d, d]
    math::eval(X);
    return X;
}

// =============================================================================
// Phase I: parallel HMC — one sample step per particle
// =============================================================================

void SOdAnisotropicHeterogeneousGaussianSampler::sample_angles_parallel(
    std::vector<double>& flat_theta)
{
    #pragma omp parallel for schedule(dynamic) num_threads(cfg_.num_threads)
    for (int i = 0; i < N_; ++i) {
        // sample_angles(1) returns one m-dimensional angle vector.
        std::vector<double> theta_i = angle_samplers_[i]->sample_angles(1);
        for (int j = 0; j < m_; ++j)
            flat_theta[static_cast<size_t>(i) * m_ + j] = theta_i[j];
    }
}

// =============================================================================
// Phase II: parallel frame building — one frame per particle with its own Γ
// =============================================================================

std::vector<double> SOdAnisotropicHeterogeneousGaussianSampler::build_frames_parallel(
    const std::vector<double>& flat_theta)
{
    std::vector<double> flat_q(static_cast<size_t>(N_) * d_ * d_);

    #pragma omp parallel for schedule(dynamic) num_threads(cfg_.num_threads)
    for (int i = 0; i < N_; ++i) {
        // Copy this particle's m angles into a small local vector.
        std::vector<double> theta_i(
            flat_theta.begin() + static_cast<ptrdiff_t>(i) * m_,
            flat_theta.begin() + static_cast<ptrdiff_t>(i + 1) * m_);

        // build_orientation_frames with N=1 calls build_one_frame once serially
        // (the inner omp parallel for over N degenerates to a single iteration).
        std::vector<double> q_i =
            phase2_samplers_[i]->build_orientation_frames(theta_i, 1);

        std::copy(q_i.begin(), q_i.end(),
                  flat_q.begin() + static_cast<ptrdiff_t>(i) * d_ * d_);
    }
    return flat_q;
}

// =============================================================================
// Phase III: build_canonical_rotation [N, d, d]
// =============================================================================

Tensor SOdAnisotropicHeterogeneousGaussianSampler::build_canonical_rotation(
    const std::vector<double>& flat_theta) const
{
    const bool odd = (d_ % 2 == 1);
    const int  d   = d_;
    const int  m   = m_;
    const int  N   = N_;

    std::vector<float> cos_vals(static_cast<size_t>(N) * m);
    std::vector<float> sin_vals(static_cast<size_t>(N) * m);
    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < N * m; i++) {
        cos_vals[i] = static_cast<float>(std::cos(flat_theta[i]));
        sin_vals[i] = static_cast<float>(std::sin(flat_theta[i]));
    }
    Tensor cos_t = math::array(cos_vals, {N, m}, cfg_.dtype);
    Tensor sin_t = math::array(sin_vals, {N, m}, cfg_.dtype);

    std::vector<Tensor> all_rows;
    all_rows.reserve(d);

    for (int j = 0; j < m; j++) {
        Tensor c_j  = math::slice(cos_t, j, j + 1, 1);
        Tensor s_j  = math::slice(sin_t, j, j + 1, 1);
        Tensor ns_j = math::multiply(s_j, Tensor(-1.0f, cfg_.dtype));

        Tensor c_e  = math::expand_dims(c_j,  {2});
        Tensor s_e  = math::expand_dims(s_j,  {2});
        Tensor ns_e = math::expand_dims(ns_j, {2});

        std::vector<Tensor> row0, row1;

        if (2 * j > 0) {
            Tensor lz = math::full({N, 1, 2 * j}, 0.0f, cfg_.dtype);
            row0.push_back(lz);
            row1.push_back(lz);
        }

        row0.push_back(c_e);
        row0.push_back(ns_e);
        row1.push_back(s_e);
        row1.push_back(c_e);

        const int right = 2 * (m - j - 1);
        if (right > 0) {
            Tensor rz = math::full({N, 1, right}, 0.0f, cfg_.dtype);
            row0.push_back(rz);
            row1.push_back(rz);
        }
        if (odd) {
            Tensor oz = math::full({N, 1, 1}, 0.0f, cfg_.dtype);
            row0.push_back(oz);
            row1.push_back(oz);
        }

        all_rows.push_back(math::concatenate(row0, 2));
        all_rows.push_back(math::concatenate(row1, 2));
    }

    if (odd) {
        std::vector<Tensor> last;
        if (d - 1 > 0)
            last.push_back(math::full({N, 1, d - 1}, 0.0f, cfg_.dtype));
        last.push_back(math::full({N, 1, 1}, 1.0f, cfg_.dtype));
        all_rows.push_back(math::concatenate(last, 2));
    }

    return math::concatenate(all_rows, 1);
}

// =============================================================================
// Diagnostics
// =============================================================================

std::vector<double> SOdAnisotropicHeterogeneousGaussianSampler::angle_acceptance_rates() const {
    std::vector<double> rates(N_, -1.0);
    for (int i = 0; i < N_; ++i)
        if (angle_samplers_[i])
            rates[i] = angle_samplers_[i]->acceptance_rate();
    return rates;
}

} // namespace anisotropic
} // namespace sampler
