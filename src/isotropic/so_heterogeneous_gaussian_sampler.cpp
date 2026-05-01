#include "sampler/isotropic/so_heterogeneous_gaussian_sampler.hpp"

#include <omp.h>
#include <cassert>
#include <cmath>
#include <vector>

namespace sampler {

namespace math = isomorphism::math;
using Tensor   = isomorphism::Tensor;
using DType    = isomorphism::DType;

// =============================================================================
// Construction
// =============================================================================

SOdHeterogeneousGaussianSampler::SOdHeterogeneousGaussianSampler(Config cfg)
    : cfg_(cfg), d_(cfg.d), m_(cfg.d / 2), N_(cfg.N)
{
    assert(d_ >= 2);
    assert(N_ >= 1);
    alphas_.assign(N_, -1.0);  // sentinel: no alpha set yet
}

// =============================================================================
// update_alphas() — re-cluster particles by alpha, reuse or rebuild chains
// =============================================================================

void SOdHeterogeneousGaussianSampler::update_alphas(
    const std::vector<double>& alphas,
    int burn_in_steps)
{
    assert(static_cast<int>(alphas.size()) == N_);
    alphas_ = alphas;

    // --- Step 1: Build new grouping (greedy clustering within alpha_rtol) ---
    // HMC-regime particles (alpha < threshold) are clustered; tangent-space
    // particles (alpha >= threshold) are excluded and handled separately in sample().

    std::vector<AlphaGroup> new_groups;

    for (int i = 0; i < N_; ++i) {
        const double a = alphas[i];
        if (a >= kHighAlphaThreshold) continue;

        int gid = -1;
        for (int g = 0; g < static_cast<int>(new_groups.size()); ++g) {
            const double rep = new_groups[g].alpha;
            if (std::abs(a - rep) <= cfg_.alpha_rtol * std::abs(rep)) {
                gid = g;
                break;
            }
        }
        if (gid < 0) {
            gid = static_cast<int>(new_groups.size());
            new_groups.push_back({a, {}, nullptr});
        }
        new_groups[gid].particle_idx.push_back(i);
    }

    // --- Step 2: Transfer chains from matching old groups ---
    // Each old chain is consumed at most once; unmatched old chains are destroyed.

    for (auto& ng : new_groups) {
        for (auto& og : groups_) {
            if (!og.chain) continue;  // already consumed
            const double delta  = std::abs(ng.alpha - og.alpha);
            const double thresh = cfg_.alpha_rtol * std::abs(og.alpha);
            if (delta <= thresh) {
                ng.chain = std::move(og.chain);
                break;
            }
        }
    }

    // --- Step 3: Parallel construction of new (unmatched) chains ---

    std::vector<int> init_gids;
    init_gids.reserve(new_groups.size());
    for (int g = 0; g < static_cast<int>(new_groups.size()); ++g) {
        if (!new_groups[g].chain) init_gids.push_back(g);
    }

    const int n_new = static_cast<int>(init_gids.size());
    #pragma omp parallel for schedule(static) num_threads(cfg_.num_threads)
    for (int ii = 0; ii < n_new; ++ii) {
        const int g  = init_gids[ii];
        auto&     ng = new_groups[g];

        SOdAngleSampler::Config acfg;
        acfg.d              = d_;
        acfg.alpha          = ng.alpha;
        acfg.num_chains     = std::min(static_cast<int>(ng.particle_idx.size()), cfg_.num_chains);
        acfg.num_threads    = 1;   // single-threaded per chain; outer loop provides parallelism
        acfg.burn_in        = burn_in_steps;
        acfg.leapfrog_steps = cfg_.leapfrog_steps;
        acfg.seed           = static_cast<uint64_t>(g) * 6364136223846793005ULL + 1;

        ng.chain = std::make_unique<SOdAngleSampler>(acfg);
    }

    groups_ = std::move(new_groups);
}

// =============================================================================
// set_m_hat() — cheap: just store the tensor
// =============================================================================

void SOdHeterogeneousGaussianSampler::set_m_hat(const Tensor& m_hat) {
    m_hat_ = m_hat;
}

// =============================================================================
// sample() — draw one sample per particle from current chain states
// =============================================================================

Tensor SOdHeterogeneousGaussianSampler::sample() {
    std::vector<int> tan_idx;
    tan_idx.reserve(N_);
    for (int i = 0; i < N_; ++i) {
        if (alphas_[i] >= kHighAlphaThreshold)
            tan_idx.push_back(i);
    }

    std::vector<double> flat_theta(static_cast<std::size_t>(N_ * m_), 0.0);

    if (!groups_.empty())
        sample_hmc_groups(flat_theta);

    if (!tan_idx.empty())
        sample_tangent_batch(tan_idx, flat_theta);

    // Phase II: batched GPU lift — identical for all particles.
    Tensor Theta   = build_canonical_rotation(flat_theta, N_); // [N, d, d]
    Tensor Q       = draw_haar_od(N_);                          // [N, d, d]
    Tensor QT      = math::transpose(Q, {0, 2, 1});
    Tensor g_tilde = math::matmul(math::matmul(Q, Theta), QT); // [N, d, d]
    Tensor m_hat_b = math::expand_dims(m_hat_, {0});            // [1, d, d]
    Tensor X       = math::matmul(g_tilde, m_hat_b);            // [N, d, d]

    math::eval(X);
    return X;
}

// =============================================================================
// Phase I (a): draw angles for all HMC groups in parallel
// =============================================================================

void SOdHeterogeneousGaussianSampler::sample_hmc_groups(std::vector<double>& flat_theta)
{
    const int n_groups = static_cast<int>(groups_.size());

    // Each group's chain draws M = group_size angle sets via sample_angles(M).
    // Groups are processed in parallel; each chain is single-threaded internally.
    #pragma omp parallel for schedule(dynamic) num_threads(cfg_.num_threads)
    for (int g = 0; g < n_groups; ++g) {
        auto&            group  = groups_[g];
        const int        M      = static_cast<int>(group.particle_idx.size());
        std::vector<double> thetas = group.chain->sample_angles(M); // M * m_ values

        for (int j = 0; j < M; ++j) {
            const int i = group.particle_idx[j];
            for (int l = 0; l < m_; ++l) {
                flat_theta[static_cast<std::size_t>(i * m_ + l)] =
                    thetas[static_cast<std::size_t>(j * m_ + l)];
            }
        }
    }
}

// =============================================================================
// Phase I (b): batched tangent-space sampler for high-alpha particles
// =============================================================================

void SOdHeterogeneousGaussianSampler::sample_tangent_batch(
    const std::vector<int>& tan_indices,
    std::vector<double>&    flat_theta)
{
    const int N_tan = static_cast<int>(tan_indices.size());

    std::vector<float> scales(N_tan);
    for (int ii = 0; ii < N_tan; ++ii) {
        const double a = alphas_[tan_indices[ii]];
        scales[ii] = static_cast<float>(1.0 / std::sqrt(2.0 * a));
    }
    Tensor scale_t = math::array(scales, {N_tan, 1, 1}, cfg_.dtype);

    Tensor A     = math::random_normal({N_tan, d_, d_}, cfg_.dtype);
    Tensor Omega = math::multiply(
        math::subtract(A, math::transpose(A, {0, 2, 1})),
        scale_t);

    Tensor B       = math::matmul(Omega, math::transpose(Omega, {0, 2, 1}));
    Tensor eigvals = math::eigvalsh(B);   // [N_tan, d] ascending
    math::eval(eigvals);

    std::vector<float> ev_cpu = math::to_float_vector(eigvals);

    for (int ii = 0; ii < N_tan; ++ii) {
        const int i = tan_indices[ii];
        for (int j = 0; j < m_; ++j) {
            const int   idx = d_ - 1 - 2 * j;
            const float ev  = ev_cpu[static_cast<std::size_t>(ii * d_ + idx)];
            flat_theta[static_cast<std::size_t>(i * m_ + j)] =
                std::sqrt(ev > 0.0f ? static_cast<double>(ev) : 0.0);
        }
    }
}

// =============================================================================
// Phase II: build_canonical_rotation  [N, d, d]
// =============================================================================

Tensor SOdHeterogeneousGaussianSampler::build_canonical_rotation(
    const std::vector<double>& flat_theta,
    int N)
{
    const bool odd = (d_ % 2 == 1);
    const int  d   = d_;
    const int  m   = m_;

    std::vector<float> cos_vals(static_cast<std::size_t>(N * m));
    std::vector<float> sin_vals(static_cast<std::size_t>(N * m));
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
// Phase II: draw_haar_od  [N, d, d]
// =============================================================================

Tensor SOdHeterogeneousGaussianSampler::draw_haar_od(int N) {
    Tensor G = math::random_normal({N, d_, d_}, cfg_.dtype);

    auto [Q, R] = math::qr(G);

    Tensor I_exp   = math::expand_dims(math::eye(d_, cfg_.dtype), {0});
    Tensor R_diag  = math::sum(math::multiply(R, I_exp), {2});
    Tensor sgn     = math::sign(R_diag);

    float  parity   = (d_ % 2 == 0) ? -1.0f : 1.0f;
    Tensor prod_sgn = math::prod(sgn, {1});
    Tensor det_Q    = math::multiply(prod_sgn, Tensor(parity));

    Tensor Q_od = math::multiply(Q, math::expand_dims(sgn, {1}));

    Tensor ones           = math::full({N, d_ - 1}, 1.0f, cfg_.dtype);
    Tensor col_correction = math::concatenate(
        {ones, math::expand_dims(det_Q, {1})}, 1);

    return math::multiply(Q_od, math::expand_dims(col_correction, {1}));
}

} // namespace sampler
