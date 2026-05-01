#include "sampler/isotropic/stiefel_heterogeneous_gaussian_sampler.hpp"

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

StiefelHeterogeneousGaussianSampler::StiefelHeterogeneousGaussianSampler(Config cfg)
    : cfg_(cfg), n_(cfg.n), k_(cfg.k), N_(cfg.N)
{
    assert(n_ >= 1 && k_ >= 1 && k_ < n_);
    assert(N_ >= 1);

    is_fat_ = (n_ < 2 * k_);
    k_eff_  = is_fat_ ? (n_ - k_) : k_;

    angle_samplers_.resize(N_);      // all null — built lazily in update_alphas
    alphas_.assign(N_, -1.0);        // sentinel: no alpha set yet
}

// =============================================================================
// update_alphas() — rebuild / warm-start chains where needed
// =============================================================================

void StiefelHeterogeneousGaussianSampler::update_alphas(
    const std::vector<double>& alphas,
    int burn_in_steps)
{
    assert(static_cast<int>(alphas.size()) == N_);

    std::vector<int> init_idx;
    std::vector<int> burnin_idx;

    for (int i = 0; i < N_; ++i) {
        const double a    = alphas[i];
        const double prev = alphas_[i];
        alphas_[i] = a;

        if (a >= kHighAlphaThreshold) continue;  // tangent-space regime

        const bool is_new      = !angle_samplers_[i];
        const bool was_tangent = (prev < 0.0) || (prev >= kHighAlphaThreshold);
        const bool alpha_moved = was_tangent ||
                                 std::abs(a - prev) > cfg_.alpha_rtol * std::abs(prev);

        if (is_new) {
            init_idx.push_back(i);
        } else if (alpha_moved) {
            burnin_idx.push_back(i);
        }
    }

    // --- Parallel first-time construction ---
    const int n_init = static_cast<int>(init_idx.size());
    #pragma omp parallel for schedule(static) num_threads(cfg_.num_threads)
    for (int ii = 0; ii < n_init; ++ii) {
        const int i = init_idx[ii];

        PrincipalAngleSampler::Config acfg;
        acfg.n              = n_;
        acfg.k              = k_eff_;
        acfg.alpha          = alphas_[i];
        acfg.num_chains     = 1;
        acfg.num_threads    = 1;
        acfg.burn_in        = burn_in_steps;
        acfg.leapfrog_steps = cfg_.leapfrog_steps;
        acfg.seed           = static_cast<uint64_t>(i) * 6364136223846793005ULL + 1;

        angle_samplers_[i] = std::make_unique<PrincipalAngleSampler>(acfg);
    }

    // --- Parallel in-place warm-start for changed alphas ---
    const int n_burnin = static_cast<int>(burnin_idx.size());
    #pragma omp parallel for schedule(static) num_threads(cfg_.num_threads)
    for (int ii = 0; ii < n_burnin; ++ii) {
        const int i = burnin_idx[ii];
        angle_samplers_[i]->set_alpha(alphas_[i], burn_in_steps);
    }
}

// =============================================================================
// set_x_hat() — store consensus; recompute dual frame for fat frames
// =============================================================================

void StiefelHeterogeneousGaussianSampler::set_x_hat(const Tensor& x_hat) {
    x_hat_ = x_hat;
    if (is_fat_) update_x_hat_prime();
}

void StiefelHeterogeneousGaussianSampler::update_x_hat_prime() {
    Tensor Z    = math::random_normal({n_, k_eff_}, cfg_.dtype);
    Tensor ZtX  = math::matmul(math::transpose(x_hat_, {1, 0}), Z);
    Tensor proj = math::matmul(x_hat_, ZtX);
    x_hat_prime_ = thin_qr(math::subtract(Z, proj));
}

// =============================================================================
// sample() — draw N samples from current chain states and consensus
// =============================================================================

Tensor StiefelHeterogeneousGaussianSampler::sample() {
    std::vector<int> hmc_idx, tan_idx;
    hmc_idx.reserve(N_);
    tan_idx.reserve(N_);
    for (int i = 0; i < N_; ++i) {
        if (alphas_[i] >= kHighAlphaThreshold)
            tan_idx.push_back(i);
        else
            hmc_idx.push_back(i);
    }

    std::vector<double> flat_theta(static_cast<std::size_t>(N_ * k_eff_), 0.0);

    if (!hmc_idx.empty())
        sample_hmc_parallel(hmc_idx, flat_theta);

    if (!tan_idx.empty())
        sample_tangent_batch(tan_idx, flat_theta);

    // Phase II: batched spectral lift — identical to StiefelGaussianSampler but
    // with per-particle alpha in draw_omega_int.
    Tensor sample_frame;

    if (!is_fat_) {
        // Tall-and-skinny (n >= 2k): O(k³) spectral lift
        auto [Theta11, Theta21, Theta12, Theta22] = compute_shape_blocks(flat_theta, k_);

        Tensor Z       = math::random_normal({N_, n_, k_}, cfg_.dtype);
        Tensor x_hat_b = math::expand_dims(x_hat_, {0});
        Tensor x_hat_T = math::transpose(x_hat_b, {0, 2, 1});
        Tensor ZtX     = math::matmul(x_hat_T, Z);
        Tensor proj    = math::matmul(x_hat_b, ZtX);
        Tensor U       = thin_qr(math::subtract(Z, proj));

        sample_frame = math::add(math::matmul(x_hat_b, Theta11),
                                 math::matmul(U, Theta21));
    } else {
        // Fat frame (n < 2k): geometric duality
        auto [Theta11, Theta21, Theta12, Theta22] = compute_shape_blocks(flat_theta, k_eff_);

        Tensor Z    = math::random_normal({N_, n_, k_eff_}, cfg_.dtype);
        Tensor xp_b = math::expand_dims(x_hat_prime_, {0});
        Tensor xp_T = math::transpose(xp_b, {0, 2, 1});
        Tensor ZtXp = math::matmul(xp_T, Z);
        Tensor proj = math::matmul(xp_b, ZtXp);
        Tensor U    = thin_qr(math::subtract(Z, proj));

        Tensor x_hat_b = math::expand_dims(x_hat_, {0});
        Tensor UtX     = math::matmul(math::transpose(U, {0, 2, 1}), x_hat_b);

        Tensor rot_Xp = math::matmul(xp_b, Theta12);
        Tensor rot_U  = math::matmul(U, Theta22);
        Tensor diff   = math::subtract(math::add(rot_Xp, rot_U), U);

        sample_frame = math::add(x_hat_b, math::matmul(diff, UtX));
    }

    math::eval(sample_frame);
    return sample_frame;
}

// =============================================================================
// Phase I (a): parallel HMC — advance each particle's chain by one sample
// =============================================================================

void StiefelHeterogeneousGaussianSampler::sample_hmc_parallel(
    const std::vector<int>& hmc_indices,
    std::vector<double>&    flat_theta)
{
    const int n_hmc = static_cast<int>(hmc_indices.size());

    #pragma omp parallel for schedule(dynamic) num_threads(cfg_.num_threads)
    for (int ii = 0; ii < n_hmc; ++ii) {
        const int i = hmc_indices[ii];

        std::vector<double> theta_i = angle_samplers_[i]->sample_angles(1);
        for (int j = 0; j < k_eff_; ++j)
            flat_theta[static_cast<std::size_t>(i * k_eff_ + j)] = theta_i[j];
    }
}

// =============================================================================
// Phase I (b): batched tangent-space sampler for high-alpha particles
// =============================================================================

void StiefelHeterogeneousGaussianSampler::sample_tangent_batch(
    const std::vector<int>& tan_indices,
    std::vector<double>&    flat_theta)
{
    const int N_tan = static_cast<int>(tan_indices.size());
    const int d_max = n_ - k_eff_;
    const int d_min = k_eff_;

    Tensor G       = math::random_normal({N_tan, d_max, d_min}, cfg_.dtype);
    Tensor GT      = math::transpose(G, {0, 2, 1});
    Tensor M       = math::matmul(GT, G);
    Tensor eigvals = math::eigvalsh(M);   // [N_tan, d_min] ascending
    math::eval(eigvals);

    std::vector<float> ev_cpu = math::to_float_vector(eigvals);

    for (int ii = 0; ii < N_tan; ++ii) {
        const int    i     = tan_indices[ii];
        const double scale = 1.0 / (2.0 * alphas_[i]);
        for (int j = 0; j < d_min; ++j) {
            const float ev = ev_cpu[static_cast<std::size_t>(ii * d_min + j)];
            flat_theta[static_cast<std::size_t>(i * k_eff_ + j)] =
                std::sqrt(ev > 0.0f ? static_cast<double>(ev) * scale : 0.0);
        }
    }
}

// =============================================================================
// Phase II: compute_shape_blocks — batched O(k³) spectral lift
// =============================================================================

StiefelHeterogeneousGaussianSampler::ShapeBlocks
StiefelHeterogeneousGaussianSampler::compute_shape_blocks(
    const std::vector<double>& flat_theta, int k)
{
    Tensor theta_vec = vec_to_tensor(flat_theta);
    Tensor theta     = math::reshape(theta_vec, {N_, k});

    Tensor omega = draw_omega_int(k);
    Tensor vr    = draw_v_right(k);

    Tensor theta_col = math::expand_dims(theta, {2});
    Tensor VrT       = math::transpose(vr, {0, 2, 1});
    Tensor B         = math::multiply(theta_col, VrT);

    Tensor neg_BT  = math::multiply(math::transpose(B, {0, 2, 1}), Tensor(-1.0f, cfg_.dtype));
    Tensor zeros_k = math::full({N_, k, k}, 0.0f, cfg_.dtype);

    Tensor top   = math::concatenate({omega, neg_BT}, 2);
    Tensor bot   = math::concatenate({B,     zeros_k}, 2);
    Tensor V_act = math::concatenate({top, bot}, 1);

    Tensor Theta_act = math::matrix_exp(V_act);

    Tensor Theta11 = math::slice(math::slice(Theta_act, 0, k,     1), 0, k,     2);
    Tensor Theta21 = math::slice(math::slice(Theta_act, k, 2 * k, 1), 0, k,     2);
    Tensor Theta12 = math::slice(math::slice(Theta_act, 0, k,     1), k, 2 * k, 2);
    Tensor Theta22 = math::slice(math::slice(Theta_act, k, 2 * k, 1), k, 2 * k, 2);

    return {Theta11, Theta21, Theta12, Theta22};
}

// =============================================================================
// draw_omega_int — per-particle scaled skew-symmetric [N, k, k]
// =============================================================================

Tensor StiefelHeterogeneousGaussianSampler::draw_omega_int(int k) {
    // sigma_i = 1/sqrt(2 * alpha_i);  the factor of 1/sqrt(2) comes from the
    // skew-symmetrisation: skew(G) = G - G^T, whose entries have variance 2.
    std::vector<float> scales(N_);
    for (int i = 0; i < N_; ++i) {
        const double sigma = 1.0 / std::sqrt(2.0 * alphas_[i]);
        scales[i] = static_cast<float>(sigma / std::sqrt(2.0));
    }
    Tensor scale_t = math::array(scales, {N_, 1, 1}, cfg_.dtype);

    Tensor G    = math::random_normal({N_, k, k}, cfg_.dtype);
    Tensor GT   = math::transpose(G, {0, 2, 1});
    Tensor skew = math::subtract(G, GT);
    return math::multiply(skew, scale_t);
}

// =============================================================================
// draw_v_right — Haar-uniform O(k) matrices [N, k, k]
// =============================================================================

Tensor StiefelHeterogeneousGaussianSampler::draw_v_right(int k) {
    Tensor G = math::random_normal({N_, k, k}, cfg_.dtype);
    auto [Q, R] = math::qr(G);

    Tensor I        = math::expand_dims(math::eye(k, cfg_.dtype), {0});  // [1, k, k]
    Tensor R_masked = math::multiply(R, I);
    Tensor R_diag   = math::sum(R_masked, {2});                           // [N, k]
    Tensor sgn      = math::sign(R_diag);

    return math::multiply(Q, math::expand_dims(sgn, {1}));
}

// =============================================================================
// thin_qr — first n_cols columns of the full Q factor
// =============================================================================

Tensor StiefelHeterogeneousGaussianSampler::thin_qr(const Tensor& a) {
    const int n_cols  = a.shape().back();
    const int col_axis = static_cast<int>(a.shape().size()) - 1;
    auto [Q_full, R] = math::qr(a);
    return math::slice(Q_full, 0, n_cols, col_axis);
}

// =============================================================================
// vec_to_tensor
// =============================================================================

Tensor StiefelHeterogeneousGaussianSampler::vec_to_tensor(const std::vector<double>& v) {
    std::vector<float> f32(v.begin(), v.end());
    return math::array(f32, {static_cast<int>(f32.size())}, cfg_.dtype);
}

} // namespace sampler
