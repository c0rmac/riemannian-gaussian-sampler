#include "sampler/anisotropic/so_anisotropic_gaussian_sampler.hpp"

#include <cassert>
#include <cmath>
#include <omp.h>
#include <vector>

namespace sampler {
namespace anisotropic {

namespace math = isomorphism::math;
using Tensor   = isomorphism::Tensor;
using DType    = isomorphism::DType;

// Derive a d×d spatial diagonal surrogate from a D×D Lie algebra precision.
// Mirrors lie_to_spatial_diagonal in the heterogeneous sampler.
static std::vector<double> lie_to_spatial_diagonal(
    const std::vector<double>& gamma_DxD, int d)
{
    const int D = d * (d - 1) / 2;
    std::vector<double> spatial(static_cast<size_t>(d) * d, 0.0);

    if (d == 2) {
        const double val = gamma_DxD[0];
        spatial[0] = val;
        spatial[3] = val;
        return spatial;
    }

    std::vector<double> gamma_plane(D);
    for (int a = 0; a < D; a++)
        gamma_plane[a] = gamma_DxD[static_cast<size_t>(a) * D + a];

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
        spatial[static_cast<size_t>(j) * d + j] = c_j;
    }
    return spatial;
}

// =============================================================================
// Construction
// =============================================================================

SOdAnisotropicGaussianSampler::SOdAnisotropicGaussianSampler(
    Tensor                     m_hat,
    int                        d,
    const std::vector<double>& gamma,
    Config                     cfg)
    : m_hat_(std::move(m_hat))
    , d_(d)
    , m_(d / 2)
    , gamma_(gamma)
    , gamma_min_(0.0)
    , cfg_(cfg)
{
    const int D = d * (d - 1) / 2;
    assert(d >= 2);
    assert(static_cast<int>(gamma.size()) == D * D);
    cfg_.angle_cfg.d = d;
    rebuild_samplers();
}

void SOdAnisotropicGaussianSampler::rebuild_samplers()
{
    // Phase I: angle sampler needs d×d spatial; derive it from the D×D Lie tensor.
    std::vector<double> gamma_spatial = lie_to_spatial_diagonal(gamma_, d_);
    angle_sampler_ = std::make_unique<SOdAnisotropicAngleSampler>(gamma_spatial, cfg_.angle_cfg);
    gamma_min_     = angle_sampler_->gamma_min();

    // Phase II: full D×D tensor passed directly; compute_surrogate() runs internally.
    cpu_sampler_ = std::make_unique<SOdHypersphereSamplerCPU>(
                       d_, gamma_, gamma_min_, cfg_.cpu_cfg);
}

// =============================================================================
// Config updates
// =============================================================================

void SOdAnisotropicGaussianSampler::set_m_hat(Tensor m_hat) {
    m_hat_ = std::move(m_hat);
}

void SOdAnisotropicGaussianSampler::update_gamma(const std::vector<double>& gamma) {
    const int D = d_ * (d_ - 1) / 2;
    assert(static_cast<int>(gamma.size()) == D * D);
    gamma_ = gamma;
    std::vector<double> gamma_spatial = lie_to_spatial_diagonal(gamma_, d_);
    angle_sampler_->update_gamma(gamma_spatial);
    gamma_min_ = angle_sampler_->gamma_min();
    cpu_sampler_->update_gamma(gamma_, gamma_min_);
}

double SOdAnisotropicGaussianSampler::angle_acceptance_rate() const {
    return angle_sampler_ ? angle_sampler_->acceptance_rate() : -1.0;
}

// =============================================================================
// sample()
// =============================================================================

Tensor SOdAnisotropicGaussianSampler::sample() {
    const int N = std::max(1, cfg_.num_samples);

    // Phase I: draw N angle vectors via HMC.
    std::vector<double> flat_theta = angle_sampler_->sample_angles(N);

    // Phase II: build orientation frames Q ∈ SO(d) via exact Lie rejection.
    std::vector<double> flat_q =
        cpu_sampler_->build_orientation_frames(flat_theta, N);
    std::vector<float> flat_q_f32(flat_q.begin(), flat_q.end());
    Tensor Q = math::array(flat_q_f32, {N, d_, d_}, cfg_.dtype);  // [N, d, d]

    // Phase III: geometric recombination.
    Tensor Theta   = build_canonical_rotation(flat_theta, N);
    Tensor QT      = math::transpose(Q, {0, 2, 1});
    Tensor g_tilde = math::matmul(math::matmul(Q, Theta), QT);
    Tensor X       = math::matmul(g_tilde, m_hat_);
    math::eval(X);

    if (N == 1) return math::slice(X, 0, 1, 0);  // [d, d]
    return X;
}

// =============================================================================
// build_canonical_rotation
// =============================================================================

Tensor SOdAnisotropicGaussianSampler::build_canonical_rotation(
    const std::vector<double>& flat_theta, int N) const
{
    const bool odd = (d_ % 2 == 1);
    const int  d   = d_;
    const int  m   = m_;

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

        std::vector<Tensor> row0_parts, row1_parts;

        if (2 * j > 0) {
            Tensor lz = math::full({N, 1, 2 * j}, 0.0f, cfg_.dtype);
            row0_parts.push_back(lz);
            row1_parts.push_back(lz);
        }

        row0_parts.push_back(c_e);
        row0_parts.push_back(ns_e);
        row1_parts.push_back(s_e);
        row1_parts.push_back(c_e);

        const int right = 2 * (m - j - 1);
        if (right > 0) {
            Tensor rz = math::full({N, 1, right}, 0.0f, cfg_.dtype);
            row0_parts.push_back(rz);
            row1_parts.push_back(rz);
        }
        if (odd) {
            Tensor oz = math::full({N, 1, 1}, 0.0f, cfg_.dtype);
            row0_parts.push_back(oz);
            row1_parts.push_back(oz);
        }

        all_rows.push_back(math::concatenate(row0_parts, 2));
        all_rows.push_back(math::concatenate(row1_parts, 2));
    }

    if (odd) {
        std::vector<Tensor> last_parts;
        if (d - 1 > 0)
            last_parts.push_back(math::full({N, 1, d - 1}, 0.0f, cfg_.dtype));
        last_parts.push_back(math::full({N, 1, 1}, 1.0f, cfg_.dtype));
        all_rows.push_back(math::concatenate(last_parts, 2));
    }

    return math::concatenate(all_rows, 1);
}

} // namespace anisotropic
} // namespace sampler
