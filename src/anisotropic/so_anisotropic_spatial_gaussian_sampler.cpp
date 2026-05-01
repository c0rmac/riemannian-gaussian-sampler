#include "sampler/anisotropic/so_anisotropic_spatial_gaussian_sampler.hpp"

#include <cassert>
#include <cmath>
#include <omp.h>
#include <vector>

namespace sampler {
namespace anisotropic {

namespace math = isomorphism::math;
using Tensor   = isomorphism::Tensor;
using DType    = isomorphism::DType;

// Extract the d diagonal entries from a row-major d×d spatial matrix.
static std::vector<double> extract_diagonal(const std::vector<double>& gamma_dxd, int d) {
    std::vector<double> diag(d);
    for (int i = 0; i < d; i++)
        diag[i] = gamma_dxd[static_cast<size_t>(i) * d + i];
    return diag;
}

// =============================================================================
// Construction
// =============================================================================

SOdAnisotropicSpatialGaussianSampler::SOdAnisotropicSpatialGaussianSampler(
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
    assert(d >= 2);
    assert(static_cast<int>(gamma.size()) == d * d);
    cfg_.angle_cfg.d = d;
    rebuild_samplers();
}

void SOdAnisotropicSpatialGaussianSampler::rebuild_samplers()
{
    // Phase I: angle sampler uses the full d×d spatial gamma.
    angle_sampler_ = std::make_unique<SOdAnisotropicAngleSampler>(gamma_, cfg_.angle_cfg);
    gamma_min_     = angle_sampler_->gamma_min();

    // Phase II: spatial CPU sampler uses only the diagonal entries.
    std::vector<double> gamma_diag = extract_diagonal(gamma_, d_);
    cpu_sampler_ = std::make_unique<SOdHypersphereSamplerCPUSpatial>(
                       d_, gamma_diag, gamma_min_, cfg_.cpu_cfg);
}

// =============================================================================
// Config updates
// =============================================================================

void SOdAnisotropicSpatialGaussianSampler::set_m_hat(Tensor m_hat) {
    m_hat_ = std::move(m_hat);
}

void SOdAnisotropicSpatialGaussianSampler::update_gamma(const std::vector<double>& gamma) {
    assert(static_cast<int>(gamma.size()) == d_ * d_);
    gamma_ = gamma;
    angle_sampler_->update_gamma(gamma_);
    gamma_min_ = angle_sampler_->gamma_min();
    std::vector<double> gamma_diag = extract_diagonal(gamma_, d_);
    cpu_sampler_->update_gamma(gamma_diag, gamma_min_);
}

double SOdAnisotropicSpatialGaussianSampler::angle_acceptance_rate() const {
    return angle_sampler_ ? angle_sampler_->acceptance_rate() : -1.0;
}

// =============================================================================
// sample()
// =============================================================================

Tensor SOdAnisotropicSpatialGaussianSampler::sample() {
    const int N = std::max(1, cfg_.num_samples);

    // Phase I: draw N angle vectors via HMC.
    std::vector<double> flat_theta = angle_sampler_->sample_angles(N);

    // Phase II: build orientation frames Q ∈ SO(d) via spatial rejection.
    std::vector<double> flat_q = cpu_sampler_->build_orientation_frames(flat_theta, N);
    std::vector<float>  flat_q_f32(flat_q.begin(), flat_q.end());
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

Tensor SOdAnisotropicSpatialGaussianSampler::build_canonical_rotation(
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
