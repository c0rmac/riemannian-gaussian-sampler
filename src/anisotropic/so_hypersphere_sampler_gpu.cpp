#include "sampler/anisotropic/so_hypersphere_sampler_gpu.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <omp.h>
#include <random>
#include <vector>

namespace sampler {
namespace anisotropic {

namespace math = isomorphism::math;
using Tensor   = isomorphism::Tensor;
using DType    = isomorphism::DType;

// =============================================================================
// Construction
// =============================================================================

SOdHypersphereSamplerGPU::SOdHypersphereSamplerGPU(
    int d,
    const std::vector<double>& gamma_flat,
    double gamma_min,
    Config cfg)
    : d_(d), m_(d / 2), gamma_min_(gamma_min), cfg_(cfg)
{
    assert(d >= 2);
    assert(static_cast<int>(gamma_flat.size()) == d * d);
    gamma_t_ = upload(gamma_flat, {d, d});
}

void SOdHypersphereSamplerGPU::update_gamma(const std::vector<double>& gamma_flat,
                                             double gamma_min)
{
    assert(static_cast<int>(gamma_flat.size()) == d_ * d_);
    gamma_t_   = upload(gamma_flat, {d_, d_});
    gamma_min_ = gamma_min;
}

// =============================================================================
// Helpers
// =============================================================================

Tensor SOdHypersphereSamplerGPU::upload(const std::vector<double>& v,
                                         std::initializer_list<int> shape) const
{
    std::vector<float> f32(v.begin(), v.end());
    std::vector<int>   s(shape);
    return math::array(f32, s, cfg_.dtype);
}

std::vector<float> SOdHypersphereSamplerGPU::readback_1d(const Tensor& t, int N) const {
    math::eval(t);
    std::vector<float> out(N);
    for (int n = 0; n < N; n++)
        out[n] = static_cast<float>(math::to_double(math::slice(t, n, n + 1, 0)));
    return out;
}

std::vector<float> SOdHypersphereSamplerGPU::readback_2d(const Tensor& t, int N, int d) const {
    math::eval(t);
    std::vector<float> out(static_cast<size_t>(N) * d);
    for (int n = 0; n < N; n++) {
        Tensor row = math::slice(t, n, n + 1, 0);   // [1, d]
        for (int i = 0; i < d; i++)
            out[static_cast<size_t>(n) * d + i] =
                static_cast<float>(math::to_double(math::slice(row, i, i + 1, 1)));
    }
    return out;
}

// =============================================================================
// Build orientation frames — GPU batched per-column loop
// =============================================================================

Tensor SOdHypersphereSamplerGPU::build_orientation_frames(
    const std::vector<double>& flat_theta, int N)
{
    const int d = d_;
    const int m = m_;

    // CPU acceptance state.
    std::vector<bool>  accepted(N, false);
    // accepted_cols[col] = flat float32 [N * d] of accepted column vectors.
    std::vector<std::vector<float>> accepted_cols(d - 1, std::vector<float>(N * d, 0.0f));

    // Xoshiro RNG for CPU-side acceptance decisions.
    Xoshiro256PlusPlus rng(cfg_.seed);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    // Gamma broadcast for batched matmul: [1, d, d].
    Tensor Gamma_b = math::expand_dims(gamma_t_, {0});

    // Build the accepted columns vector as a [N, d, col] tensor iteratively.
    // We maintain Q_built_f32 as a CPU-side flat array [N * d * col]
    // in the layout needed for math::array({N, d, col}).
    std::vector<float> Q_built_f32;  // [N * d * col], grows per iteration

    for (int col = 0; col < d - 1; col++) {
        const int    pair_k       = col / 2;
        const double theta_k      = flat_theta[pair_k];   // representative (same for all samples)
        // Per-sample weight scales: theta[n*m + pair_k]^2 / 2.
        std::vector<float> ws_cpu(N);
        for (int n = 0; n < N; n++) {
            double tk = flat_theta[static_cast<size_t>(n) * m + pair_k];
            ws_cpu[n] = static_cast<float>(0.5 * tk * tk);
        }
        (void)theta_k;

        Tensor ws_t = math::array(ws_cpu, {N}, cfg_.dtype);   // [N]

        // Reset acceptance for this column.
        std::fill(accepted.begin(), accepted.end(), false);
        std::vector<float> best_log_w(N, -std::numeric_limits<float>::infinity());
        std::vector<float> best_proposal(N * d, 0.0f);

        // Build Q_built tensor [N, d, col] from Q_built_f32 (for col > 0).
        Tensor Q_built_t;
        if (col > 0) {
            Q_built_t = math::array(Q_built_f32, {N, d, col}, cfg_.dtype);
        }

        for (int attempt = 0; attempt < cfg_.max_retry_loops; attempt++) {
            int pending = 0;
            for (int n = 0; n < N; n++) if (!accepted[n]) pending++;
            if (pending == 0) break;

            // 1. Draw Gaussian proposals [N, d].
            Tensor proposals = math::random_normal({N, d}, cfg_.dtype);   // [N, d]

            // 2. Gram-Schmidt: project out the already-accepted columns.
            if (col > 0) {
                // Q_built_t: [N, d, col].  proposals: [N, d].
                // dots = Q_built^T @ proposals_3d → [N, col, 1]
                Tensor p3d  = math::expand_dims(proposals, {2});                       // [N, d, 1]
                Tensor dots = math::matmul(math::transpose(Q_built_t, {0, 2, 1}), p3d);  // [N, col, 1]
                // correction = Q_built @ dots → [N, d, 1]
                Tensor corr = math::matmul(Q_built_t, dots);                           // [N, d, 1]
                // Squeeze last dim: sum over axis 2 (size 1).
                Tensor corr_2d = math::sum(corr, {2});                                 // [N, d]
                proposals = math::subtract(proposals, corr_2d);
            }

            // 3. Normalise.
            // norm2 [N], then divide each row.
            Tensor norm2   = math::sum(math::multiply(proposals, proposals), {1});     // [N]
            math::eval(norm2);
            std::vector<float> norm2_cpu = readback_1d(norm2, N);

            math::eval(proposals);
            std::vector<float> prop_cpu = readback_2d(proposals, N, d);

            // Normalise on CPU (avoids division-by-tensor complexity).
            for (int n = 0; n < N; n++) {
                float nrm = (norm2_cpu[n] > 1e-28f) ? std::sqrt(norm2_cpu[n]) : 1.0f;
                float inv = 1.0f / nrm;
                for (int i = 0; i < d; i++)
                    prop_cpu[static_cast<size_t>(n) * d + i] *= inv;
            }

            // 4. Upload normalised proposals and compute q^T Γ q.
            Tensor unit_t = math::array(prop_cpu, {N, d}, cfg_.dtype);                // [N, d]

            // Gq = Γ @ q^T for each sample: (1,d,d) @ (N,d,1) → (N,d,1)
            Tensor q3d    = math::expand_dims(unit_t, {2});                            // [N, d, 1]
            Tensor Gq     = math::matmul(Gamma_b, q3d);                               // [N, d, 1]
            // q^T Γ q = sum(q * Γq, axis 1) [N, 1] then squeeze.
            Tensor qGq_3d = math::multiply(q3d, Gq);                                  // [N, d, 1]
            Tensor qGq_t  = math::sum(qGq_3d, {1});                                   // [N, 1]
            Tensor qGq    = math::sum(qGq_t,   {1});                                  // [N]

            math::eval(qGq);
            std::vector<float> qGq_cpu = readback_1d(qGq, N);

            // 5. CPU acceptance: log(u) ≤ −ws * (qGq − γ_min).
            for (int n = 0; n < N; n++) {
                if (accepted[n]) continue;

                const float log_w = -ws_cpu[n] * (qGq_cpu[n] - static_cast<float>(gamma_min_));

                if (log_w > best_log_w[n]) {
                    best_log_w[n] = log_w;
                    for (int i = 0; i < d; i++)
                        best_proposal[static_cast<size_t>(n) * d + i] =
                            prop_cpu[static_cast<size_t>(n) * d + i];
                }

                const float log_u = std::log(static_cast<float>(unif01(rng)));
                if (log_u <= log_w) {
                    accepted[n] = true;
                    for (int i = 0; i < d; i++)
                        accepted_cols[col][static_cast<size_t>(n) * d + i] =
                            prop_cpu[static_cast<size_t>(n) * d + i];
                }
            }
        }

        // Fallback for any sample still unaccepted.
        for (int n = 0; n < N; n++) {
            if (!accepted[n])
                for (int i = 0; i < d; i++)
                    accepted_cols[col][static_cast<size_t>(n) * d + i] =
                        best_proposal[static_cast<size_t>(n) * d + i];
        }

        // Rebuild Q_built_f32 [N, d, col+1] in [n, row, col] layout.
        Q_built_f32.assign(static_cast<size_t>(N) * d * (col + 1), 0.0f);
        for (int n = 0; n < N; n++)
            for (int r = 0; r < d; r++)
                for (int c = 0; c <= col; c++)
                    Q_built_f32[static_cast<size_t>(n) * d * (col + 1) + r * (col + 1) + c] =
                        accepted_cols[c][static_cast<size_t>(n) * d + r];
    }

    // -------------------------------------------------------------------------
    // Last column: Gram-Schmidt from best standard basis vector + det sign fix.
    // Each sample is independent — all locals below are thread-private.
    // -------------------------------------------------------------------------
    std::vector<float> last_col(N * d, 0.0f);
    #pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < N; n++) {
        // Build column-major Q_cm [d × (d-1)] for Gram-Schmidt.
        std::vector<double> Q_cm(static_cast<size_t>(d) * (d - 1));
        for (int c = 0; c < d - 1; c++)
            for (int r = 0; r < d; r++)
                Q_cm[static_cast<size_t>(r) + c * d] =
                    accepted_cols[c][static_cast<size_t>(n) * d + r];

        // Choose the most numerically independent basis vector.
        int    best_j    = 0;
        double best_proj = -1.0;
        for (int j = 0; j < d; j++) {
            double max_dot = 0.0;
            for (int c = 0; c < d - 1; c++)
                max_dot = std::max(max_dot, std::abs(Q_cm[static_cast<size_t>(j) + c * d]));
            if (1.0 - max_dot > best_proj) { best_proj = 1.0 - max_dot; best_j = j; }
        }

        std::vector<double> v(d, 0.0);
        v[best_j] = 1.0;

        for (int c = 0; c < d - 1; c++) {
            double dot = 0.0;
            for (int r = 0; r < d; r++) dot += v[r] * Q_cm[static_cast<size_t>(r) + c * d];
            for (int r = 0; r < d; r++) v[r] -= dot * Q_cm[static_cast<size_t>(r) + c * d];
        }
        double norm2 = 0.0;
        for (int r = 0; r < d; r++) norm2 += v[r] * v[r];

        // Fallback: try all basis vectors.
        if (norm2 < 1e-28) {
            for (int j = 0; j < d && norm2 < 1e-28; j++) {
                std::fill(v.begin(), v.end(), 0.0);
                v[j] = 1.0;
                for (int c = 0; c < d - 1; c++) {
                    double dot = 0.0;
                    for (int r = 0; r < d; r++) dot += v[r] * Q_cm[static_cast<size_t>(r) + c * d];
                    for (int r = 0; r < d; r++) v[r] -= dot * Q_cm[static_cast<size_t>(r) + c * d];
                }
                norm2 = 0.0;
                for (int r = 0; r < d; r++) norm2 += v[r] * v[r];
            }
        }
        const double inv_norm = 1.0 / std::sqrt(norm2);
        for (int r = 0; r < d; r++) v[r] *= inv_norm;

        // Compute det sign via Gaussian elimination on the full d×d matrix.
        // Assemble row-major A from column-major Q_cm + v.
        std::vector<double> A(static_cast<size_t>(d) * d);
        for (int r = 0; r < d; r++) {
            for (int c = 0; c < d - 1; c++)
                A[static_cast<size_t>(r) * d + c] = Q_cm[static_cast<size_t>(r) + c * d];
            A[static_cast<size_t>(r) * d + (d - 1)] = v[r];
        }

        int sgn = 1;
        for (int c = 0; c < d; c++) {
            int    piv   = c;
            double maxv  = std::abs(A[static_cast<size_t>(c) * d + c]);
            for (int r = c + 1; r < d; r++) {
                double vv = std::abs(A[static_cast<size_t>(r) * d + c]);
                if (vv > maxv) { maxv = vv; piv = r; }
            }
            if (maxv < 1e-14) { sgn = 0; break; }
            if (piv != c) {
                for (int k = 0; k < d; k++)
                    std::swap(A[static_cast<size_t>(c) * d + k],
                              A[static_cast<size_t>(piv) * d + k]);
                sgn = -sgn;
            }
            if (A[static_cast<size_t>(c) * d + c] < 0.0) sgn = -sgn;
            const double inv_p = 1.0 / A[static_cast<size_t>(c) * d + c];
            for (int r = c + 1; r < d; r++) {
                const double f = A[static_cast<size_t>(r) * d + c] * inv_p;
                for (int k = c; k < d; k++)
                    A[static_cast<size_t>(r) * d + k] -= f * A[static_cast<size_t>(c) * d + k];
            }
        }
        if (sgn < 0)
            for (int r = 0; r < d; r++) v[r] *= -1.0;

        for (int r = 0; r < d; r++)
            last_col[static_cast<size_t>(n) * d + r] = static_cast<float>(v[r]);
    }

    // -------------------------------------------------------------------------
    // Assemble full Q [N, d, d] flat float32, then upload to tensor.
    // Layout: Q[n, row, col] row-major.
    // -------------------------------------------------------------------------
    std::vector<float> flat_q(static_cast<size_t>(N) * d * d, 0.0f);
    // Read-only from accepted_cols / last_col; write to disjoint n-slices of flat_q.
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++)
        for (int r = 0; r < d; r++) {
            for (int c = 0; c < d - 1; c++)
                flat_q[static_cast<size_t>(n) * d * d + r * d + c] =
                    accepted_cols[c][static_cast<size_t>(n) * d + r];
            flat_q[static_cast<size_t>(n) * d * d + r * d + (d - 1)] =
                last_col[static_cast<size_t>(n) * d + r];
        }

    return math::array(flat_q, {N, d, d}, cfg_.dtype);
}

} // namespace anisotropic
} // namespace sampler
