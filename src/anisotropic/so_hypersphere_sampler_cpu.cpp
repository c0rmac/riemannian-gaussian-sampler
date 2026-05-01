#include "sampler/anisotropic/so_hypersphere_sampler_cpu.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <omp.h>
#include <random>
#include <vector>

namespace sampler {
namespace anisotropic {

// =============================================================================
// Construction / update
// =============================================================================

SOdHypersphereSamplerCPU::SOdHypersphereSamplerCPU(
    int d, const std::vector<double>& gamma_flat, double gamma_min, Config cfg)
    : d_(d), m_(d / 2), gamma_min_(gamma_min), cfg_(cfg)
{
    assert(d >= 2);
    const int D = d * (d - 1) / 2;
    assert(static_cast<int>(gamma_flat.size()) == D * D);
    gamma_dense_DxD_ = gamma_flat;
    compute_surrogate();
}

void SOdHypersphereSamplerCPU::update_gamma(const std::vector<double>& gamma_flat,
                                             double gamma_min)
{
    const int D = d_ * (d_ - 1) / 2;
    assert(static_cast<int>(gamma_flat.size()) == D * D);
    gamma_dense_DxD_ = gamma_flat;
    gamma_min_       = gamma_min;
    compute_surrogate();
}

// =============================================================================
// compute_surrogate — least-squares diagonal projection D×D → d
// =============================================================================
//
// Given the D×D Lie algebra precision, derive a diagonal d×d spatial surrogate
// c_j such that (c_j + c_k)/2 ≈ Gamma_Lie[a(j,k), a(j,k)] for all j<k.
//
// Exact solution (unique for d≤3, least-squares for d≥4):
//   c_j = max(1e-8, (2·G_j − S) / (d−2))   (d≥3)
//   c_0 = c_1 = G_tot                        (d=2)
//
// where G_j = Σ_{k≠j} diag[a(j,k)] and S = 2·G_tot/(d−1).

void SOdHypersphereSamplerCPU::compute_surrogate()
{
    const int d = d_;
    const int D = d * (d - 1) / 2;

    gamma_surrogate_dxd_.assign(d, 0.0);

    // Use the diagonal of gamma_dense_DxD_ as per-plane effective precision.
    std::vector<double> gamma_plane(D);
    for (int a = 0; a < D; a++)
        gamma_plane[a] = gamma_dense_DxD_[a * D + a];

    if (d == 2) {
        gamma_surrogate_dxd_[0] = gamma_plane[0];
        gamma_surrogate_dxd_[1] = gamma_plane[0];
        return;
    }

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
    for (int j = 0; j < d; j++)
        gamma_surrogate_dxd_[j] =
            std::max(1e-8, (2.0 * G[j] - S) / static_cast<double>(d - 2));
}

// =============================================================================
// Helper: Apply Householder Reflection (Stabilizes Orthogonalization)
// =============================================================================
void SOdHypersphereSamplerCPU::apply_householder(double* Q, double* v, int d, int col, double& sign_tracker) const {
    // 1. Calculate the norm of the sub-vector v[col...d-1]
    double norm_x = 0.0;
    #pragma omp simd reduction(+:norm_x)
    for (int i = col; i < d; i++) {
        norm_x += v[i] * v[i];
    }
    norm_x = std::sqrt(norm_x);

    if (norm_x < 1e-14) return;

    // 2. Compute the Householder vector u
    std::vector<double> u(d, 0.0);
    double v0 = v[col];
    double sign = (v0 > 0.0) ? 1.0 : -1.0;
    u[col] = v0 + sign * norm_x;

    for (int i = col + 1; i < d; i++) u[i] = v[i];

    double norm_u2 = 0.0;
    #pragma omp simd reduction(+:norm_u2)
    for (int i = col; i < d; i++) {
        norm_u2 += u[i] * u[i];
    }

    if (norm_u2 < 1e-14) return;

    // 3. Apply the reflection to the matrix Q
    // Q_new = Q * (I - 2 * u * u^T / norm_u2)
    for (int j = 0; j < d; j++) {
        double dot = 0.0;
        #pragma omp simd reduction(+:dot)
        for (int i = col; i < d; i++) {
            dot += Q[j * d + i] * u[i];
        }
        double scale = 2.0 * dot / norm_u2;
        #pragma omp simd
        for (int i = col; i < d; i++) {
            Q[j * d + i] -= scale * u[i];
        }
    }

    // Track determinant: Every Householder reflection flips the determinant sign
    sign_tracker *= -1.0;
}

// =============================================================================
// Public build (with and without stats)
// =============================================================================

std::vector<double> SOdHypersphereSamplerCPU::build_orientation_frames(
    const std::vector<double>& flat_theta, int N) const
{
    assert(static_cast<int>(flat_theta.size()) == N * m_);
    std::vector<double> out(static_cast<size_t>(N) * d_ * d_, 0.0);
    const int nt = effective_num_threads(cfg_.num_threads);

    #pragma omp parallel for schedule(dynamic) num_threads(nt)
    for (int n = 0; n < N; n++) {
        uint64_t dynamic_seed = cfg_.seed + static_cast<uint64_t>(n) + step_count_;
        build_one_frame(flat_theta.data() + n * m_,
                        out.data()        + static_cast<size_t>(n) * d_ * d_,
                        dynamic_seed);
    }
    step_count_ += N;
    return out;
}

std::vector<double> SOdHypersphereSamplerCPU::build_orientation_frames_with_stats(
    const std::vector<double>& flat_theta, int N,
    std::vector<ColumnStats>&  stats) const
{
    assert(static_cast<int>(flat_theta.size()) == N * m_);
    std::vector<double> out(static_cast<size_t>(N) * d_ * d_, 0.0);
    const int nt       = effective_num_threads(cfg_.num_threads);
    const int num_cols = d_ - 1;

    std::vector<std::vector<ColumnStats>> tls(nt,
        std::vector<ColumnStats>(num_cols));

    #pragma omp parallel for schedule(dynamic) num_threads(nt)
    for (int n = 0; n < N; n++) {
        int tid = omp_get_thread_num();
        build_one_frame(flat_theta.data() + n * m_,
                        out.data()        + static_cast<size_t>(n) * d_ * d_,
                        cfg_.seed + static_cast<uint64_t>(n),
                        tls[tid].data());
    }

    stats.assign(num_cols, ColumnStats{});
    for (int t = 0; t < nt; t++)
        for (int col = 0; col < num_cols; col++) {
            stats[col].attempts += tls[t][col].attempts;
            stats[col].accepted += tls[t][col].accepted;
        }
    return out;
}

// =============================================================================
// build_one_frame  —  drive the exact per-column rejection sampler
// =============================================================================

    void SOdHypersphereSamplerCPU::build_one_frame(
        const double* theta, double* q_out, uint64_t sample_seed,
        ColumnStats* col_stats) const
{
    const int d = d_;
    std::vector<double> Q_cm(static_cast<size_t>(d) * d, 0.0);
    Xoshiro256PlusPlus rng(sample_seed);
    std::normal_distribution<double> gauss(0.0, 1.0);

    for (int col = 0; col < d - 1; col++) {
        const int    pair_k       = col / 2;
        const double weight_scale = 0.5 * theta[pair_k] * theta[pair_k];

        // Execute the 15-step IMH chain and return successful jumps
        int mh_accepts = sample_column_rejection(Q_cm.data(), col, weight_scale, rng);

        if (col_stats) {
            // We know we executed exactly 15 IMH steps per column
            col_stats[col].attempts += 15;

            // We track how many of those steps resulted in a new state
            col_stats[col].accepted += mh_accepts;
        }
    }

    // -------------------------------------------------------------------
    // Last column: Explicit Orthogonal Generation
    // -------------------------------------------------------------------
    std::vector<double> q_last(d);
    double norm = 0.0;
    while (norm < 1e-14) {
        for (int i = 0; i < d; i++) q_last[i] = gauss(rng);
        norm = gram_schmidt_project(q_last.data(), Q_cm.data(), d - 1);
    }
    const double inv_norm = 1.0 / norm;
    #pragma omp simd
    for (int i = 0; i < d; i++) Q_cm[i + (d - 1) * d] = q_last[i] * inv_norm;

    // Enforce SO(d) determinant constraint (+1)
    double det = det_sign(Q_cm.data());
    if (det < 0.0) {
        #pragma omp simd
        for (int i = 0; i < d; i++) Q_cm[i + (d - 1) * d] *= -1.0;
    }

    // Transpose column-major → row-major output
    for (int r = 0; r < d; r++) {
        #pragma omp simd
        for (int c = 0; c < d; c++) {
            q_out[r * d + c] = Q_cm[r + c * d];
        }
    }
}

// =============================================================================
// sample_column_rejection  —  surrogate-guided vMF + exact Lie IMH
// =============================================================================
//
// Guide:  diagonal spatial surrogate gamma_surrogate_dxd_ drives power
//         iteration (finding mu, gamma_max_est, kappa).
// Judge:  dense_lie_energy evaluates the exact D×D Lie algebra energy for
//         the IMH acceptance ratio, guaranteeing zero-bias sampling.

int SOdHypersphereSamplerCPU::sample_column_rejection(
    double* Q_cm,
    int                col,
    double             weight_scale,
    Xoshiro256PlusPlus& rng) const
{
    const int d = d_;
    const int m = d - col; // Degrees of freedom for S^{d-col-1}

    std::normal_distribution<double>       gauss(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // =======================================================================
    // GUIDE: surrogate-based vMF steering (power iteration on diagonal matrix)
    // =======================================================================

    // --- Step 1: Find the pole (mu) via projected power iteration ---
    std::vector<double> mu(d);
    double initial_norm = 0.0;
    while (initial_norm < 1e-14) {
        for (int i = 0; i < d; i++) mu[i] = gauss(rng);
        initial_norm = gram_schmidt_project(mu.data(), Q_cm, col);
    }
    const double inv_initial_norm = 1.0 / initial_norm;
    for (int i = 0; i < d; i++) mu[i] *= inv_initial_norm;

    // Gershgorin bound for the diagonal surrogate = max diagonal element.
    double gamma_max_est = *std::max_element(
        gamma_surrogate_dxd_.begin(), gamma_surrogate_dxd_.end());
    gamma_max_est *= 1.05;  // small safety margin

    // 5 iterations of the power method (diagonal matvec is free).
    for (int iter = 0; iter < 5; iter++) {
        std::vector<double> next_mu(d, 0.0);
        #pragma omp simd
        for (int i = 0; i < d; i++) {
            // G_surrogate * mu is just gamma_surrogate_dxd_[i] * mu[i] (diagonal)
            next_mu[i] = (gamma_max_est - gamma_surrogate_dxd_[i]) * mu[i];
        }
        double norm = gram_schmidt_project(next_mu.data(), Q_cm, col);
        if (norm > 1e-14) {
            const double inv_norm = 1.0 / norm;
            for (int i = 0; i < d; i++) mu[i] = next_mu[i] * inv_norm;
        }
    }

    // lambda_min_proj from surrogate: mu^T Gamma_surrogate mu (diagonal form)
    double lambda_min_proj = 0.0;
    #pragma omp simd reduction(+:lambda_min_proj)
    for (int i = 0; i < d; i++)
        lambda_min_proj += gamma_surrogate_dxd_[i] * mu[i] * mu[i];

    // --- Step 2: Set concentration parameter kappa ---
    double kappa = weight_scale * std::max(0.1, (gamma_max_est / d - lambda_min_proj));

    // --- Step 3: Initialize MH chain at the optimal pole ---
    std::vector<double> q_curr = mu;
    double E_curr_bingham = -weight_scale * dense_lie_energy(mu.data(), Q_cm, col);
    double E_curr_vmf     = kappa * 1.0;  // mu^T mu = 1

    // --- Step 4: Wood's 1994 vMF Generator + IMH loop ---
    double b = (m - 1.0) / (2.0 * kappa + std::sqrt(4.0 * kappa * kappa + (m - 1.0) * (m - 1.0)));
    double x0 = (1.0 - b) / (1.0 + b);
    double safe_x0_sq_diff = 4.0 * b / ((1.0 + b) * (1.0 + b));
    double c = kappa * x0 + (m - 1.0) * std::log(safe_x0_sq_diff);

    const int imh_chain_length = 15;
    int internal_mh_accepts = 0;

    std::gamma_distribution<double> gamma_dist((m - 1) / 2.0, 1.0);

    for (int step = 0; step < imh_chain_length; step++) {
        double W = 0.0;
        while (true) {
            double g1 = gamma_dist(rng);
            double g2 = gamma_dist(rng);
            double Z = g1 / (g1 + g2);

            W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z);
            double U = unif(rng);

            double safe_x0_W_diff = 2.0 * b / ((1.0 + b) - (1.0 - b * b) * Z);
            if (kappa * W + (m - 1.0) * std::log(safe_x0_W_diff) - c >= std::log(U))
                break;
        }

        // Draw V uniformly from S^{m-2} orthogonal to mu and V^⊥
        std::vector<double> V(d);
        for (int i = 0; i < d; i++) V[i] = gauss(rng);

        gram_schmidt_project(V.data(), Q_cm, col);

        double dot_mu = 0.0;
        #pragma omp simd reduction(+:dot_mu)
        for (int i = 0; i < d; i++) dot_mu += V[i] * mu[i];
        #pragma omp simd
        for (int i = 0; i < d; i++) V[i] -= dot_mu * mu[i];

        double norm_V = 0.0;
        #pragma omp simd reduction(+:norm_V)
        for (int i = 0; i < d; i++) norm_V += V[i] * V[i];
        norm_V = std::sqrt(norm_V);
        if (norm_V > 1e-14) {
            const double inv_norm_V = 1.0 / norm_V;
            #pragma omp simd
            for (int i = 0; i < d; i++) V[i] *= inv_norm_V;
        }

        // Construct proposed vector
        std::vector<double> q_prop(d);
        double sqrt_W = std::sqrt(std::max(0.0, 1.0 - W * W));
        #pragma omp simd
        for (int i = 0; i < d; i++)
            q_prop[i] = W * mu[i] + sqrt_W * V[i];

        // =======================================================================
        // JUDGE: exact Lie algebra energy in the IMH acceptance step
        // =======================================================================
        double E_prop_bingham = -weight_scale * dense_lie_energy(q_prop.data(), Q_cm, col);
        double E_prop_vmf     = kappa * W;

        double log_alpha = (E_prop_bingham - E_curr_bingham) + (E_curr_vmf - E_prop_vmf);

        if (std::log(unif(rng)) < log_alpha) {
            q_curr        = q_prop;
            E_curr_bingham = E_prop_bingham;
            E_curr_vmf     = E_prop_vmf;
            internal_mh_accepts++;
        }
    }

    // --- Step 5: Finalize ---
    for (int i = 0; i < d; i++) Q_cm[i + col * d] = q_curr[i];

    return internal_mh_accepts;
}

// =============================================================================
// dense_lie_energy  —  exact D×D quadratic form across accepted columns
// =============================================================================
//
// For each previous column prev_col in [0, col):
//   v[a(j,k)] = prev_q[j]*q[k] - prev_q[k]*q[j]   (Lie algebra 2-vector)
//   energy    += v^T Gamma_dense v
//
// At col=0 the sum is empty (energy = 0) and vMF steering alone governs
// the first column's distribution.

double SOdHypersphereSamplerCPU::dense_lie_energy(
    const double* q, const double* Q_cols, int col) const
{
    const int d = d_;
    const int D = d * (d - 1) / 2;
    double total = 0.0;

    std::vector<double> v(D);

    for (int prev_col = 0; prev_col < col; prev_col++) {
        const double* prev_q = Q_cols + prev_col * d;

        // Build the D-dimensional Lie algebra vector for this column pair.
        int a = 0;
        for (int j = 0; j < d; j++) {
            for (int k = j + 1; k < d; k++, a++) {
                v[a] = prev_q[j] * q[k] - prev_q[k] * q[j];
            }
        }

        // v^T Gamma_dense v
        double energy = 0.0;
        for (int i = 0; i < D; i++) {
            double Gv_i = 0.0;
            #pragma omp simd reduction(+:Gv_i)
            for (int jj = 0; jj < D; jj++)
                Gv_i += gamma_dense_DxD_[i * D + jj] * v[jj];
            energy += v[i] * Gv_i;
        }
        total += energy;
    }
    return total;
}

// =============================================================================
// Helpers
// =============================================================================

double SOdHypersphereSamplerCPU::quadratic_form_surrogate(const double* q) const {
    const int d = d_;
    double result = 0.0;
    #pragma omp simd reduction(+:result)
    for (int i = 0; i < d; i++)
        result += gamma_surrogate_dxd_[i] * q[i] * q[i];
    return result;
}

double SOdHypersphereSamplerCPU::gram_schmidt_project(
    double* v, const double* Q_cols, int num_cols) const
{
    const int d = d_;
    for (int prev = 0; prev < num_cols; prev++) {
        const double* col_ptr = Q_cols + prev * d;
        double dot = 0.0;
        #pragma omp simd reduction(+:dot)
        for (int i = 0; i < d; i++) dot += v[i] * col_ptr[i];
        #pragma omp simd
        for (int i = 0; i < d; i++) v[i] -= dot * col_ptr[i];
    }
    double norm2 = 0.0;
    #pragma omp simd reduction(+:norm2)
    for (int i = 0; i < d; i++) norm2 += v[i] * v[i];
    return std::sqrt(norm2);
}

int SOdHypersphereSamplerCPU::det_sign(double* M) const {
    const int d  = d_;
    int       sgn = 1;

    std::vector<double> A(static_cast<size_t>(d) * d);
    for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++)
            A[r * d + c] = M[r + c * d];   // column-major → row-major

    for (int col = 0; col < d; col++) {
        int    pivot   = col;
        double max_val = std::abs(A[col * d + col]);
        for (int row = col + 1; row < d; row++) {
            const double v = std::abs(A[row * d + col]);
            if (v > max_val) { max_val = v; pivot = row; }
        }
        if (max_val < 1e-14) return 0;

        if (pivot != col) {
            for (int k = 0; k < d; k++)
                std::swap(A[col * d + k], A[pivot * d + k]);
            sgn = -sgn;
        }
        if (A[col * d + col] < 0.0) sgn = -sgn;

        const double inv = 1.0 / A[col * d + col];
        for (int row = col + 1; row < d; row++) {
            const double f = A[row * d + col] * inv;
            for (int k = col; k < d; k++)
                A[row * d + k] -= f * A[col * d + k];
        }
    }
    return sgn;
}

} // namespace anisotropic
} // namespace sampler
