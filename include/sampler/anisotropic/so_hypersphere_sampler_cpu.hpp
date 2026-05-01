#pragma once

#include "sampler/xoshiro.hpp"
#include "sampler/thread_config.hpp"
#include <cstdint>
#include <random>
#include <vector>

namespace sampler {
namespace anisotropic {

// Phase II of Algorithm 4.8 — CPU implementation via Exact Sequential Rejection.
//
// For each orientation frame Q ∈ SO(d), columns are sampled sequentially.
// Column col is drawn from the Vector Bingham distribution restricted to the
// sphere S^{d-col-1} in the orthogonal complement of accepted columns.
//
// Surrogate MCMC trick: internally maintains a split-brain architecture:
//   Guide  — uses a diagonal d×d spatial surrogate (least-squares projection
//             of the D×D Lie tensor) to steer vMF proposals via power iteration.
//   Judge  — evaluates the exact D×D Lie algebra energy in the IMH acceptance
//             step, guaranteeing mathematically exact, zero-bias sampling.
//
// N frames are built in parallel with OpenMP.
class SOdHypersphereSamplerCPU {
public:
    struct Config {
        int      num_threads     = 1;
        double   aniso_threshold = 5.0;  // weight_scale trigger for vMF steering
        uint64_t seed            = static_cast<uint64_t>(std::random_device{}());
    };

    // gamma_flat: row-major D×D precision matrix in the Lie algebra so(d),
    //             where D = d*(d-1)/2.  Must be symmetric and positive-definite.
    // gamma_min:  smallest eigenvalue (used by Phase I; stored but not consumed
    //             by Phase II internally).
    SOdHypersphereSamplerCPU(int d,
                              const std::vector<double>& gamma_flat,
                              double gamma_min,
                              Config cfg);

    struct ColumnStats {
        int64_t attempts = 0;   // total rejection proposals for this col
        int64_t accepted = 0;   // successfully accepted proposals (always 1 per col)
        double  acceptance_rate() const {
            return attempts > 0 ? static_cast<double>(accepted) / attempts : 0.0;
        }
    };

    // Stabilizes orthogonalization and tracks the determinant parity.
    void apply_householder(double* Q, double* v, int d, int col, double& sign_tracker) const;

    // Build orientation frames for N samples.
    //
    // flat_theta : length N*m, sample n occupies [n*m, (n+1)*m).
    // Returns    : flat Q matrices, length N*d*d, sample n occupies
    //              [n*d*d, (n+1)*d*d) in row-major order.
    std::vector<double> build_orientation_frames(const std::vector<double>& flat_theta,
                                                  int N) const;

    // Like build_orientation_frames but also returns per-column rejection
    // stats summed over all N samples. stats is resized to d-1.
    std::vector<double> build_orientation_frames_with_stats(
        const std::vector<double>& flat_theta,
        int N,
        std::vector<ColumnStats>&  stats) const;

    void update_gamma(const std::vector<double>& gamma_flat, double gamma_min);

private:
    int d_, m_;
    std::vector<double> gamma_dense_DxD_;      // row-major D×D Lie algebra precision
    std::vector<double> gamma_surrogate_dxd_;  // length d, diagonal spatial surrogate
    double gamma_min_;
    Config cfg_;

    mutable uint64_t step_count_ = 0;

    // Build one d×d row-major Q matrix from theta[m] using the given rng seed.
    // If col_stats != nullptr, increments col_stats[col].attempts/.accepted.
    void build_one_frame(const double* theta,
                         double* q_out,
                         uint64_t      sample_seed,
                         ColumnStats* col_stats = nullptr) const;

    // Runs the exact rejection sampler on S^{d-col-1} ∩ V^⊥ (the orthogonal complement
    // of the first `col` columns of Q_cm). On return, Q_cm[:, col] is filled with
    // the accepted sample. Returns the number of attempts it took to accept.
    int sample_column_rejection(double* Q_cm,
                                int                col,
                                double             weight_scale,
                                Xoshiro256PlusPlus& rng) const;

    // q^T Gamma_surrogate q using the diagonal spatial surrogate.
    double quadratic_form_surrogate(const double* q) const;

    // Exact Lie algebra energy: sum over prev_col in [0,col) of
    //   v^T Gamma_dense v  where v[a] = prev_q[j]*q[k] - prev_q[k]*q[j]
    // for basis element (j,k) at index a.
    double dense_lie_energy(const double* q, const double* Q_cols, int col) const;

    // Recompute gamma_surrogate_dxd_ from gamma_dense_DxD_ via least-squares projection.
    void compute_surrogate();

    // In-place Gram-Schmidt: project v ∈ ℝ^d away from the first num_cols
    // columns of Q_cols (column-major d×num_cols). Returns the residual norm.
    double gram_schmidt_project(double* v,
                                const double* Q_cols,
                                int num_cols) const;

    // Compute sign(det(Q)) for a column-major d×d matrix via Gaussian elimination.
    int det_sign(double* Q_copy) const;
};

} // namespace anisotropic
} // namespace sampler