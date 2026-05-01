#pragma once

#include "sampler/xoshiro.hpp"
#include "sampler/thread_config.hpp"
#include <cstdint>
#include <random>
#include <vector>

namespace sampler {
namespace anisotropic {

// Phase II of Algorithm 4.8 — CPU implementation for the spatial-precision variant.
//
// Identical in structure to SOdHypersphereSamplerCPU but takes the d-dimensional
// diagonal spatial surrogate directly instead of a full D×D Lie algebra matrix.
// This eliminates the compute_surrogate() projection step and makes the Guide and
// Judge consistent under the spatial approximation:
//
//   Guide: vMF steering via power iteration on gamma_spatial (diagonal d-vector).
//   Judge: per-plane energy  gamma_plane(j,k) = (gamma_spatial[j] + gamma_spatial[k]) / 2
//          summed over all Lie basis elements — no off-diagonal Lie terms.
//
// Suitable when only the spatial precision is known (e.g. the CMA-ES path which
// maintains a d×d covariance rather than the full D×D Lie tensor).
class SOdHypersphereSamplerCPUSpatial {
public:
    struct Config {
        int      num_threads     = 1;
        double   aniso_threshold = 5.0;
        uint64_t seed            = static_cast<uint64_t>(std::random_device{}());
    };

    // gamma_spatial: length d, diagonal entries of the d×d spatial precision.
    // gamma_min:     minimum eigenvalue from Phase I (stored; not consumed internally).
    SOdHypersphereSamplerCPUSpatial(int d,
                                    const std::vector<double>& gamma_spatial,
                                    double gamma_min,
                                    Config cfg);

    struct ColumnStats {
        int64_t attempts = 0;
        int64_t accepted = 0;
        double  acceptance_rate() const {
            return attempts > 0 ? static_cast<double>(accepted) / attempts : 0.0;
        }
    };

    void apply_householder(double* Q, double* v, int d, int col, double& sign_tracker) const;

    // Build orientation frames for N samples.
    //
    // flat_theta : length N*m, sample n occupies [n*m, (n+1)*m).
    // Returns    : flat Q matrices, length N*d*d, sample n occupies
    //              [n*d*d, (n+1)*d*d) in row-major order.
    std::vector<double> build_orientation_frames(const std::vector<double>& flat_theta,
                                                  int N) const;

    std::vector<double> build_orientation_frames_with_stats(
        const std::vector<double>& flat_theta,
        int N,
        std::vector<ColumnStats>&  stats) const;

    // gamma_spatial: length d diagonal entries.
    void update_gamma(const std::vector<double>& gamma_spatial, double gamma_min);

private:
    int d_, m_;
    std::vector<double> gamma_spatial_d_;  // length d, diagonal spatial precision
    double gamma_min_;
    Config cfg_;

    mutable uint64_t step_count_ = 0;

    void build_one_frame(const double* theta,
                         double*       q_out,
                         uint64_t      sample_seed,
                         ColumnStats*  col_stats = nullptr) const;

    int sample_column_rejection(double*             Q_cm,
                                int                  col,
                                double               weight_scale,
                                Xoshiro256PlusPlus&  rng) const;

    // Per-plane spatial approximation of the Lie energy:
    //   sum_{prev_col < col} sum_{j<k} (gamma[j]+gamma[k])/2 * (p[j]*q[k]-p[k]*q[j])^2
    double spatial_lie_energy(const double* q, const double* Q_cols, int col) const;

    double gram_schmidt_project(double* v, const double* Q_cols, int num_cols) const;

    int det_sign(double* M) const;
};

} // namespace anisotropic
} // namespace sampler
