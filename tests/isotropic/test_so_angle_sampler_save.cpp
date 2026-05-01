#include "sampler/isotropic/so_angle_sampler.hpp"
#include "sampler/isotropic/so_gaussian_sampler.hpp"
#include "sampler/thread_config.hpp"

#include <hdf5.h>
#include <isomorphism/math.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace math = isomorphism::math;
using Tensor   = isomorphism::Tensor;
using DType    = isomorphism::DType;

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

static constexpr int    D       = 50;    // SO(d) dimension → m = 3 angles
static constexpr int    N       = 50000; // samples per alpha
static constexpr double ALPHAS[]= {0.001, 0.04, 1.0, 100.0, 100000.0, 1000000.0, 10000000.0, 1000000000.0, 100000000000000000.0};
static constexpr int    N_ALPHA = 9;
static const char*      OUTFILE = "so_angle_samples.h5";

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

// Write a scalar double attribute onto an open HDF5 object.
static void write_double_attr(hid_t obj, const char* name, double val) {
    hid_t sp  = H5Screate(H5S_SCALAR);
    hid_t aid = H5Acreate2(obj, name, H5T_NATIVE_DOUBLE,
                           sp, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(aid, H5T_NATIVE_DOUBLE, &val);
    H5Aclose(aid);
    H5Sclose(sp);
}

static void write_int_attr(hid_t obj, const char* name, int val) {
    hid_t sp  = H5Screate(H5S_SCALAR);
    hid_t aid = H5Acreate2(obj, name, H5T_NATIVE_INT,
                           sp, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(aid, H5T_NATIVE_INT, &val);
    H5Aclose(aid);
    H5Sclose(sp);
}

// -----------------------------------------------------------------------------
// High-α tangent-space angle sampler
// -----------------------------------------------------------------------------
// For α ≥ kHighAlphaThreshold the HMC chain freezes (step size → 0).
// Instead, sample angles from a scaled skew-symmetric Gaussian matrix:
//   Ω = (A − Aᵀ) / √(2α),  A_ij ~ N(0,1)
// Eigenvalues of a real skew-symmetric matrix are purely imaginary ±iθⱼ.
// Its singular values therefore come in pairs (θⱼ, θⱼ) in decreasing order,
// so the m principal angles are the even-indexed singular values S[2j].

static std::vector<double> sample_tangent_angles(int d, int m, int N, double alpha) {
    float  scale = static_cast<float>(1.0 / std::sqrt(2.0 * alpha));
    Tensor A     = math::random_normal({N, d, d}, DType::Float32);
    Tensor Omega = math::multiply(
        math::subtract(A, math::transpose(A, {0, 2, 1})),
        Tensor(scale, DType::Float32));                         // [N, d, d]

    // B = Ω·Ωᵀ is symmetric PSD with eigenvalues θⱼ² (paired, ascending).
    Tensor B      = math::matmul(Omega, math::transpose(Omega, {0, 2, 1}));
    Tensor eigvals = math::eigvalsh(B);                         // [N, d] ascending
    math::eval(eigvals);

    std::vector<double> flat(static_cast<std::size_t>(N * m));
    for (int n = 0; n < N; ++n) {
        Tensor En = math::slice(eigvals, n, n + 1, 0);         // [1, d]
        for (int j = 0; j < m; ++j) {
            int    idx = d - 1 - 2 * j;
            Tensor v   = math::slice(En, idx, idx + 1, 1);     // [1, 1]
            double ev  = math::to_double(v);
            flat[static_cast<std::size_t>(n * m + j)] = std::sqrt(ev > 0.0 ? ev : 0.0);
        }
    }
    return flat;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
    math::set_default_device_cpu();
    sampler::set_num_threads(4);

    const int m = D / 2;   // number of angles
    std::printf("SO(%d) angle sampler — m=%d angles, %d samples per alpha\n"
                "Output: %s\n\n", D, m, N, OUTFILE);

    // -------------------------------------------------------------------------
    // Create HDF5 file
    // -------------------------------------------------------------------------
    hid_t file_id = H5Fcreate(OUTFILE, H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        std::fprintf(stderr, "ERROR: could not create %s\n", OUTFILE);
        return 1;
    }

    // Store global metadata on the root group.
    hid_t root = H5Gopen2(file_id, "/", H5P_DEFAULT);
    write_int_attr(root, "d", D);
    write_int_attr(root, "m", m);
    write_int_attr(root, "n_samples", N);
    H5Gclose(root);

    // HMC is only viable below this concentration — above it the step size
    // collapses to zero and the chain freezes.
    static constexpr double HMC_THRESHOLD = 1e9;

    auto write_dataset = [&](hid_t gid, const char* name,
                             const std::vector<double>& data) {
        hsize_t dims[2] = {static_cast<hsize_t>(N), static_cast<hsize_t>(m)};
        hid_t   sid = H5Screate_simple(2, dims, nullptr);
        hid_t   did = H5Dcreate2(gid, name, H5T_NATIVE_DOUBLE, sid,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(did, H5T_NATIVE_DOUBLE,
                 H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
        H5Dclose(did);
        H5Sclose(sid);
    };

    // -------------------------------------------------------------------------
    // Sample for each alpha and write to a group
    // -------------------------------------------------------------------------
    for (int ai = 0; ai < N_ALPHA; ++ai) {
        double alpha = ALPHAS[ai];
        std::printf("alpha = %.10g\n", alpha);

        char group_name[64];
        std::snprintf(group_name, sizeof(group_name), "alpha_%.10g", alpha);
        hid_t gid = H5Gcreate2(file_id, group_name,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        write_double_attr(gid, "alpha", alpha);
        write_int_attr(gid, "d", D);
        write_int_attr(gid, "m", m);

        // --- Tangent-space sampler (always) ---
        std::printf("  [tangent-space sampler]\n");
        write_dataset(gid, "angles_tangent",
                      sample_tangent_angles(D, m, N, alpha));
        std::printf("  tangent: collected %d samples\n", N);

        // --- HMC sampler (sub-threshold only) ---
        const int has_hmc = (alpha < HMC_THRESHOLD) ? 1 : 0;
        write_int_attr(gid, "has_hmc", has_hmc);

        if (has_hmc) {
            sampler::SOdAngleSampler::Config cfg;
            cfg.d          = D;
            cfg.alpha      = alpha;
            cfg.num_chains = 1;
            cfg.burn_in    = 2000;
            cfg.seed       = 42 + static_cast<uint64_t>(ai);

            sampler::SOdAngleSampler samp(cfg);
            std::printf("  [HMC sampler]  acceptance_rate = %.3f\n",
                        samp.acceptance_rate());

            std::vector<double> hmc_angles(static_cast<std::size_t>(N * m));
            for (int i = 0; i < N; ++i) {
                std::vector<double> a = samp.sample_angles(1);
                for (int j = 0; j < m; ++j)
                    hmc_angles[static_cast<std::size_t>(i * m + j)] =
                        a[static_cast<std::size_t>(j)];
            }
            write_dataset(gid, "angles_hmc", hmc_angles);
            std::printf("  HMC:     collected %d samples\n", N);
        }

        H5Gclose(gid);
    }

    H5Fclose(file_id);
    std::printf("\nSaved to %s\n", OUTFILE);
    return 0;
}
