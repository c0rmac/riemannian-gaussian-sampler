#include "sampler/anisotropic/so_anisotropic_angle_sampler.hpp"
#include "sampler/anisotropic/so_hypersphere_sampler_cpu.hpp"
#include "sampler/thread_config.hpp"

#include <hdf5.h>
#include <isomorphism/math.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace math = isomorphism::math;
using AngleSampler = sampler::anisotropic::SOdAnisotropicAngleSampler;
using HyperSampler = sampler::anisotropic::SOdHypersphereSamplerCPU;

// =============================================================================
// Configuration
// =============================================================================

static constexpr int    D          = 4;
static constexpr int    N_SAMPLES  = 90000;
static constexpr double GAMMA_LOW  = 0.2;
static constexpr double GAMMA_HIGH = 5.0;
static const char*      OUTFILE    = "so_anisotropic_samples.h5";

// =============================================================================
// HDF5 helpers
// =============================================================================

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

// Write a flat double vector as an n-dimensional HDF5 dataset.
static void write_dataset(hid_t loc, const char* name,
                          const std::vector<double>& data,
                          const std::vector<hsize_t>& dims)
{
    hid_t sid = H5Screate_simple(static_cast<int>(dims.size()),
                                  dims.data(), nullptr);
    hid_t did = H5Dcreate2(loc, name, H5T_NATIVE_DOUBLE, sid,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(did, H5T_NATIVE_DOUBLE,
             H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(did);
    H5Sclose(sid);
}

// =============================================================================
// main
// =============================================================================

int main() {
    sampler::set_num_threads(8);
    math::set_default_device_gpu();

    const int d = D;
    const int N = N_SAMPLES;
    const int m = d / 2;

    std::printf("SO(%d) anisotropic Phase II visualisation\n", d);
    std::printf("  gamma_low=%.2f  gamma_high=%.2f  N=%d\n",
                GAMMA_LOW, GAMMA_HIGH, N);

    // Build spatial gamma (d×d) for Phase I.
    std::vector<double> gamma_spatial(d * d, 0.0);
    for (int i = 0; i < d; i++)
        gamma_spatial[i * d + i] = (i < d / 2) ? GAMMA_LOW : GAMMA_HIGH;

    // Build D×D Lie gamma for Phase II.
    const int D_lie = d * (d - 1) / 2;
    std::vector<double> gamma_lie(static_cast<size_t>(D_lie) * D_lie, 0.0);
    {
        int a = 0;
        for (int j = 0; j < d; j++)
            for (int k = j + 1; k < d; k++, a++)
                gamma_lie[a * D_lie + a] =
                    (j < d / 2 && k < d / 2) ? GAMMA_LOW : GAMMA_HIGH;
    }

    // --- Phase I: sample angles ---
    AngleSampler::Config acfg;
    acfg.d          = d;
    acfg.num_chains = 4;
    acfg.burn_in    = 1500;
    AngleSampler angle_samp(gamma_spatial, acfg);
    std::printf("  Phase I acceptance rate: %.3f\n", angle_samp.acceptance_rate());

    auto flat_theta = angle_samp.sample_angles(N);

    // --- Phase II: build orientation frames (D×D Lie gamma) ---
    HyperSampler::Config hcfg;
    hcfg.num_threads = sampler::effective_num_threads(8);
    HyperSampler hyper(d, gamma_lie, angle_samp.gamma_min(), hcfg);

    std::vector<HyperSampler::ColumnStats> stats;
    auto flat_Q = hyper.build_orientation_frames_with_stats(flat_theta, N, stats);

    std::printf("  Phase II per-column acceptance rates:\n");
    for (int col = 0; col < d - 1; col++)
        std::printf("    col %d: %.4f\n", col, stats[col].acceptance_rate());

    // --- Save to HDF5 ---
    hid_t fid = H5Fcreate(OUTFILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fid < 0) {
        std::fprintf(stderr, "ERROR: could not create %s\n", OUTFILE);
        return 1;
    }

    hid_t root = H5Gopen2(fid, "/", H5P_DEFAULT);
    write_int_attr(root, "d", d);
    write_int_attr(root, "N", N);
    write_int_attr(root, "m", m);
    write_double_attr(root, "gamma_min", angle_samp.gamma_min());
    write_double_attr(root, "gamma_max", GAMMA_HIGH);
    H5Gclose(root);

    // Gamma matrix (spatial d×d for reference)
    write_dataset(fid, "gamma", gamma_spatial, {static_cast<hsize_t>(d),
                                                 static_cast<hsize_t>(d)});

    // Angles: (N, m)
    write_dataset(fid, "angles", flat_theta, {static_cast<hsize_t>(N),
                                               static_cast<hsize_t>(m)});

    // Orientation frames: (N, d, d) row-major Q matrices
    write_dataset(fid, "frames", flat_Q, {static_cast<hsize_t>(N),
                                           static_cast<hsize_t>(d),
                                           static_cast<hsize_t>(d)});

    H5Fclose(fid);
    std::printf("  Saved to %s\n", OUTFILE);

    // --- Launch Python visualisation script ---
    std::string script = std::string(SOURCE_DIR) + "/visualise_so_samples.py";
    std::string cmd    = "python3 " + script + " " + OUTFILE;
    std::printf("  Running: %s\n", cmd.c_str());
    int ret = std::system(cmd.c_str());
    if (ret != 0)
        std::fprintf(stderr, "  WARNING: Python script exited with code %d\n", ret);

    return 0;
}
