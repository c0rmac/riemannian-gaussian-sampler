# Riemannian Gaussian Sampler

Exact spectral sampling from isotropic Gaussian distributions on the **Stiefel manifold** $V(n, k)$ and the **rotation group** $SO(d)$.

Both samplers produce samples that lie *exactly* on the manifold (up to floating-point precision) with the correct concentration around a user-supplied mean frame, at a fraction of the cost of rejection sampling or geodesic random walks.

---

## Algorithms

| Manifold                | Algorithm | Phase I | Phase II |
|-------------------------|---|---|---|
| **$V(n, k)$** — Stiefel | Algorithm 4.2 | HMC over k principal angles | O(k³) spectral lift via matrix exponential on the 2k×2k active core |
| **$SO(d)$** — Rotations | Algorithm 4.3 | HMC over ⌊d/2⌋ rotation angles | O(d) exact Givens-block diagonal + Haar O(d) conjugation |

Phase I runs once (burn-in) at construction time. Each call to `sample()` runs Phase II only — producing a batch of N manifold-valued samples in a single pass.

---

## Installation

### Prerequisites: isomorphism

This library uses [isomorphism](https://github.com/c0rmac/isomorphism) as its tensor backend.
Install it first, choosing the backend that matches your hardware:

```bash
brew tap c0rmac/homebrew-isomorphism

# Apple Silicon (recommended — uses the Metal GPU via MLX)
brew install isomorphism --with-mlx

# Any CPU (Eigen)
brew install isomorphism --with-eigen

# PyTorch / LibTorch
brew install isomorphism --with-torch
```

For full installation options and building from source, see the
[isomorphism documentation](https://github.com/c0rmac/isomorphism).

### Install the sampler

```bash
brew tap c0rmac/homebrew-riemannian-gaussian-sampler
brew install riemannian-gaussian-sampler
```

### Building from source

```bash
git clone https://github.com/c0rmac/riemannian-gaussian-sampler.git
cd riemannian-gaussian-sampler
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build
```

If isomorphism is not on the default CMake prefix path, point to its source tree directly:

```bash
cmake -S . -B build \
  -DISOMORPHISM_DIR=/path/to/isomorphism \
  -DCMAKE_BUILD_TYPE=Release
```

### CMake integration

```cmake
find_package(sampler REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE sampler::sampler)
```

---

## Quick Examples

### $SO(d)$ — sample a random rotation near the identity

```cpp
#include <sampler/so_gaussian_sampler.hpp>
#include <isomorphism/math.hpp>

namespace math = isomorphism::math;

int main() {
    const int d = 4;

    // Mean rotation: identity
    isomorphism::Tensor M_hat = math::eye(d, isomorphism::DType::Float32);

    sampler::SOdGaussianSampler::Config cfg;
    cfg.alpha       = 2.0;   // concentration: higher = tighter around M_hat
    cfg.num_samples = 1;

    sampler::SOdGaussianSampler samp(M_hat, d, cfg);

    // Returns [1, d, d].  Call sample() repeatedly — burn-in only runs once.
    isomorphism::Tensor X = samp.sample();
}
```

### $V(n, k)$ — sample a random frame near a given frame

```cpp
#include <sampler/stiefel_gaussian_sampler.hpp>
#include <isomorphism/math.hpp>

namespace math = isomorphism::math;

int main() {
    const int n = 10, k = 3;

    // Mean frame: first k columns of I_n (canonical Stiefel origin)
    std::vector<float> data(n * k, 0.f);
    for (int i = 0; i < k; ++i) data[i * k + i] = 1.f;
    isomorphism::Tensor X_hat = math::array(data, {n, k}, isomorphism::DType::Float32);

    sampler::StiefelGaussianSampler::Config cfg;
    cfg.alpha       = 1.0;
    cfg.num_samples = 1;

    sampler::StiefelGaussianSampler samp(X_hat, n, k, cfg);

    // Returns [n, k].  X^T X ≈ I_k.
    isomorphism::Tensor X = samp.sample();
}
```

---

## Advanced Examples

### Batched sampling

Set `num_samples = N` to draw a batch of N independent samples in one call.
The returned tensor has shape `[N, d, d]` (SO) or `[N, n, k]` (Stiefel).

```cpp
sampler::SOdGaussianSampler::Config cfg;
cfg.alpha                  = 1.0;
cfg.num_samples            = 256;   // 256 independent SO(4) rotations per call
cfg.angle_cfg.num_chains   = 256;   // one HMC chain per sample — fully parallel
cfg.angle_cfg.num_threads  = 8;     // use 8 threads across those chains

sampler::SOdGaussianSampler samp(M_hat, 4, cfg);

isomorphism::Tensor batch = samp.sample();  // [256, 4, 4]
math::eval(batch);                          // flush lazy evaluation (MLX backend)
```

The `num_chains` chains run in parallel during burn-in; at sampling time each
chain advances one thinned HMC step and the resulting angle vector is lifted to
the manifold independently, giving exact i.i.d. samples across the batch.

### Controlling thread usage

By default all samplers are single-threaded (polite for library consumers). To
enable parallelism, set `num_threads` per sampler or globally:

```cpp
#include <sampler/thread_config.hpp>

// Per-sampler: use 4 threads for this sampler's HMC chains
cfg.angle_cfg.num_threads = 4;
cfg.angle_cfg.num_chains  = 16;

// Global override: cap all samplers to 8 threads for this process
sampler::set_num_threads(8);

// Clear the global cap (revert to per-sampler values)
sampler::set_num_threads(0);
```

The global override (`set_num_threads`) takes priority over the per-sampler
`num_threads` field. This lets you tune thread usage once at application startup
without modifying individual sampler configs.

### Interoperability with native backend types

If your application already holds MLX arrays, torch tensors, or Eigen matrices
you can pass them directly to the sampler without any CPU round-trip using the
interop headers provided by isomorphism.

#### MLX (Apple Silicon)

```cpp
#include <isomorphism/interop/mlx.hpp>
#include <sampler/so_gaussian_sampler.hpp>

namespace iso_mlx = isomorphism::interop::mlx;

// Wrap a native MLX array as an isomorphism::Tensor (zero-copy)
mlx::core::array m_hat_mlx = /* your array */;
isomorphism::Tensor M_hat = iso_mlx::wrap(m_hat_mlx);

sampler::SOdGaussianSampler samp(M_hat, d, cfg);
isomorphism::Tensor X = samp.sample();

// Unwrap the result back to a native MLX array (zero-copy)
mlx::core::array X_mlx = iso_mlx::unwrap(X);
```

#### PyTorch / LibTorch

```cpp
#include <isomorphism/interop/torch.hpp>
#include <sampler/stiefel_gaussian_sampler.hpp>

namespace iso_torch = isomorphism::interop::torch;

torch::Tensor x_hat_torch = torch::eye(n).narrow(1, 0, k);
isomorphism::Tensor X_hat = iso_torch::wrap(x_hat_torch);

sampler::StiefelGaussianSampler samp(X_hat, n, k, cfg);
isomorphism::Tensor X = samp.sample();

torch::Tensor X_torch = iso_torch::unwrap(X);
```

#### Eigen (CPU)

```cpp
#include <isomorphism/interop/eigen.hpp>
#include <sampler/stiefel_gaussian_sampler.hpp>

namespace iso_eigen = isomorphism::interop::eigen;

Eigen::MatrixXf x_hat_eigen = Eigen::MatrixXf::Identity(n, k);
isomorphism::Tensor X_hat = iso_eigen::wrap(x_hat_eigen);

sampler::StiefelGaussianSampler samp(X_hat, n, k, cfg);
isomorphism::Tensor X = samp.sample();

// Zero-copy map back into Eigen (contiguous tensors only)
auto X_map = iso_eigen::to_matrix_map(X);   // Eigen::Map<MatrixXf>

// Or copy into a new MatrixXf
Eigen::MatrixXf X_eigen = iso_eigen::to_matrix(X);
```

### Reusing a sampler across algorithm steps

`sample()` is cheap to call repeatedly — burn-in runs only at construction
time. The typical pattern for an iterative algorithm is:

```cpp
// Build once (triggers burn-in)
sampler::SOdGaussianSampler samp(M_hat, d, cfg);

for (int step = 0; step < n_steps; ++step) {
    // Draw a fresh batch — only Phase II runs
    isomorphism::Tensor X = samp.sample();
    math::eval(X);

    // ... update M_hat based on X ...

    // If the mean frame changes substantially, rebuild with the new mean.
    // This does NOT re-run burn-in; call rebuild_angle_sampler() explicitly
    // only if alpha or the HMC config has changed.
    samp = sampler::SOdGaussianSampler(new_M_hat, d, cfg);
}
```

### Monitoring HMC health

```cpp
// Acceptance rate after burn-in should sit in [0.30, 0.95].
// Values outside this range suggest epsilon needs tuning.
double acc = samp.angle_acceptance_rate();
std::printf("HMC acceptance rate: %.3f\n", acc);

// Adjust epsilon and warm-up via angle_cfg if needed:
cfg.angle_cfg.init_epsilon  = 5e-4;   // coarser initial step size
cfg.angle_cfg.target_accept = 0.70;   // dual-averaging target
cfg.angle_cfg.burn_in       = 3000;   // longer warm-up
```

---

## Configuration Reference

### `SOdGaussianSampler::Config`

| Field | Default | Description                                                   |
|---|---|---------------------------------------------------------------|
| `num_samples` | `1` | Batch size N — number of $SO(d)$ matrices per `sample()` call |
| `alpha` | `1.0` | Concentration α = λ/δ². Higher = tighter around M̂            |
| `dtype` | `Float32` | Tensor element type                                           |
| `angle_cfg` | — | Nested `SOdAngleSampler::Config` (see below)                  |

### `StiefelGaussianSampler::Config`

| Field | Default | Description                                                  |
|---|---|--------------------------------------------------------------|
| `num_samples` | `1` | Batch size N — number of $V(n,k)$ frames per `sample()` call |
| `alpha` | `1.0` | Concentration α = λ/δ². Higher = tighter around X̂           |
| `dtype` | `Float32` | Tensor element type                                          |
| `angle_cfg` | — | Nested `PrincipalAngleSampler::Config` (see below)           |

### HMC sub-config (shared fields)

These fields appear in both `SOdAngleSampler::Config` and
`PrincipalAngleSampler::Config` and are set via `cfg.angle_cfg.*`:

| Field | Default | Description |
|---|---|---|
| `num_chains` | `1` | Parallel HMC chains (set equal to `num_samples` for i.i.d. batches) |
| `num_threads` | `1` | OpenMP threads for this sampler's HMC regions |
| `burn_in` | `2000` | Burn-in steps per chain |
| `leapfrog_steps` | `5` | Leapfrog steps per HMC trajectory |
| `init_epsilon` | `1e-4` | Initial step size (dual-averaging adapts it during burn-in) |
| `target_accept` | `0.65` | Dual-averaging acceptance rate target |
| `thinning` | `1` | HMC steps between consecutive samples |
| `seed` | `0` | RNG seed (0 = time-based; chain c receives seed + c) |

---

## Project Structure

```
include/sampler/
    sampler_base.hpp              # Abstract SamplerBase interface
    thread_config.hpp             # Global thread budget (set_num_threads)
    angle_sampler_hmc.hpp         # CRTP HMC base (declaration only)
    principal_angle_sampler.hpp   # Stiefel principal angle density
    so_angle_sampler.hpp          # $SO(d)$ rotation angle density
    stiefel_gaussian_sampler.hpp  # $V(n,k)$ top-level sampler (Algorithm 4.2)
    so_gaussian_sampler.hpp       # $SO(d)$ top-level sampler  (Algorithm 4.3)

src/
    thread_config.cpp
    angle_sampler_hmc.cpp         # CRTP method bodies + explicit instantiations
    principal_angle_sampler.cpp
    so_angle_sampler.cpp
    stiefel_gaussian_sampler.cpp
    so_gaussian_sampler.cpp
```

---

## License

MIT
