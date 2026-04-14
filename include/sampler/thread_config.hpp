#pragma once

namespace sampler {

// =============================================================================
// Library-level thread budget (Layer 2)
//
// Call set_num_threads() once at application startup to cap the number of
// OpenMP threads used by all sampler parallel regions.  This lets callers
// share cores with their own thread pools without oversubscription.
//
//   sampler::set_num_threads(4);   // use at most 4 cores for all samplers
//   sampler::set_num_threads(0);   // revert to per-Config value (default)
//
// Per-sampler control (Layer 1):
//   Set AngleSamplerHMC::Config::num_threads to the desired value when
//   constructing a sampler.  The default is 1 (single-threaded, polite).
//   Setting num_chains = N and num_threads = N gives full parallel burn-in.
//
// Resolution order (highest priority first):
//   1. Global override (set_num_threads > 0)
//   2. Per-Config value (Config::num_threads, default 1)
// =============================================================================

// Set a process-wide thread cap for all sampler parallel regions.
// Pass 0 to clear the override and fall back to per-Config values.
void set_num_threads(int n);

// Read the current global override (0 = no override set).
int get_num_threads();

// Resolve the effective thread count for a parallel region.
// If a global override is active (> 0) it wins; otherwise cfg_threads is used.
int effective_num_threads(int cfg_threads);

} // namespace sampler
