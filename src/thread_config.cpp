#include "sampler/thread_config.hpp"

#include <atomic>
#include <algorithm>

namespace sampler {

namespace {
    // 0 = no override; positive value caps all sampler OMP regions.
    std::atomic<int> g_num_threads{0};
} // namespace

void set_num_threads(int n) {
    g_num_threads.store(std::max(0, n), std::memory_order_relaxed);
}

int get_num_threads() {
    return g_num_threads.load(std::memory_order_relaxed);
}

int effective_num_threads(int cfg_threads) {
    const int global = g_num_threads.load(std::memory_order_relaxed);
    return (global > 0) ? global : cfg_threads;
}

} // namespace sampler
