#pragma once
#include <cstdint>
#include <limits>

namespace sampler {

    class Xoshiro256PlusPlus {
    public:
        using result_type = uint64_t;

        // Required by C++ URBG concept
        static constexpr result_type min() { return std::numeric_limits<result_type>::min(); }
        static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

        explicit Xoshiro256PlusPlus(uint64_t seed_val = 0) {
            seed(seed_val);
        }

        void seed(uint64_t seed_val) {
            // Use SplitMix64 to initialize the 256-bit state from a 64-bit seed
            uint64_t z = (seed_val += 0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
            s[0] = z ^ (z >> 31);

            z = (seed_val += 0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
            s[1] = z ^ (z >> 31);

            z = (seed_val += 0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
            s[2] = z ^ (z >> 31);

            z = (seed_val += 0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
            s[3] = z ^ (z >> 31);
        }

        // Required by C++ URBG concept
        inline result_type operator()() {
            const uint64_t result = rotl(s[0] + s[3], 23) + s[0];

            const uint64_t t = s[1] << 17;

            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];

            s[2] ^= t;
            s[3] = rotl(s[3], 45);

            return result;
        }

    private:
        uint64_t s[4];

        static inline uint64_t rotl(const uint64_t x, int k) {
            return (x << k) | (x >> (64 - k));
        }
    };

} // namespace sampler