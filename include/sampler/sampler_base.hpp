#pragma once

#include <isomorphism/tensor.hpp>

namespace sampler {

// Abstract base class for all manifold samplers.
// Concrete subclasses implement sample() to return a single draw from their
// respective target distributions.
class SamplerBase {
public:
    virtual ~SamplerBase() = default;
    virtual isomorphism::Tensor sample() = 0;
};

} // namespace sampler
