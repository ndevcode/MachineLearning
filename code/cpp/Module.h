// Copyright 2026 N Dev (ndevcode)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "BasicTensor.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <random>

namespace ml {

class Module {
public:
    virtual ~Module() = default;
    virtual BasicTensor forward(const BasicTensor& x) const = 0;
    BasicTensor operator()(const BasicTensor& x) const { return forward(x); }
    virtual std::vector<BasicTensor> parameters() const = 0;
    void zeroGrad();
    void step(float learningRate);
};

class Layer final : public Module {
public:
    Layer(ndp::ForkJoinPool& pool, std::size_t nin, std::size_t nout,
          Activation activation, std::mt19937& rng);
    Layer(BasicTensor weights, BasicTensor bias, std::vector<Activation> activations);

    BasicTensor forward(const BasicTensor& x) const override;
    std::vector<BasicTensor> parameters() const override { return {weights_, bias_}; }
    const BasicTensor& weights() const { return weights_; }
    const BasicTensor& bias() const { return bias_; }
    const std::vector<Activation>& activations() const { return activations_; }

private:
    BasicTensor weights_, bias_;
    std::vector<Activation> activations_;
};

// The mean per-example loss of one shard of a batch, given the model's output for batch
// rows [first, first + rows). Called concurrently from several threads, so it may only
// read shared data.
using ShardLoss =
    std::function<BasicTensor(const BasicTensor& output, std::size_t first, std::size_t rows)>;

class MLP final : public Module {
public:
    // He-initialized dense layers, ReLU hidden activations and configurable output.
    MLP(ndp::ForkJoinPool& pool, std::size_t nin, const Shape& widths,
        Activation output = Activation::Linear, std::uint32_t seed = 42);
    BasicTensor forward(const BasicTensor& x) const override;
    std::vector<BasicTensor> parameters() const override;
    const std::vector<Layer>& layers() const { return layers_; }

    // Data-parallel forward and backward, for layers too small to split well inside each
    // operation. Splits x {batch, in} into `shards` row blocks (fewer if the batch is
    // smaller). Each block runs its whole forward and backward as one task with its own
    // copy of the parameters, so a step is one parallel call. The block gradients are then
    // added to the parameters' gradients in block order, so results depend on `shards`
    // but not on the thread count. Accumulates like backward(); returns the batch mean loss.
    float backwardSharded(const BasicTensor& x, std::size_t shards, const ShardLoss& loss);

    // Publishes a new checkpoint. Existing files are preserved.
    void save(const std::filesystem::path& path) const;
    static MLP load(ndp::ForkJoinPool& pool, const std::filesystem::path& path);

private:
    explicit MLP(std::vector<Layer> layers) : layers_(std::move(layers)) {}
    // Fresh parameter leaves holding copies of this model's values.
    MLP replicate() const;
    std::vector<Layer> layers_;
};

} // namespace ml
