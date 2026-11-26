#pragma once

#include "BasicTensor.h"
#include <cstdint>
#include <filesystem>
#include <random>

namespace ml {

class Module {
public:
    virtual ~Module() = default;
    virtual BasicTensor forward(const BasicTensor& x) const = 0;
    BasicTensor operator()(const BasicTensor& x) const { return forward(x); }
    virtual std::vector<BasicTensor> parameters() const = 0;
    void zero_grad();
    void step(float learning_rate);
};

class Layer final : public Module {
public:
    Layer(ndp::ForkJoinPool& pool, std::size_t nin, std::size_t nout,
          Activation activation, std::mt19937& rng);
    Layer(BasicTensor weights, BasicTensor bias, std::vector<Activation> activations);

    BasicTensor forward(const BasicTensor& x) const override;
    std::vector<BasicTensor> parameters() const override { return {W, b}; }
    const BasicTensor& weights() const { return W; }
    const BasicTensor& bias() const { return b; }
    const std::vector<Activation>& activations() const { return activations_; }

private:
    BasicTensor W, b;
    std::vector<Activation> activations_;
};

class MLP final : public Module {
public:
    // He-initialized dense layers, ReLU hidden activations and configurable output.
    MLP(ndp::ForkJoinPool& pool, std::size_t nin, const Shape& widths,
        Activation output = Activation::Linear, std::uint32_t seed = 42);
    BasicTensor forward(const BasicTensor& x) const override;
    std::vector<BasicTensor> parameters() const override;
    const std::vector<Layer>& layers() const { return layers_; }
    // Publishes a new checkpoint. Existing files are preserved.
    void save(const std::filesystem::path& path) const;
    static MLP load(ndp::ForkJoinPool& pool, const std::filesystem::path& path);

private:
    explicit MLP(std::vector<Layer> layers) : layers_(std::move(layers)) {}
    std::vector<Layer> layers_;
};

} // namespace ml
