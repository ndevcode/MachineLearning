#include "Module.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ml {
namespace {

Values initialize(std::size_t in, std::size_t out, std::mt19937& rng) {
    if (!in || !out || out > Values().max_size() / in)
        throw std::invalid_argument("invalid layer dimensions");
    std::normal_distribution<float> normal(0, std::sqrt(2.f / in));
    Values weights(in*out);
    for (float& w : weights) w = normal(rng);
    return weights;
}

// Read counts against the remaining bytes before allocating from file metadata.
class Reader {
public:
    explicit Reader(const std::filesystem::path& path) : file(path, std::ios::binary | std::ios::ate) {
        if (!file) throw std::runtime_error("cannot open checkpoint: " + path.string());
        remaining = static_cast<std::uintmax_t>(file.tellg());
        file.seekg(0);
    }

    template<class T> std::vector<T> read(std::size_t n) {
        if (n > remaining / sizeof(T)) throw std::runtime_error("truncated checkpoint");
        std::vector<T> v(n);
        auto bytes = n*sizeof(T);
        if (!file.read(reinterpret_cast<char*>(v.data()), bytes))
            throw std::runtime_error("checkpoint read error");
        remaining -= bytes;
        return v;
    }
    std::uint32_t count() { return read<std::uint32_t>(1)[0]; }
private:
    std::ifstream file;
    std::uintmax_t remaining;
};

std::uint32_t file_count(std::uint64_t size) {
    if (size > std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("layer too large for checkpoint");
    return static_cast<std::uint32_t>(size);
}

template<class T> void write(std::ostream& stream, const T* data, std::size_t size) {
    stream.write(reinterpret_cast<const char*>(data), size*sizeof(T));
}

} // namespace

void Module::zero_grad() { for (auto p : parameters()) p.zero_grad(); }
void Module::step(float lr) { for (auto p : parameters()) p.step(lr); }

Layer::Layer(ndp::ForkJoinPool& pool, std::size_t nin, std::size_t nout,
             Activation activation, std::mt19937& rng)
    : W(pool, {nout, nin}, initialize(nin, nout, rng)), b(pool, {nout}, Values(nout)),
      activations_(nout, activation) {}

Layer::Layer(BasicTensor weights, BasicTensor bias, std::vector<Activation> activations)
    : W(std::move(weights)), b(std::move(bias)), activations_(std::move(activations)) {
    if (W.shape().size() != 2 || b.shape() != Shape{W.shape()[0]} ||
        activations_.size() != W.shape()[0] || &W.pool() != &b.pool())
        throw std::invalid_argument("invalid layer parameters");
}

BasicTensor Layer::forward(const BasicTensor& x) const {
    // Same math as (x.matmul(W.T()) + b).activate(activations_), without materializing Wᵀ
    // and with one barrier per layer instead of four. AutogradTest checks the two agree.
    return x.dense(W, b, activations_);
}

MLP::MLP(ndp::ForkJoinPool& pool, std::size_t nin, const Shape& widths,
         Activation output, std::uint32_t seed) {
    if (widths.empty()) throw std::invalid_argument("MLP needs an output layer");
    std::mt19937 rng(seed);
    for (std::size_t i = 0; i < widths.size(); ++i) {
        auto activation = i+1 == widths.size() ? output : Activation::ReLU;
        layers_.emplace_back(pool, nin, widths[i], activation, rng);
        nin = widths[i];
    }
}

BasicTensor MLP::forward(const BasicTensor& x) const {
    BasicTensor y = x;
    for (const auto& layer : layers_) y = layer(y);
    return y;
}
std::vector<BasicTensor> MLP::parameters() const {
    std::vector<BasicTensor> params;
    for (const Layer& layer : layers_) {
        params.push_back(layer.weights());
        params.push_back(layer.bias());
    }
    return params;
}

void MLP::save(const std::filesystem::path& path) const {
    // Stage on the same filesystem. Linking publishes only a complete file and
    // fails atomically if another trainer already created the destination.
    namespace fs = std::filesystem;
    fs::path staging;
    for (unsigned attempt = 0; ; ++attempt) {
        staging = path.string() + ".tmp." + std::to_string(attempt);
        if (fs::create_directory(staging)) break;
    }
    try {
        auto staged = staging / "model";
        std::ofstream file;
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.open(staged, std::ios::binary);
        auto write_count = [&](std::uint64_t n) {
            auto count = file_count(n);
            write(file, &count, 1);
        };
        write_count(layers_.size()+1);
        write_count(layers_.size());
        auto inputs = layers_.front().weights().shape()[1];
        write_count(inputs);
        for (const auto& layer : layers_) write_count(layer.bias().value().size());
        std::vector<std::uint8_t> codes(inputs, 0); // Input neurons pass through.
        for (const auto& layer : layers_)
            for (auto a : layer.activations()) codes.push_back(static_cast<std::uint8_t>(a));
        write(file, codes.data(), codes.size());
        for (const auto& layer : layers_) {
            auto width = layer.weights().shape()[1];
            const auto& weights = layer.weights().value();
            const auto& bias = layer.bias().value();
            write_count(weights.size() + bias.size());
            for (std::size_t i = 0; i < bias.size(); ++i) {
                write(file, weights.data()+i*width, width);
                write(file, &bias[i], 1);
            }
        }
        file.close();
        fs::create_hard_link(staged, path);
    } catch (...) {
        std::error_code ignored;
        fs::remove_all(staging, ignored);
        throw;
    }
    fs::remove_all(staging);
}

MLP MLP::load(ndp::ForkJoinPool& pool, const std::filesystem::path& path) {
    Reader reader(path);
    auto count = reader.count(), connections = reader.count();
    if (count < 2 || connections != count-1) throw std::runtime_error("invalid checkpoint topology");
    auto widths = reader.read<std::uint32_t>(count);
    std::vector<std::vector<Activation>> activations;
    for (auto width : widths) {
        if (!width) throw std::runtime_error("empty layer in checkpoint");
        auto codes = reader.read<std::uint8_t>(width);
        std::vector<Activation> row;
        for (auto code : codes) {
            if (code > static_cast<unsigned>(Activation::Linear))
                throw std::runtime_error("unknown checkpoint activation");
            row.push_back(static_cast<Activation>(code));
        }
        activations.push_back(std::move(row));
    }
    std::vector<Layer> layers;
    for (std::size_t i = 0; i < connections; ++i) {
        std::size_t in = widths[i], out = widths[i+1];
        auto n = reader.count();
        if (n != (std::uint64_t(in)+1)*out) throw std::runtime_error("wrong checkpoint weight count");
        auto packed = reader.read<float>(n);
        Values weights(in*out), bias(out);
        for (std::size_t j = 0; j < out; ++j) {
            auto start = packed.begin() + j*(in+1);
            std::copy_n(start, in, weights.begin()+j*in);
            bias[j] = start[in];
        }
        layers.emplace_back(BasicTensor(pool, {out, in}, std::move(weights)),
                            BasicTensor(pool, {out}, std::move(bias)), std::move(activations[i+1]));
    }
    return MLP(std::move(layers));
}

} // namespace ml
