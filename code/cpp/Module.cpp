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

#include "Module.h"
#include "Parallel.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <system_error>

#if defined(__linux__)
#include <cerrno>
#include <cstdio>   // renameat2, RENAME_NOREPLACE (glibc 2.28+, under _GNU_SOURCE)
#include <fcntl.h>  // AT_FDCWD
#endif

namespace ml {
namespace {

// Checkpoints are little-endian: the byte order of the x86 machines that wrote the legacy
// files, now stated rather than inherited from the host. Big-endian hosts swap.
static_assert(std::endian::native == std::endian::little || std::endian::native == std::endian::big,
              "checkpoint I/O assumes a little- or big-endian host");
template<class T> T littleEndian(T value) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 4, "checkpoints hold bytes and 32-bit values");
    if constexpr (std::endian::native == std::endian::little || sizeof(T) == 1) {
        return value;
    } else {
        auto bits = std::bit_cast<std::uint32_t>(value);
        bits = (bits >> 24) | ((bits >> 8) & 0xff00u) | ((bits << 8) & 0xff0000u) | (bits << 24);
        return std::bit_cast<T>(bits);
    }
}

Values initialize(std::size_t in, std::size_t out, std::mt19937& rng) {
    if (!in || !out || out > Values().max_size() / in)
        throw std::invalid_argument("Layer: cannot allocate " + std::to_string(out) + " x " +
                                    std::to_string(in) + " weights");
    std::normal_distribution<float> normal(0, std::sqrt(2.f / static_cast<float>(in)));
    Values weights(in*out);
    for (float& w : weights) w = normal(rng);
    return weights;
}

// Read counts against the remaining bytes before allocating from file metadata.
class Reader {
public:
    explicit Reader(const std::filesystem::path& path)
        : path_(path), file_(path, std::ios::binary | std::ios::ate) {
        if (!file_) throw std::runtime_error("cannot open checkpoint: " + path.string());
        remaining_ = static_cast<std::uintmax_t>(file_.tellg());
        file_.seekg(0);
    }

    template<class T> std::vector<T> read(std::size_t n) {
        if (n > remaining_ / sizeof(T))
            throw std::runtime_error("truncated checkpoint " + path_.string() + ": " +
                                     std::to_string(n*sizeof(T)) + " bytes needed, " +
                                     std::to_string(remaining_) + " left");
        std::vector<T> v(n);
        const auto bytes = n*sizeof(T);
        if (!file_.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes)))
            throw std::runtime_error("checkpoint read error: " + path_.string());
        remaining_ -= bytes;
        for (T& x : v) x = littleEndian(x);  // no-op on little-endian hosts
        return v;
    }
    std::uint32_t count() { return read<std::uint32_t>(1)[0]; }

private:
    std::filesystem::path path_;
    std::ifstream file_;
    std::uintmax_t remaining_ = 0;
};

std::uint32_t fileCount(std::uint64_t size) {
    if (size > std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("checkpoint count " + std::to_string(size) + " exceeds 32 bits");
    return static_cast<std::uint32_t>(size);
}

template<class T> void write(std::ostream& stream, const T* data, std::size_t size) {
    if constexpr (std::endian::native == std::endian::little || sizeof(T) == 1) {
        stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size*sizeof(T)));
    } else {
        std::vector<T> swapped(data, data + size);
        for (T& x : swapped) x = littleEndian(x);
        stream.write(reinterpret_cast<const char*>(swapped.data()),
                     static_cast<std::streamsize>(size*sizeof(T)));
    }
}

// Errors from link() on filesystems that have no hard links (FAT, exFAT, some network
// shares) rather than errors about these particular paths.
bool linksUnsupported(const std::error_code& error) {
    return error == std::errc::operation_not_permitted || error == std::errc::operation_not_supported ||
           error == std::errc::function_not_supported;
}

// Publishes `from` at `to` only if nothing exists there, atomically. Where neither the
// filesystem nor the platform can promise that, throws rather than risk replacing another
// trainer's checkpoint.
void renameNoReplace(const std::filesystem::path& from, const std::filesystem::path& to) {
    namespace fs = std::filesystem;
#if defined(__linux__) && defined(RENAME_NOREPLACE)
    if (::renameat2(AT_FDCWD, from.c_str(), AT_FDCWD, to.c_str(), RENAME_NOREPLACE) == 0) return;
    const std::error_code error(errno, std::generic_category());
    if (error == std::errc::invalid_argument || error == std::errc::function_not_supported)
        throw fs::filesystem_error("MLP::save: this filesystem has neither hard links nor a no-replace "
                                   "rename, so the checkpoint cannot be published without risking "
                                   "replacing an existing file", from, to, error);
    throw fs::filesystem_error("MLP::save: cannot publish checkpoint", from, to, error);
#else
    throw fs::filesystem_error("MLP::save: this filesystem has no hard links and this platform has no "
                               "atomic no-replace rename", from, to,
                               std::make_error_code(std::errc::operation_not_supported));
#endif
}

// The parameters step() and zeroGrad() act on: frozen (Grad::None) ones are skipped, the
// rest must be leaves. All are checked before the caller changes any, so a throw leaves
// every parameter as it was.
std::vector<BasicTensor> trainable(const Module& module, const char* op) {
    std::vector<BasicTensor> params = module.parameters();
    for (std::size_t i = 0; i < params.size(); ++i)
        if (params[i].requiresGrad() && !params[i].isLeaf())
            throw std::logic_error(std::string(op) + ": parameter " + std::to_string(i) + " of shape " +
                                   toString(params[i].shape()) + " is the result of an operation, "
                                   "not a leaf");
    std::erase_if(params, [](const BasicTensor& p) { return !p.requiresGrad(); });
    return params;
}

} // namespace

void Module::zeroGrad() { for (auto p : trainable(*this, "zeroGrad")) p.zeroGrad(); }

void Module::step(float learningRate) {
    if (!std::isfinite(learningRate) || learningRate < 0)
        throw std::invalid_argument("step: learning rate must be finite and non-negative, got " +
                                    std::to_string(learningRate));
    for (auto p : trainable(*this, "step")) p.step(learningRate);
}

Layer::Layer(ndp::ForkJoinPool& pool, std::size_t nin, std::size_t nout,
             Activation activation, std::mt19937& rng)
    : weights_(pool, {nout, nin}, initialize(nin, nout, rng)), bias_(pool, {nout}, Values(nout)),
      activations_(nout, activation) {}

Layer::Layer(BasicTensor weights, BasicTensor bias, std::vector<Activation> activations)
    : weights_(std::move(weights)), bias_(std::move(bias)), activations_(std::move(activations)) {
    const Shape& ws = weights_.shape();
    if (ws.size() != 2 || bias_.shape() != Shape{ws[0]} || activations_.size() != ws[0])
        throw std::invalid_argument("Layer: expected weights {out, in}, bias {out} and out "
                                    "activations; got weights " + toString(ws) + ", bias " +
                                    toString(bias_.shape()) + ", " +
                                    std::to_string(activations_.size()) + " activations");
    if (&weights_.pool() != &bias_.pool())
        throw std::invalid_argument("Layer: weights and bias belong to different pools");
    for (const BasicTensor* p : {&weights_, &bias_})
        if (p->requiresGrad() && !p->isLeaf())
            throw std::invalid_argument("Layer: trainable " + std::string(p == &weights_ ? "weights" : "bias") +
                                        " of shape " + toString(p->shape()) + " must be a leaf tensor, "
                                        "not the result of an operation");
}

BasicTensor Layer::forward(const BasicTensor& x) const {
    // Same math as (x.matmul(weights_.transpose()) + bias_).activate(activations_), without
    // materializing Wᵀ and with one barrier per layer instead of four. AutogradTest checks
    // the two agree.
    return x.dense(weights_, bias_, activations_);
}

MLP::MLP(ndp::ForkJoinPool& pool, std::size_t nin, const Shape& widths,
         Activation output, std::uint32_t seed) {
    if (widths.empty()) throw std::invalid_argument("MLP: widths must name at least the output layer");
    std::mt19937 rng(seed);
    for (std::size_t i = 0; i < widths.size(); ++i) {
        const auto activation = i + 1 == widths.size() ? output : Activation::ReLU;
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

MLP MLP::replicate() const {
    std::vector<Layer> copies;
    for (const Layer& layer : layers_) {
        const BasicTensor& w = layer.weights();
        const BasicTensor& b = layer.bias();
        auto grad = [](const BasicTensor& p) { return p.requiresGrad() ? Grad::Required : Grad::None; };
        copies.emplace_back(BasicTensor(w.pool(), w.shape(), w.value(), grad(w)),
                            BasicTensor(b.pool(), b.shape(), b.value(), grad(b)), layer.activations());
    }
    return MLP(std::move(copies));
}

float MLP::backwardSharded(const BasicTensor& x, std::size_t shards, const ShardLoss& loss) {
    const std::size_t inputs = layers_.front().weights().shape()[1];
    if (!gradEnabled())
        throw std::logic_error("backwardSharded: gradients are disabled by a NoGradGuard on this thread");
    if (x.shape().size() != 2 || x.shape()[1] != inputs)
        throw std::invalid_argument("backwardSharded: expected x {batch, " + std::to_string(inputs) +
                                    "}; got " + toString(x.shape()));
    if (shards == 0 || !loss)
        throw std::invalid_argument("backwardSharded: needs a positive shard count and a loss");
    // A leaf x that needs a gradient gets one, as backward() would give it. A non-leaf x
    // cannot: its gradient would have to continue into the graph that built it, and the
    // shards' graphs start at copies of x.
    const bool inputGrad = x.requiresGrad();
    if (inputGrad && !x.isLeaf())
        throw std::logic_error("backwardSharded: x of shape " + toString(x.shape()) + " is the result "
                               "of an operation; gradients cannot flow back past it. Pass a leaf "
                               "(Grad::None for data), or build one graph and call backward()");
    auto trainableOnly = [](std::vector<BasicTensor> ps) {
        std::erase_if(ps, [](const BasicTensor& p) { return !p.requiresGrad(); });
        return ps;
    };
    auto params = trainableOnly(parameters());  // frozen parameters get no gradient
    if (params.empty() && !inputGrad)
        throw std::logic_error("backwardSharded: neither x nor any parameter requires a gradient");
    const std::size_t rows = x.shape()[0];
    shards = std::min(shards, rows);
    ndp::ForkJoinPool& pool = x.pool();
    std::vector<std::vector<Values>> shardGrads(shards);  // [shard][trainable parameter]
    std::vector<double> shardLosses(shards);
    Values inputGrads(inputGrad ? rows*inputs : 0);       // shard s owns its rows

    // Step 1: one task per thread, each running shards t, t + tasks, ... on its own replica,
    // so tasks share nothing they write. Operations inside a shard use whatever workers are
    // still idle (none, once every thread has a shard); either way a shard's arithmetic
    // does not depend on which thread runs it.
    //
    // Every shard's gradients are kept until Step 2 (shards x parameters floats). Summing
    // per task instead would save memory, but the grouping, and so the rounding, would then
    // depend on the thread count.
    const auto tasks = static_cast<unsigned>(std::min<std::size_t>(pool.size(), shards));
    detail::runTasks(pool, tasks, [&](unsigned task) {
        MLP replica = replicate();
        const auto replicaParams = trainableOnly(replica.parameters());
        for (std::size_t s = task; s < shards; s += tasks) {
            const auto [first, end] =
                detail::chunk(rows, static_cast<unsigned>(s), static_cast<unsigned>(shards));
            const std::size_t count = end - first;
            const auto begin = x.value().begin() + std::ptrdiff_t(first*inputs);
            BasicTensor slice(pool, {count, inputs}, Values(begin, begin + std::ptrdiff_t(count*inputs)),
                              inputGrad ? Grad::Required : Grad::None);
            replica.zeroGrad();
            const BasicTensor shardLoss = loss(replica(slice), first, count);
            if (!shardLoss.shape().empty())
                throw std::invalid_argument("backwardSharded: the loss must be a scalar; got shape " +
                                            toString(shardLoss.shape()));
            // Weighted by the shard's share of the batch, the shard losses sum to the batch mean.
            const BasicTensor weighted = shardLoss * (static_cast<float>(count) / static_cast<float>(rows));
            weighted.backward();
            shardLosses[s] = weighted.value()[0];
            for (const auto& p : replicaParams) shardGrads[s].push_back(p.grad());
            if (inputGrad)
                std::copy(slice.grad().begin(), slice.grad().end(),
                          inputGrads.begin() + std::ptrdiff_t(first*inputs));
        }
    });

    // Step 2: nothing is accumulated until every shard has succeeded. Each parameter element
    // sums its shard gradients in shard order, one owner per element; each input row came
    // from exactly one shard.
    if (inputGrad) BasicTensor(x).accumulateGrad(inputGrads);
    for (std::size_t p = 0; p < params.size(); ++p) {
        Values total(params[p].value().size());
        detail::parallelFor(pool, total.size(), shards, [&](std::size_t i) {
            float sum = 0;
            for (std::size_t s = 0; s < shards; ++s) sum += shardGrads[s][p][i];
            total[i] = sum;
        });
        params[p].accumulateGrad(total);
    }
    return static_cast<float>(std::accumulate(shardLosses.begin(), shardLosses.end(), 0.0));
}

void MLP::save(const std::filesystem::path& path) const {
    // Stage on the same filesystem. Linking publishes only a complete file and fails
    // atomically if another trainer already created the destination (or a dangling
    // symlink sits there).
    namespace fs = std::filesystem;
    fs::path staging;
    for (unsigned attempt = 0; ; ++attempt) {
        staging = path.string() + ".tmp." + std::to_string(attempt);
        if (fs::create_directory(staging)) break;
    }
    try {
        const auto staged = staging / "model";
        std::ofstream file;
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.open(staged, std::ios::binary);
        auto writeCount = [&](std::uint64_t n) {
            const auto count = fileCount(n);
            write(file, &count, 1);
        };
        writeCount(layers_.size() + 1);
        writeCount(layers_.size());
        const auto inputs = layers_.front().weights().shape()[1];
        writeCount(inputs);
        for (const auto& layer : layers_) writeCount(layer.bias().value().size());
        std::vector<std::uint8_t> codes(inputs, 0);  // Input neurons pass through.
        for (const auto& layer : layers_)
            for (auto a : layer.activations()) codes.push_back(static_cast<std::uint8_t>(a));
        write(file, codes.data(), codes.size());
        for (const auto& layer : layers_) {
            const auto width = layer.weights().shape()[1];
            const auto& weights = layer.weights().value();
            const auto& bias = layer.bias().value();
            writeCount(weights.size() + bias.size());
            for (std::size_t i = 0; i < bias.size(); ++i) {
                write(file, weights.data() + i*width, width);
                write(file, &bias[i], 1);
            }
        }
        file.close();
        std::error_code linkError;
        fs::create_hard_link(staged, path, linkError);
        if (linkError && !linksUnsupported(linkError))
            throw fs::filesystem_error("MLP::save: cannot publish checkpoint", staged, path, linkError);
        // No hard links here. A plain rename() would replace a checkpoint that another
        // trainer published, so only an atomic no-replace rename will do.
        if (linkError) renameNoReplace(staged, path);
    } catch (...) {
        std::error_code ignored;
        fs::remove_all(staging, ignored);
        throw;
    }
    fs::remove_all(staging);
}

MLP MLP::load(ndp::ForkJoinPool& pool, const std::filesystem::path& path) {
    Reader reader(path);
    const auto count = reader.count(), connections = reader.count();
    if (count < 2 || connections != count - 1)
        throw std::runtime_error("checkpoint " + path.string() + ": " + std::to_string(count) +
                                 " layers and " + std::to_string(connections) +
                                 " connections is not a chain");
    const auto widths = reader.read<std::uint32_t>(count);
    std::vector<std::vector<Activation>> activations;
    for (std::size_t layer = 0; layer < widths.size(); ++layer) {
        if (!widths[layer])
            throw std::runtime_error("checkpoint " + path.string() + ": layer " +
                                     std::to_string(layer) + " is empty");
        const auto codes = reader.read<std::uint8_t>(widths[layer]);
        std::vector<Activation> row;
        for (auto code : codes) {
            if (code > static_cast<unsigned>(Activation::Linear))
                throw std::runtime_error("checkpoint " + path.string() + ": unknown activation code " +
                                         std::to_string(code) + " in layer " + std::to_string(layer));
            row.push_back(static_cast<Activation>(code));
        }
        activations.push_back(std::move(row));
    }
    std::vector<Layer> layers;
    for (std::size_t i = 0; i < connections; ++i) {
        const std::size_t in = widths[i], out = widths[i + 1];
        const auto n = reader.count();
        if (n != (std::uint64_t(in) + 1)*out)
            throw std::runtime_error("checkpoint " + path.string() + ": connection " + std::to_string(i) +
                                     " holds " + std::to_string(n) + " values; " + std::to_string(out) +
                                     " x (" + std::to_string(in) + " + 1) expected");
        const auto packed = reader.read<float>(n);
        Values weights(in*out), bias(out);
        for (std::size_t j = 0; j < out; ++j) {
            const auto start = packed.begin() + std::ptrdiff_t(j*(in + 1));
            std::copy_n(start, in, weights.begin() + std::ptrdiff_t(j*in));
            bias[j] = start[std::ptrdiff_t(in)];
        }
        layers.emplace_back(BasicTensor(pool, {out, in}, std::move(weights)),
                            BasicTensor(pool, {out}, std::move(bias)), std::move(activations[i + 1]));
    }
    return MLP(std::move(layers));
}

} // namespace ml
