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

#include <algorithm>
#include <array>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>

namespace {
namespace fs = std::filesystem;
using ml::BasicTensor;
using ml::Grad;
using ml::MLP;
using ml::Values;

fs::path idxPath(const fs::path& dir, const std::string& prefix, const std::string& suffix) {
    auto path = dir / (prefix + "." + suffix);
    return fs::exists(path) ? path : dir / (prefix + "-" + suffix);
}

std::uint32_t readU32(std::istream& file) {
    std::array<unsigned char, 4> bytes{};
    if (!file.read(reinterpret_cast<char*>(bytes.data()), 4)) throw std::runtime_error("truncated IDX header");
    std::uint32_t value = 0;
    for (auto b : bytes) value = (value << 8) | b;
    return value;
}

std::vector<std::uint8_t> payload(std::ifstream& file, const fs::path& path, std::uintmax_t count) {
    const auto available = fs::file_size(path) - static_cast<std::uintmax_t>(file.tellg());
    if (available != count || count > std::vector<std::uint8_t>().max_size())
        throw std::runtime_error("IDX file " + path.string() + " holds " + std::to_string(available) +
                                 " payload bytes; its header promises " + std::to_string(count));
    std::vector<std::uint8_t> bytes(count);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("truncated IDX payload: " + path.string());
    return bytes;
}

struct Dataset {
    std::size_t height, width;
    std::vector<std::uint8_t> pixels, labels;

    static Dataset load(const fs::path& dir, const std::string& split) {
        const auto images = idxPath(dir, split + "-images", "idx3-ubyte");
        const auto labels = idxPath(dir, split + "-labels", "idx1-ubyte");
        std::ifstream img(images, std::ios::binary), lab(labels, std::ios::binary);
        if (!img || !lab) throw std::runtime_error("missing " + split + " IDX files in " + dir.string());
        if (readU32(img) != 2051 || readU32(lab) != 2049)
            throw std::runtime_error("bad IDX magic in " + images.string() + " or " + labels.string());
        const auto n = readU32(img), labelCount = readU32(lab);
        const auto h = readU32(img), w = readU32(img);
        const std::uint64_t area = std::uint64_t(h)*w;
        if (!n || n != labelCount || !area || area > Values().max_size()/n)
            throw std::runtime_error("MNIST " + split + ": " + std::to_string(n) + " images of " +
                                     std::to_string(h) + "x" + std::to_string(w) + " and " +
                                     std::to_string(labelCount) + " labels do not match");
        Dataset data{h, w, payload(img, images, area*n), payload(lab, labels, n)};
        for (std::size_t i = 0; i < data.labels.size(); ++i)
            if (data.labels[i] > 9)
                throw std::runtime_error("MNIST " + split + ": label " + std::to_string(data.labels[i]) +
                                         " at index " + std::to_string(i) + " exceeds 9");
        return data;
    }

    // Images indexes[0..n) scaled to [0, 1], optionally shifted by up to one pixel.
    // Input data needs no gradient.
    BasicTensor batch(ndp::ForkJoinPool& pool, const std::size_t* indexes, std::size_t n,
                      std::mt19937* rng) const {
        const auto area = height*width;
        Values values(n*area);
        std::uniform_int_distribution<int> offset(-1, 1);
        for (std::size_t i = 0; i < n; ++i) {
            const auto image = pixels.data() + indexes[i]*area;
            const auto dest = values.data() + i*area;
            const int dx = rng ? offset(*rng) : 0, dy = rng ? offset(*rng) : 0;
            if (!dx && !dy) {
                std::transform(image, image + area, dest, [](auto p) { return float(p) / 255.f; });
                continue;
            }
            for (std::size_t y = 0; y < height; ++y)
                for (std::size_t x = 0; x < width; ++x) {
                    const auto sx = static_cast<std::int64_t>(x) - dx, sy = static_cast<std::int64_t>(y) - dy;
                    if (sx >= 0 && sy >= 0 && std::size_t(sx) < width && std::size_t(sy) < height)
                        dest[y*width + x] = float(image[std::size_t(sy)*width + std::size_t(sx)]) / 255.f;
                }
        }
        return BasicTensor(pool, {n, area}, std::move(values), Grad::None);
    }
};

void train(ndp::ForkJoinPool& pool, const Dataset& data, const fs::path& checkpoint,
           std::size_t epochs, std::size_t batchSize) {
    MLP model(pool, data.width*data.height, {16, 8, 16, 10}, ml::Activation::Sigmoid);
    std::mt19937 rng(42);
    std::vector<std::size_t> order(data.labels.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);  // seeded: reproducible runs
        double squaredError = 0;
        for (std::size_t i = 0; i < order.size();) {
            const auto n = std::min(batchSize, order.size() - i);
            const auto x = data.batch(pool, order.data() + i, n, epoch ? &rng : nullptr);
            Values targets(n*10);
            for (std::size_t j = 0; j < n; ++j) targets[j*10 + data.labels[order[i + j]]] = 1;
            model.zeroGrad();
            // Operation-level splitting, not MLP::backwardSharded: the 784-wide first layer
            // dominates and splits well, and sharding measured 2x slower here (README).
            const auto err = model(x) - BasicTensor(pool, {n, 10}, std::move(targets), Grad::None);
            const auto sumSquared = (err * err).sum();
            squaredError += sumSquared.value()[0];
            // The starter sums squared error over outputs. Average only over examples for SGD.
            (sumSquared / static_cast<float>(n)).backward();
            model.step(0.01f);
            i += n;
        }
        std::cout << "Epoch " << epoch + 1 << "  mean squared error "
                  << squaredError / (10.0 * double(data.labels.size())) << std::endl;
    }
    model.save(checkpoint);
}

void evaluate(ndp::ForkJoinPool& pool, const Dataset& data, const MLP& model) {
    const auto inputs = model.layers().front().weights().shape()[1];
    const auto outputs = model.layers().back().bias().value().size();
    if (inputs != data.height*data.width || outputs != 10)
        throw std::runtime_error("checkpoint maps " + std::to_string(inputs) + " inputs to " +
                                 std::to_string(outputs) + " outputs; MNIST needs " +
                                 std::to_string(data.height*data.width) + " to 10");
    ml::NoGradGuard inference;  // no graph and no gradient buffers
    std::vector<std::size_t> order(data.labels.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::size_t correct = 0;
    for (std::size_t i = 0; i < data.labels.size();) {
        const auto n = std::min<std::size_t>(128, data.labels.size() - i);
        const auto predictions = model(data.batch(pool, order.data() + i, n, nullptr));
        for (std::size_t j = 0; j < n; ++j) {
            const auto scores = predictions.value().begin() + std::ptrdiff_t(10*j);
            correct += std::max_element(scores, scores + 10) - scores == data.labels[i + j];
        }
        i += n;
    }
    std::cout << correct << "/" << data.labels.size() << " correct ("
              << std::fixed << std::setprecision(2)
              << 100.0*double(correct)/double(data.labels.size()) << "%)\n";
}

unsigned positive(std::string_view arg) {
    unsigned n = 0;
    const auto parsed = std::from_chars(arg.data(), arg.data() + arg.size(), n);
    if (parsed.ec != std::errc{} || parsed.ptr != arg.data() + arg.size() || !n)
        throw std::invalid_argument("expected a positive integer: " + std::string(arg));
    return n;
}
} // namespace

int main(int argc, char** argv) {
    if (argc > 6 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
        std::cout << "Usage: mnist [IDX_DIR [CHECKPOINT [EPOCHS [BATCH_SIZE [THREADS]]]]]\n";
        return argc > 6 ? 1 : 0;
    }
    try {
        const fs::path dir = argc > 1 ? argv[1] : "./assets/mnist";
        const fs::path model = argc > 2 ? fs::path(argv[2]) : dir / "improved_network.dat";
        const auto epochs = argc > 3 ? positive(argv[3]) : 25;
        const auto batch = argc > 4 ? positive(argv[4]) : 1;
        ndp::ForkJoinPool pool(argc > 5 ? positive(argv[5]) : std::max(1u, std::thread::hardware_concurrency()));
        if (fs::exists(model)) {
            // Training arguments would be ignored: refuse rather than evaluate something else.
            if (argc > 3)
                throw std::invalid_argument("checkpoint " + model.string() + " already exists; delete it "
                                            "to retrain with EPOCHS/BATCH_SIZE, or omit them to evaluate it");
            std::cerr << "mnist: evaluating existing checkpoint " << model.string()
                      << " (delete it to retrain)\n";
        } else {
            train(pool, Dataset::load(dir, "train"), model, epochs, batch);
        }
        evaluate(pool, Dataset::load(dir, "t10k"), MLP::load(pool, model));
    } catch (const std::exception& e) {
        std::cerr << "mnist: " << e.what() << '\n';
        return 1;
    }
}
