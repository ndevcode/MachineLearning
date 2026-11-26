#include "Module.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
namespace fs = std::filesystem;
using ml::BasicTensor;
using ml::MLP;
using ml::Values;

fs::path idx_path(const fs::path& dir, const std::string& prefix, const std::string& suffix) {
    auto path = dir / (prefix + "." + suffix);
    return fs::exists(path) ? path : dir / (prefix + "-" + suffix);
}

std::uint32_t read_u32(std::istream& file) {
    std::array<unsigned char, 4> bytes{};
    if (!file.read(reinterpret_cast<char*>(bytes.data()), 4)) throw std::runtime_error("truncated IDX header");
    std::uint32_t value = 0;
    for (auto b : bytes) value = (value << 8) | b;
    return value;
}

std::vector<std::uint8_t> payload(std::ifstream& file, const fs::path& path, std::uintmax_t count) {
    if (fs::file_size(path) - static_cast<std::uintmax_t>(file.tellg()) != count ||
        count > std::vector<std::uint8_t>().max_size())
        throw std::runtime_error("incorrect IDX size: " + path.string());
    std::vector<std::uint8_t> bytes(count);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
        throw std::runtime_error("truncated IDX payload: " + path.string());
    return bytes;
}

struct Dataset {
    std::size_t height, width;
    std::vector<std::uint8_t> pixels, labels;

    static Dataset load(const fs::path& dir, const std::string& split) {
        auto images = idx_path(dir, split + "-images", "idx3-ubyte");
        auto labels = idx_path(dir, split + "-labels", "idx1-ubyte");
        std::ifstream img(images, std::ios::binary), lab(labels, std::ios::binary);
        if (!img || !lab) throw std::runtime_error("missing " + split + " IDX files in " + dir.string());
        if (read_u32(img) != 2051 || read_u32(lab) != 2049) throw std::runtime_error("bad IDX magic");
        auto n = read_u32(img), num_labels = read_u32(lab);
        auto h = read_u32(img), w = read_u32(img);
        std::uint64_t area = std::uint64_t(h)*w;
        if (!n || n != num_labels || !area || area > Values().max_size()/n)
            throw std::runtime_error("invalid MNIST dimensions or counts");
        Dataset data{h, w, payload(img, images, area*n), payload(lab, labels, n)};
        for (auto y : data.labels) if (y > 9) throw std::runtime_error("MNIST label exceeds 9");
        return data;
    }

    BasicTensor batch(ndp::ThreadPool& pool, std::size_t first, std::size_t n, std::mt19937* rng) const {
        auto area = height*width;
        Values values(n*area);
        std::uniform_int_distribution<int> offset(-1, 1);
        for (std::size_t i = 0; i < n; ++i) {
            auto image = pixels.data() + (first+i)*area;
            auto dest = values.data() + i*area;
            int dx = rng ? offset(*rng) : 0, dy = rng ? offset(*rng) : 0;
            if (!dx && !dy) {
                std::transform(image, image+area, dest, [](auto p) { return p / 255.f; });
                continue;
            }
            for (std::size_t y = 0; y < height; ++y)
                for (std::size_t x = 0; x < width; ++x) {
                    auto sx = static_cast<std::int64_t>(x) - dx, sy = static_cast<std::int64_t>(y) - dy;
                    if (sx >= 0 && sy >= 0 && std::size_t(sx) < width && std::size_t(sy) < height)
                        dest[y*width+x] = image[sy*width+sx] / 255.f;
                }
        }
        return BasicTensor(pool, {n, area}, std::move(values));
    }
};

void train(ndp::ThreadPool& pool, const Dataset& data, const fs::path& checkpoint,
           std::size_t epochs, std::size_t batch_size) {
    MLP model(pool, data.width*data.height, {16, 8, 16, 10}, ml::Activation::Sigmoid);
    std::mt19937 rng(42);
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        double squared_error = 0;
        for (std::size_t i = 0; i < data.labels.size();) {
            auto n = std::min(batch_size, data.labels.size()-i);
            auto x = data.batch(pool, i, n, epoch ? &rng : nullptr);
            Values targets(n*10);
            for (std::size_t j = 0; j < n; ++j) targets[j*10 + data.labels[i+j]] = 1;
            model.zero_grad();
            auto err = model(x) - BasicTensor(pool, {n, 10}, std::move(targets));
            auto sum_squared = (err * err).sum();
            squared_error += sum_squared.value()[0];
            // The starter sums over outputs. Average only over examples for SGD.
            auto loss = sum_squared / static_cast<float>(n);
            loss.backward();
            model.step(0.01f);
            i += n;
        }
        std::cout << "Epoch " << epoch+1 << "  mean squared error "
                  << squared_error / (10.0 * data.labels.size()) << std::endl;
    }
    model.save(checkpoint);
}

void evaluate(ndp::ThreadPool& pool, const Dataset& data, const MLP& model) {
    if (model.layers().front().weights().shape()[1] != data.height*data.width ||
        model.layers().back().bias().value().size() != 10)
        throw std::runtime_error("checkpoint dimensions do not match MNIST");
    std::size_t correct = 0;
    for (std::size_t i = 0; i < data.labels.size();) {
        auto n = std::min<std::size_t>(128, data.labels.size()-i);
        auto predictions = model(data.batch(pool, i, n, nullptr));
        for (std::size_t j = 0; j < n; ++j) {
            auto scores = predictions.value().begin()+10*j;
            correct += std::max_element(scores, scores+10)-scores == data.labels[i+j];
        }
        i += n;
    }
    std::cout << correct << "/" << data.labels.size() << " correct ("
              << std::fixed << std::setprecision(2) << 100.0*correct/data.labels.size() << "%)\n";
}

unsigned positive(std::string_view arg) {
    unsigned n = 0;
    auto parsed = std::from_chars(arg.data(), arg.data()+arg.size(), n);
    if (parsed.ec != std::errc{} || parsed.ptr != arg.data()+arg.size() || !n)
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
        fs::path dir = argc > 1 ? argv[1] : "./assets/mnist";
        fs::path model = argc > 2 ? fs::path(argv[2]) : dir / "improved_network.dat";
        auto epochs = argc > 3 ? positive(argv[3]) : 25;
        auto batch = argc > 4 ? positive(argv[4]) : 1;
        ndp::ThreadPool pool(argc > 5 ? positive(argv[5]) : std::max(1u, std::thread::hardware_concurrency()));
        if (!fs::exists(model)) train(pool, Dataset::load(dir, "train"), model, epochs, batch);
        evaluate(pool, Dataset::load(dir, "t10k"), MLP::load(pool, model));
    } catch (const std::exception& e) {
        std::cerr << "mnist: " << e.what() << '\n';
        return 1;
    }
}
