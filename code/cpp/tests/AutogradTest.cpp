#include "Module.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace ml;
namespace fs = std::filesystem;

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void near(float actual, float expected, float tolerance = 1e-5f) {
    if (!(std::abs(actual - expected) <= tolerance))
        throw std::runtime_error("expected " + std::to_string(expected) +
                                 ", got " + std::to_string(actual));
}

template<class Function> void rejects(Function run) {
    try { run(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid operation succeeded");
}

void shared_graph(ndp::ForkJoinPool& pool) {
    BasicTensor x(pool, 2), y(pool, 3);
    auto product = x*y;
    auto loss = product*product + product + x.pow(0);
    near(loss.value()[0], 43);
    loss.backward();
    near(product.grad()[0], 13);
    near(x.grad()[0], 39);
    near(y.grad()[0], 26);
    loss.backward();
    near(x.grad()[0], 78);
    x.zero_grad();
    near(x.grad()[0], 0);
    x.step(0.1f);
    rejects([&] { loss.backward(); });
    rejects([&] { BasicTensor bad(pool, {1, 2}, {1}); });
}

void broadcast_and_reduce(ndp::ForkJoinPool& pool) {
    BasicTensor a(pool, {2,1,2}, {1,2,3,4});
    BasicTensor b(pool, {3,1}, {2,3,4});
    auto z = a*b;
    check(z.shape() == Shape{2,3,2}, "broadcast shape");
    auto sums = z.T().sum(-2, true);
    check(sums.shape() == Shape{2,1,2}, "reduction shape");
    check(sums.value() == Values{9,27,18,36}, "transpose or reduction value");
    sums.backward();
    check(a.grad() == Values{9,9,9,9}, "broadcast left gradient");
    check(b.grad() == Values{10,10,10}, "broadcast right gradient");
    rejects([&] { z.sum(3); });
    rejects([&] { a + BasicTensor(pool, {3}, {1,2,3}); });
}

void matmul_vectors(ndp::ForkJoinPool& pool) {
    BasicTensor v(pool, {2}, {2,3}), w(pool, {2}, {5,7});
    auto dot = v.matmul(w);
    check(dot.shape().empty(), "dot shape");
    near(dot.value()[0], 31);
    dot.backward();
    check(v.grad() == w.value() && w.grad() == v.value(), "dot gradient");
    BasicTensor matrix(pool, {2,3}, {1,2,3,4,5,6});
    auto out = v.matmul(matrix);
    check(out.value() == Values{14,19,24}, "vector-matrix product");
    auto col = matrix.T().matmul(w);
    check(col.value() == Values{33,45,57}, "matrix-vector product");
}

// Finite differences of every input, weight and bias in a batched dense layer.
void numerical_gradients(ndp::ForkJoinPool& pool) {
    std::vector<Values> data{{1,2,3,0.5f,1.5f,2.5f}, {1,0,-1,0.3f,0.2f,0.1f}, {2.5f,0.7f}};
    auto tensors = [&] {
        return std::vector<BasicTensor>{{pool, {2,3}, data[0]}, {pool, {2,3}, data[1]}, {pool, {2}, data[2]}};
    };
    auto objective = [&](const std::vector<BasicTensor>& p) {
        Layer layer(p[1], p[2], {Activation::ReLU, Activation::Sigmoid});
        auto y = layer(p[0]);
        auto transformed = ((y.exp().log() / (1.f + y)).pow(2)).mean();
        return transformed + y.cross_entropy({0,1});
    };
    auto variables = tensors();
    objective(variables).backward();
    for (std::size_t v = 0; v < data.size(); ++v) {
        for (std::size_t i = 0; i < data[v].size(); ++i) {
            float original = data[v][i];
            constexpr float delta = 0.001f;
            data[v][i] = original + delta;
            float plus = objective(tensors()).value()[0];
            data[v][i] = original - delta;
            float minus = objective(tensors()).value()[0];
            data[v][i] = original;
            near(variables[v].grad()[i], (plus-minus)/(2*delta), 0.0002f);
        }
    }
    BasicTensor logits(pool, {2,2}, {1000,-1000,1000,1000});
    auto ce = logits.cross_entropy({1,0});
    near(ce.value()[0], (2000.f+std::log(2.f))/2, 0.0001f);
    ce.backward();
    check(logits.grad() == Values{0.5f,-0.5f,-0.25f,0.25f}, "stable softmax gradient");
}

// The fused dense node against the unfused composition of independently tested ops,
// for a vector, a small batch, and a batch large enough to split across threads.
void dense_matches_composition(ndp::ForkJoinPool& pool) {
    const Activation cycle[]{Activation::ReLU, Activation::Sigmoid, Activation::Linear};
    for (Shape xs : {Shape{5}, Shape{3,5}, Shape{67,300}}) {
        const std::size_t in = xs.back(), out = in == 5 ? 4 : 90;
        const std::size_t rows = xs.size() == 2 ? xs[0] : 1;
        std::vector<Activation> acts(out);
        for (std::size_t j = 0; j < out; ++j) acts[j] = cycle[j % 3];
        auto fill = [](std::size_t n, float phase) {
            Values v(n);
            for (std::size_t i = 0; i < n; ++i) v[i] = std::sin(0.7f*float(i) + phase);
            return v;
        };
        Values xv = fill(rows*in, 0), wv = fill(out*in, 1), bv = fill(out, 2), cv = fill(rows*out, 3);
        Shape ys = xs.size() == 2 ? Shape{rows, out} : Shape{out};
        auto run = [&](bool fused) {
            BasicTensor x(pool, xs, xv), w(pool, {out, in}, wv), b(pool, {out}, bv);
            auto y = fused ? x.dense(w, b, acts) : (x.matmul(w.T()) + b).activate(acts);
            (y * BasicTensor(pool, ys, cv)).sum().backward();  // non-uniform dY
            return std::vector<Values>{y.value(), x.grad(), w.grad(), b.grad()};
        };
        auto fused = run(true), reference = run(false);
        for (std::size_t t = 0; t < fused.size(); ++t)
            for (std::size_t i = 0; i < fused[t].size(); ++i)
                near(fused[t][i], reference[t][i], 1e-5f * (1 + std::abs(reference[t][i])));
    }
    BasicTensor x(pool, {2,3}, {1,2,3,4,5,6}), w(pool, {2,3}, {1,1,1,1,1,1}), b(pool, {2}, {0,0});
    rejects([&] { x.dense(w, b, {Activation::ReLU}); });
    rejects([&] { x.dense(w.T(), b, {Activation::ReLU, Activation::ReLU}); });
    rejects([&] { w.dense(w, b, {Activation::ReLU, Activation::ReLU}); });
}

std::vector<Values> training_step(ndp::ForkJoinPool& pool) {
    MLP model(pool, 69, {80,10});
    Values data(65*69);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = std::sin(float(i));
    BasicTensor x(pool, {65,69}, std::move(data));
    auto out = model(x);
    auto loss = out.cross_entropy(std::vector<std::size_t>(65, 1));
    loss.backward();
    std::vector<Values> results{out.value(), x.grad()};
    for (const auto& p : model.parameters()) results.push_back(p.grad());
    model.step(0.01f);
    for (const auto& p : model.parameters()) results.push_back(p.value());
    check(model(x).cross_entropy(std::vector<std::size_t>(65, 1)).value()[0] < loss.value()[0],
          "SGD did not lower loss");
    model.zero_grad();
    for (auto p : model.parameters())
        for (float g : p.grad()) near(g, 0);
    return results;
}

void checkpoint(ndp::ForkJoinPool& pool, const fs::path& dir) {
    // Original Network format. Mixed activation types and interleaved biases.
    fs::path fixture = dir/"legacy.dat", saved = dir/"saved.dat";
    {
        std::ofstream file(fixture, std::ios::binary);
        const std::uint32_t header[]{2,1,2,2}, count = 6;
        const std::uint8_t activations[]{0,0,1,0};
        const float packed[]{3,4,1,2,-1,0.5f};
        file.write(reinterpret_cast<const char*>(header), sizeof(header));
        file.write(reinterpret_cast<const char*>(activations), sizeof(activations));
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        file.write(reinterpret_cast<const char*>(packed), sizeof(packed));
    }
    BasicTensor x(pool, {2}, {2,4});
    auto model = MLP::load(pool, fixture);
    auto y = model(x);
    near(y.value()[0], 23);
    near(y.value()[1], 1/(1+std::exp(-0.5f)));
    model.save(saved);
    check(MLP::load(pool, saved)(x).value() == y.value(), "checkpoint prediction changed");
    rejects([&] { model.save(saved); });
    fs::resize_file(saved, fs::file_size(saved)-1);
    rejects([&] { MLP::load(pool, saved); });
}
} // namespace

int main() {
    fs::path temp;
    try {
        ndp::ForkJoinPool single(1), odd(3), multi(4);
        shared_graph(multi);
        broadcast_and_reduce(multi);
        matmul_vectors(multi);
        numerical_gradients(multi);
        dense_matches_composition(single);
        dense_matches_composition(multi);
        const auto serial = training_step(single);
        check(serial == training_step(odd) && serial == training_step(multi),
              "thread count changed training results");
        for (unsigned i = 0; ; ++i) {
            auto candidate = fs::temp_directory_path()/("cpu-autograd-test-" + std::to_string(i));
            if (fs::create_directory(candidate)) { temp = candidate; break; }
        }
        checkpoint(multi, temp);
        fs::remove_all(temp);
    } catch (const std::exception& e) {
        if (!temp.empty()) fs::remove_all(temp);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
