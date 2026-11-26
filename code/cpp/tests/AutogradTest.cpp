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

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace ml;
namespace fs = std::filesystem;

namespace {
void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void near(float actual, float expected, float tolerance = 1e-5f) {
    if (!(std::abs(actual - expected) <= tolerance))
        throw std::runtime_error("expected " + std::to_string(expected) +
                                 ", got " + std::to_string(actual));
}

void nearAll(const Values& actual, const Values& expected, float relative, const std::string& what) {
    check(actual.size() == expected.size(), what + ": size differs");
    for (std::size_t i = 0; i < actual.size(); ++i)
        if (!(std::abs(actual[i] - expected[i]) <= relative * (1 + std::abs(expected[i]))))
            throw std::runtime_error(what + ": element " + std::to_string(i) + " expected " +
                                     std::to_string(expected[i]) + ", got " + std::to_string(actual[i]));
}

template<class Function> void rejects(Function run) {
    try { run(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid operation succeeded");
}

// The exception's message must contain `text`, so failures name the shapes or values involved.
template<class Exception, class Function> void rejectsWith(const std::string& text, Function run) {
    try {
        run();
    } catch (const Exception& e) {
        check(std::string(e.what()).find(text) != std::string::npos,
              "message \"" + std::string(e.what()) + "\" lacks \"" + text + "\"");
        return;
    }
    throw std::runtime_error("expected an exception mentioning \"" + text + "\"");
}

Values wave(std::size_t n, float phase) {
    Values v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = std::sin(0.7f*float(i) + phase);
    return v;
}

void sharedGraph(ndp::ForkJoinPool& pool) {
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
    x.zeroGrad();
    near(x.grad()[0], 0);
    x.step(0.1f);
    rejects([&] { loss.backward(); });
    rejects([&] { BasicTensor bad(pool, {1, 2}, {1}); });
}

void broadcastAndReduce(ndp::ForkJoinPool& pool) {
    BasicTensor a(pool, {2,1,2}, {1,2,3,4});
    BasicTensor b(pool, {3,1}, {2,3,4});
    auto z = a*b;
    check(z.shape() == Shape{2,3,2}, "broadcast shape");
    auto sums = z.transpose().sum(-2, true);
    check(sums.shape() == Shape{2,1,2}, "reduction shape");
    check(sums.value() == Values{9,27,18,36}, "transpose or reduction value");
    sums.backward();
    check(a.grad() == Values{9,9,9,9}, "broadcast left gradient");
    check(b.grad() == Values{10,10,10}, "broadcast right gradient");
    rejects([&] { z.sum(3); });
    rejects([&] { a + BasicTensor(pool, {3}, {1,2,3}); });
}

// Golden reference for broadcasting: output values and both operands' gradients, computed
// with independent per-element index arithmetic, for every operator and a range of shape
// pairs, including ones large enough to split across threads.
void broadcastMatchesReference(ndp::ForkJoinPool& pool) {
    const std::vector<std::pair<Shape, Shape>> pairs{
        {{2,1,2}, {3,1}}, {{4,1,3}, {2,1}}, {{5}, {}}, {{}, {2,3}}, {{3,1,4,1}, {1,5,1,2}},
        {{2,3}, {2,3}}, {{1}, {4,1}}, {{64,1,300}, {7,1}}, {{7,1}, {64,1,300}}};
    for (const auto& [as, bs] : pairs) {
        // Reference index map: right-align, a size-1 axis repeats.
        const std::size_t rank = std::max(as.size(), bs.size());
        Shape dims(rank), ad(rank, 1), bd(rank, 1);
        std::copy(as.begin(), as.end(), ad.begin() + std::ptrdiff_t(rank - as.size()));
        std::copy(bs.begin(), bs.end(), bd.begin() + std::ptrdiff_t(rank - bs.size()));
        std::size_t total = 1, aSize = 1, bSize = 1;
        for (std::size_t k = 0; k < rank; ++k) {
            dims[k] = std::max(ad[k], bd[k]);
            total *= dims[k];
            aSize *= ad[k];
            bSize *= bd[k];
        }
        auto index = [&](std::size_t i, const Shape& in) {
            std::size_t result = 0, stride = 1;
            for (std::size_t k = rank; k-- > 0;) {
                const std::size_t coord = i % dims[k];
                i /= dims[k];
                if (in[k] != 1) result += coord * stride;
                stride *= in[k];
            }
            return result;
        };
        Values av = wave(aSize, 0.3f), bv = wave(bSize, 1.1f), weights = wave(total, 2.0f);
        for (float& v : bv) v = 1.5f + v;  // keep divisors away from zero
        for (char op : {'+', '-', '*', '/'}) {
            BasicTensor a(pool, as, av), b(pool, bs, bv);
            BasicTensor out = op == '+' ? a + b : op == '-' ? a - b : op == '*' ? a * b : a / b;
            (out * BasicTensor(pool, dims, weights, Grad::None)).sum().backward();
            Values expected(total), da(aSize, 0), db(bSize, 0);
            for (std::size_t i = 0; i < total; ++i) {
                const std::size_t ia = index(i, ad), ib = index(i, bd);
                const float x = av[ia], y = bv[ib], g = weights[i];
                switch (op) {
                    case '+': expected[i] = x + y; da[ia] += g; db[ib] += g; break;
                    case '-': expected[i] = x - y; da[ia] += g; db[ib] -= g; break;
                    case '*': expected[i] = x * y; da[ia] += g * y; db[ib] += g * x; break;
                    default: expected[i] = x / y; da[ia] += g / y; db[ib] += -g * x / (y * y); break;
                }
            }
            const std::string what = std::string("operator") + op + " " + toString(as) + " " + toString(bs);
            check(out.shape() == dims, what + ": shape");
            nearAll(out.value(), expected, 1e-6f, what + " value");
            nearAll(a.grad(), da, 1e-5f, what + " left gradient");
            nearAll(b.grad(), db, 1e-5f, what + " right gradient");
        }
    }
}

void matmulVectors(ndp::ForkJoinPool& pool) {
    BasicTensor v(pool, {2}, {2,3}), w(pool, {2}, {5,7});
    auto product = v.matmul(w);
    check(product.shape().empty(), "dot shape");
    near(product.value()[0], 31);
    product.backward();
    check(v.grad() == w.value() && w.grad() == v.value(), "dot gradient");
    BasicTensor matrix(pool, {2,3}, {1,2,3,4,5,6});
    auto out = v.matmul(matrix);
    check(out.value() == Values{14,19,24}, "vector-matrix product");
    auto col = matrix.transpose().matmul(w);
    check(col.value() == Values{33,45,57}, "matrix-vector product");
}

// Golden reference for matmul: values and both gradients against naive triple loops, on a
// shape large enough to split across threads.
void matmulMatchesReference(ndp::ForkJoinPool& pool) {
    const std::size_t m = 37, k = 129, n = 53;
    Values av = wave(m*k, 0.2f), bv = wave(k*n, 0.9f), weights = wave(m*n, 1.7f);
    BasicTensor a(pool, {m, k}, av), b(pool, {k, n}, bv);
    auto c = a.matmul(b);
    (c * BasicTensor(pool, {m, n}, weights, Grad::None)).sum().backward();
    Values expected(m*n, 0), da(m*k, 0), db(k*n, 0);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t q = 0; q < k; ++q) {
                expected[i*n + j] += av[i*k + q] * bv[q*n + j];
                da[i*k + q] += weights[i*n + j] * bv[q*n + j];
                db[q*n + j] += av[i*k + q] * weights[i*n + j];
            }
    nearAll(c.value(), expected, 1e-5f, "matmul value");
    nearAll(a.grad(), da, 1e-5f, "matmul left gradient");
    nearAll(b.grad(), db, 1e-5f, "matmul right gradient");
}

// Finite differences of every input, weight and bias in a batched dense layer.
void numericalGradients(ndp::ForkJoinPool& pool) {
    std::vector<Values> data{{1,2,3,0.5f,1.5f,2.5f}, {1,0,-1,0.3f,0.2f,0.1f}, {2.5f,0.7f}};
    auto tensors = [&] {
        return std::vector<BasicTensor>{{pool, {2,3}, data[0]}, {pool, {2,3}, data[1]}, {pool, {2}, data[2]}};
    };
    auto objective = [&](const std::vector<BasicTensor>& p) {
        Layer layer(p[1], p[2], {Activation::ReLU, Activation::Sigmoid});
        auto y = layer(p[0]);
        auto transformed = ((y.exp().log() / (1.f + y)).pow(2)).mean();
        return transformed + y.crossEntropy({0,1});
    };
    auto variables = tensors();
    objective(variables).backward();
    for (std::size_t v = 0; v < data.size(); ++v) {
        for (std::size_t i = 0; i < data[v].size(); ++i) {
            const float original = data[v][i];
            constexpr float delta = 0.001f;
            data[v][i] = original + delta;
            const float plus = objective(tensors()).value()[0];
            data[v][i] = original - delta;
            const float minus = objective(tensors()).value()[0];
            data[v][i] = original;
            near(variables[v].grad()[i], (plus - minus)/(2*delta), 0.0002f);
        }
    }
    BasicTensor logits(pool, {2,2}, {1000,-1000,1000,1000});
    auto ce = logits.crossEntropy({1,0});
    near(ce.value()[0], (2000.f + std::log(2.f))/2, 0.0001f);
    ce.backward();
    check(logits.grad() == Values{0.5f,-0.5f,-0.25f,0.25f}, "stable softmax gradient");
}

// The fused dense node against the unfused composition of independently tested ops,
// for a vector, a small batch, and a batch large enough to split across threads.
void denseMatchesComposition(ndp::ForkJoinPool& pool) {
    const Activation cycle[]{Activation::ReLU, Activation::Sigmoid, Activation::Linear};
    for (Shape xs : {Shape{5}, Shape{3,5}, Shape{67,300}}) {
        const std::size_t in = xs.back(), out = in == 5 ? 4 : 90;
        const std::size_t rows = xs.size() == 2 ? xs[0] : 1;
        std::vector<Activation> acts(out);
        for (std::size_t j = 0; j < out; ++j) acts[j] = cycle[j % 3];
        Values xv = wave(rows*in, 0), wv = wave(out*in, 1), bv = wave(out, 2), cv = wave(rows*out, 3);
        Shape ys = xs.size() == 2 ? Shape{rows, out} : Shape{out};
        auto run = [&](bool fused) {
            BasicTensor x(pool, xs, xv), w(pool, {out, in}, wv), b(pool, {out}, bv);
            auto y = fused ? x.dense(w, b, acts) : (x.matmul(w.transpose()) + b).activate(acts);
            (y * BasicTensor(pool, ys, cv)).sum().backward();  // non-uniform dY
            return std::vector<Values>{y.value(), x.grad(), w.grad(), b.grad()};
        };
        auto fused = run(true), reference = run(false);
        for (std::size_t t = 0; t < fused.size(); ++t)
            nearAll(fused[t], reference[t], 1e-5f, "dense vs composition " + toString(xs));
    }
    BasicTensor x(pool, {2,3}, {1,2,3,4,5,6}), w(pool, {2,3}, {1,1,1,1,1,1}), b(pool, {2}, {0,0});
    rejects([&] { x.dense(w, b, {Activation::ReLU}); });
    rejects([&] { x.dense(w.transpose(), b, {Activation::ReLU, Activation::ReLU}); });
    rejects([&] { w.dense(w, b, {Activation::ReLU, Activation::ReLU}); });
}

// Division by zero and log of a non-positive value throw instead of returning a padded result.
void domainErrors(ndp::ForkJoinPool& pool) {
    BasicTensor x(pool, {3}, {1, 2, 3});
    rejectsWith<std::domain_error>("element 1 of divisor shape {3}",
                                   [&] { x / BasicTensor(pool, {3}, {4, 0, 5}); });
    rejectsWith<std::domain_error>("division by zero", [&] { x / 0.f; });
    rejectsWith<std::domain_error>("element 0 of shape {2}",
                                   [&] { BasicTensor(pool, {2}, {0, 1}).log(); });
    rejectsWith<std::domain_error>("element 1", [&] { BasicTensor(pool, {2}, {1, -1}).log(); });
    rejectsWith<std::domain_error>("element 0", [&] { BasicTensor(pool, {1}, {std::nanf("")}).log(); });
    auto small = x / 1e-30f;  // tiny but nonzero divisors divide exactly
    near(small.value()[0], 1e30f, 1e24f);

    // pow: undefined points and poles throw; so does an infinite slope when a gradient is needed.
    rejectsWith<std::domain_error>("element 1 of shape {2} is -1",
                                   [&] { BasicTensor(pool, {2}, {4, -1}).pow(0.5f); });
    rejectsWith<std::domain_error>("is undefined", [&] { BasicTensor(pool, {1}, {0}).pow(-1); });
    rejectsWith<std::domain_error>("infinite slope", [&] { BasicTensor(pool, {1}, {0}).pow(0.5f); });
    near(BasicTensor(pool, {1}, {0}, Grad::None).pow(0.5f).value()[0], 0);  // value alone is fine
    near(BasicTensor(pool, {1}, {-2}).pow(2).value()[0], 4);               // integer exponent
    rejectsWith<std::overflow_error>("pow: element 0", [&] { BasicTensor(pool, {1}, {10}).pow(50); });
    rejectsWith<std::overflow_error>("exp: element 1 of shape {2} overflows",
                                     [&] { BasicTensor(pool, {2}, {0, 100}).exp(); });
    near(BasicTensor(pool, {1}, {-200}).exp().value()[0], 0);  // underflow to 0 is exact enough
}

// Destroying the last handle to a long chain must not recurse once per node. 300,000
// nodes overflowed an 8 MB stack before Node's destructor unlinked parents iteratively.
void deepGraphTeardown(ndp::ForkJoinPool& pool) {
    BasicTensor x(pool, 1.f);
    {
        BasicTensor y = x;
        for (int i = 0; i < 400'000; ++i) y = y * 1.f;
        check(y.requiresGrad(), "the chain records a graph");
    }
    check(x.value()[0] == 1.f, "the leaf survives its consumers");
}

// Results are new nodes, never aliases of their input: y.grad restarts at 1 on every
// backward pass while the leaf x accumulates.
void noAliasing(ndp::ForkJoinPool& pool) {
    for (int variant = 0; variant < 2; ++variant) {
        BasicTensor x(pool, {3}, {1, 2, 3});
        auto y = variant == 0 ? x.activate(std::vector<Activation>(3, Activation::Linear)) : x.transpose();
        y.backward();
        y.backward();
        check(x.grad() == Values{2, 2, 2}, "leaf gradient accumulates");
        check(y.grad() == Values{1, 1, 1}, variant == 0 ? "activate(Linear) aliased its input"
                                                          : "transpose of a vector aliased its input");
    }
}

// transpose and sum by axis against naive index arithmetic, including the gradients.
void transposeAndSumMatchReference(ndp::ForkJoinPool& pool) {
    const Shape s{3, 1, 4, 5};
    const std::size_t n = 60, rank = s.size();
    Values xv = wave(n, 0.4f), weights = wave(n, 1.3f);
    auto coords = [&](std::size_t i) {
        Shape c(rank);
        for (std::size_t a = rank; a-- > 0;) { c[a] = i % s[a]; i /= s[a]; }
        return c;
    };
    BasicTensor x(pool, s, xv);
    auto t = x.transpose();
    check(t.shape() == Shape{5, 4, 1, 3}, "transpose shape");
    (t * BasicTensor(pool, t.shape(), weights, Grad::None)).sum().backward();
    Values expected(n), dx(n);
    for (std::size_t i = 0; i < n; ++i) {
        const Shape c = coords(i);
        const std::size_t o = ((c[3]*4 + c[2])*1 + c[1])*3 + c[0];  // reversed axes
        expected[o] = xv[i];
        dx[i] = weights[o];
    }
    check(t.value() == expected && x.grad() == dx, "transpose value or gradient");

    for (int axis = 0; axis < int(rank); ++axis) {
        BasicTensor y(pool, s, xv);
        auto r = y.sum(axis);
        Shape rs = s;
        rs.erase(rs.begin() + axis);
        check(r.shape() == rs, "sum shape");
        std::size_t outCount = 1;
        for (auto d : rs) outCount *= d;
        const Values w = wave(outCount, 2.1f);
        (r * BasicTensor(pool, rs, w, Grad::None)).sum().backward();
        std::vector<double> totals(outCount, 0);
        Values dy(n);
        for (std::size_t i = 0; i < n; ++i) {
            Shape c = coords(i);
            c.erase(c.begin() + axis);
            std::size_t o = 0;
            for (std::size_t a = 0; a < rs.size(); ++a) o = o*rs[a] + c[a];
            totals[o] += xv[i];
            dy[i] = w[o];
        }
        Values sums(outCount);
        for (std::size_t o = 0; o < outCount; ++o) sums[o] = static_cast<float>(totals[o]);
        check(r.value() == sums && y.grad() == dy, "sum over axis " + std::to_string(axis));
    }
}

void errorMessagesCarryShapes(ndp::ForkJoinPool& pool) {
    BasicTensor a(pool, {2,3}, wave(6, 0)), b(pool, {4}, wave(4, 0));
    rejectsWith<std::invalid_argument>("{2, 3} x {2, 3}", [&] { a.matmul(a); });
    rejectsWith<std::invalid_argument>("cannot broadcast shapes {2, 3} and {4}", [&] { a + b; });
    rejectsWith<std::invalid_argument>("axis 2 is out of range for shape {2, 3}", [&] { a.sum(2); });
    rejectsWith<std::invalid_argument>("label 3 at row 1", [&] { a.crossEntropy({0, 3}); });
    rejectsWith<std::invalid_argument>("needs 6 values, got 4", [&] { BasicTensor(pool, {2,3}, wave(4, 0)); });
    rejectsWith<std::invalid_argument>("weights {2, 3}, bias {4}",
                                       [&] { Layer(a, b, std::vector<Activation>(2, Activation::ReLU)); });
}

// Grad::None data and NoGradGuard: no gradient buffers, no graph, the same values, and
// parameter gradients unchanged by skipping the input gradient.
void gradientModes(ndp::ForkJoinPool& pool) {
    MLP model(pool, 40, {30, 6});
    const Values xv = wave(9*40, 0.5f);
    auto paramGrads = [&](Grad inputGrad) {
        model.zeroGrad();
        BasicTensor x(pool, {9, 40}, xv, inputGrad);
        model(x).crossEntropy(std::vector<std::size_t>(9, 2)).backward();
        std::vector<Values> grads;
        for (const auto& p : model.parameters()) grads.push_back(p.grad());
        check(x.requiresGrad() == (inputGrad == Grad::Required), "input requiresGrad");
        if (inputGrad == Grad::None) rejects([&] { x.grad(); });
        return grads;
    };
    check(paramGrads(Grad::None) == paramGrads(Grad::Required), "skipping dX changed parameter gradients");

    BasicTensor data(pool, {2}, {1, 2}, Grad::None);
    auto derived = (data * 2.f + 1.f).exp();
    check(!derived.requiresGrad(), "operations on data alone need no gradient");
    rejects([&] { derived.backward(); });
    rejects([&] { data.zeroGrad(); });
    rejects([&] { data.step(0.1f); });

    BasicTensor x(pool, {9, 40}, xv, Grad::None);
    const Values withGraph = model(x).value();
    {
        NoGradGuard guard;
        check(!gradEnabled(), "guard disables gradients");
        auto y = model(x);
        check(!y.requiresGrad() && y.value() == withGraph, "inference values or graph");
        rejects([&] { y.backward(); });
        {
            NoGradGuard nested;
        }
        check(!gradEnabled(), "a nested guard restores the outer state");
    }
    check(gradEnabled(), "guard restores gradients");
}

// backwardSharded against one graph over the whole batch, and its determinism.
std::vector<Values> shardedStep(ndp::ForkJoinPool& pool, std::size_t shards, float* lossOut) {
    MLP model(pool, 69, {32, 10});
    BasicTensor x(pool, {65, 69}, wave(65*69, 0.1f), Grad::None);
    std::vector<std::size_t> labels(65);
    for (std::size_t r = 0; r < labels.size(); ++r) labels[r] = (r * 7) % 10;
    const float loss = model.backwardSharded(x, shards, [&](const BasicTensor& out, std::size_t first,
                                                            std::size_t rows) {
        return out.crossEntropy(std::vector<std::size_t>(labels.begin() + std::ptrdiff_t(first),
                                                         labels.begin() + std::ptrdiff_t(first + rows)));
    });
    if (lossOut) *lossOut = loss;
    std::vector<Values> grads;
    for (const auto& p : model.parameters()) grads.push_back(p.grad());
    return grads;
}

void shardedTraining(ndp::ForkJoinPool& single, ndp::ForkJoinPool& odd, ndp::ForkJoinPool& multi) {
    // Reference: the same model and batch as one graph.
    MLP model(single, 69, {32, 10});
    BasicTensor x(single, {65, 69}, wave(65*69, 0.1f), Grad::None);
    std::vector<std::size_t> labels(65);
    for (std::size_t r = 0; r < labels.size(); ++r) labels[r] = (r * 7) % 10;
    auto loss = model(x).crossEntropy(labels);
    loss.backward();

    float shardedLoss = 0;
    const auto sharded = shardedStep(multi, 8, &shardedLoss);
    near(shardedLoss, loss.value()[0], 1e-5f);
    const auto params = model.parameters();
    for (std::size_t p = 0; p < params.size(); ++p)
        nearAll(sharded[p], params[p].grad(), 1e-5f, "sharded gradient " + std::to_string(p));

    check(shardedStep(single, 8, nullptr) == sharded && shardedStep(odd, 8, nullptr) == sharded,
          "thread count changed sharded gradients");
    check(shardedStep(multi, 1000, nullptr).size() == params.size(), "more shards than rows");

    MLP small(multi, 69, {10});
    BasicTensor wrong(multi, {65, 68}, wave(65*68, 0), Grad::None);
    auto anyLoss = [](const BasicTensor& out, std::size_t, std::size_t) { return out.sum(); };
    rejectsWith<std::invalid_argument>("expected x {batch, 69}; got {65, 68}",
                                       [&] { small.backwardSharded(wrong, 4, anyLoss); });
    rejects([&] { small.backwardSharded(x, 0, anyLoss); });
    rejectsWith<std::invalid_argument>("must be a scalar", [&] {
        small.backwardSharded(BasicTensor(multi, {65, 69}, wave(65*69, 0), Grad::None), 4,
                              [](const BasicTensor& out, std::size_t, std::size_t) { return out; });
    });
    NoGradGuard guard;
    rejects([&] { small.backwardSharded(x, 4, anyLoss); });
}

std::vector<Values> trainingStep(ndp::ForkJoinPool& pool) {
    MLP model(pool, 69, {80,10});
    Values data(65*69);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = std::sin(float(i));
    BasicTensor x(pool, {65,69}, std::move(data));
    auto out = model(x);
    auto loss = out.crossEntropy(std::vector<std::size_t>(65, 1));
    loss.backward();
    std::vector<Values> results{out.value(), x.grad()};
    for (const auto& p : model.parameters()) results.push_back(p.grad());
    model.step(0.01f);
    for (const auto& p : model.parameters()) results.push_back(p.value());
    check(model(x).crossEntropy(std::vector<std::size_t>(65, 1)).value()[0] < loss.value()[0],
          "SGD did not lower loss");
    model.zeroGrad();
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
    near(y.value()[1], 1/(1 + std::exp(-0.5f)));
    model.save(saved);
    check(MLP::load(pool, saved)(x).value() == y.value(), "checkpoint prediction changed");
    {
        std::ifstream file(saved, std::ios::binary);
        unsigned char first[4]{};
        file.read(reinterpret_cast<char*>(first), 4);
        check(first[0] == 2 && first[1] == 0 && first[2] == 0 && first[3] == 0,
              "checkpoint counts are little-endian");
    }
    rejects([&] { model.save(saved); });
    fs::resize_file(saved, fs::file_size(saved) - 1);
    rejectsWith<std::runtime_error>("truncated checkpoint", [&] { MLP::load(pool, saved); });
}
} // namespace

int main() {
    fs::path temp;
    try {
        ndp::ForkJoinPool single(1), odd(3), multi(4);
        sharedGraph(multi);
        broadcastAndReduce(multi);
        broadcastMatchesReference(single);
        broadcastMatchesReference(multi);
        matmulVectors(multi);
        matmulMatchesReference(multi);
        numericalGradients(multi);
        denseMatchesComposition(single);
        denseMatchesComposition(multi);
        domainErrors(multi);
        deepGraphTeardown(single);
        noAliasing(multi);
        transposeAndSumMatchReference(single);
        transposeAndSumMatchReference(multi);
        errorMessagesCarryShapes(multi);
        gradientModes(multi);
        shardedTraining(single, odd, multi);
        const auto serial = trainingStep(single);
        check(serial == trainingStep(odd) && serial == trainingStep(multi),
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
    std::cout << "All autograd tests PASSED.\n";
}
