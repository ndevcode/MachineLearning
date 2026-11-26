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

#include "base/ForkJoinPool.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ml {

using Shape = std::vector<std::size_t>;
using Values = std::vector<float>;
enum class Activation { Sigmoid, ReLU, Linear };

// "{2, 3}", for error messages.
std::string toString(const Shape& shape);

// Whether a leaf tensor collects a gradient. Parameters do; input data and targets do not,
// and operations skip the gradient work for operands that need none.
enum class Grad { Required, None };

// True unless a NoGradGuard is alive on this thread.
bool gradEnabled();

// While alive, operations on this thread record no graph: their results need no gradient,
// keep no parents and allocate no gradient buffer. Use it for inference.
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
    NoGradGuard(NoGradGuard&&) = delete;
    NoGradGuard& operator=(NoGradGuard&&) = delete;

private:
    bool previous_;
};

// A copy shares values, gradients and graph history. The pool must outlive its tensors.
// Invalid shapes, pools, labels and domain errors (division by zero, log of a non-positive
// value) throw with the shapes or values involved.
class BasicTensor {
public:
    BasicTensor(ndp::ForkJoinPool& pool, Shape shape, Values values, Grad grad = Grad::Required);
    BasicTensor(ndp::ForkJoinPool& pool, float scalar, Grad grad = Grad::Required);

    const Shape& shape() const;
    const Values& value() const;
    // Throws std::logic_error unless requiresGrad().
    const Values& grad() const;
    bool requiresGrad() const;
    ndp::ForkJoinPool& pool() const;

    BasicTensor operator+(const BasicTensor& rhs) const;
    BasicTensor operator-(const BasicTensor& rhs) const;
    BasicTensor operator*(const BasicTensor& rhs) const;
    BasicTensor operator/(const BasicTensor& rhs) const;
    BasicTensor operator+(float x) const { return *this + constant(x); }
    BasicTensor operator-(float x) const { return *this - constant(x); }
    BasicTensor operator*(float x) const { return *this * constant(x); }
    BasicTensor operator/(float x) const { return *this / constant(x); }
    friend BasicTensor operator+(float x, const BasicTensor& y) { return y + x; }
    friend BasicTensor operator*(float x, const BasicTensor& y) { return y * x; }
    friend BasicTensor operator-(float x, const BasicTensor& y) { return y.constant(x) - y; }
    friend BasicTensor operator/(float x, const BasicTensor& y) { return y.constant(x) / y; }
    BasicTensor pow(float exponent) const;
    BasicTensor log() const;
    BasicTensor exp() const;
    BasicTensor relu() const;
    BasicTensor sigmoid() const;
    BasicTensor activate(const std::vector<Activation>& columns) const;

    BasicTensor matmul(const BasicTensor& rhs) const;
    // One dense layer as a single graph node: activations[j](x Wᵀ + b) for x {in} or
    // {batch, in}, weights {out, in}, bias {out}. Forward is one parallel pass and so is
    // backward; within each pass every thread writes only the output indexes it owns.
    BasicTensor dense(const BasicTensor& weights, const BasicTensor& bias,
                      const std::vector<Activation>& activations) const;
    BasicTensor transpose() const;
    BasicTensor sum(std::optional<int> axis = {}, bool keepDims = false) const;
    BasicTensor mean() const;
    BasicTensor crossEntropy(const std::vector<std::size_t>& labels) const;

    void backward() const;
    void zeroGrad();
    void step(float learningRate);
    // Adds delta, one value per element, to this tensor's gradient. Combines gradients
    // computed in separate graphs, as MLP::backwardSharded does.
    void accumulateGrad(const Values& delta);

private:
    struct Node;
    using Ptr = std::shared_ptr<Node>;
    using Backward = std::function<void(const Node&)>;
    BasicTensor constant(float x) const { return BasicTensor(pool(), x, Grad::None); }
    // Records parents and backward only when gradients are enabled and a parent needs one.
    BasicTensor result(Shape shape, Values values, std::vector<Ptr> parents,
                       Backward backward) const;
    enum class Binary { Add, Subtract, Multiply, Divide };
    BasicTensor binary(const BasicTensor& rhs, Binary op) const;
    template<class Function, class Derivative>
    BasicTensor unary(Function f, Derivative derivative) const;

    Ptr node_;
};

} // namespace ml
