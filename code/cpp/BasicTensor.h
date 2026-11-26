#pragma once

#include "base/ForkJoinPool.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace ml {

using Shape = std::vector<std::size_t>;
using Values = std::vector<float>;
enum class Activation { Sigmoid, ReLU, Linear };

// A copy shares values, gradients and graph history. The pool must outlive its tensors.
class BasicTensor {
public:
    BasicTensor(ndp::ForkJoinPool& pool, Shape shape, Values values);
    BasicTensor(ndp::ForkJoinPool& pool, float scalar);

    const Shape& shape() const;
    const Values& value() const;
    const Values& grad() const;
    ndp::ForkJoinPool& pool() const;

    BasicTensor operator+(const BasicTensor& rhs) const;
    BasicTensor operator-(const BasicTensor& rhs) const;
    BasicTensor operator*(const BasicTensor& rhs) const;
    BasicTensor operator/(const BasicTensor& rhs) const;
    BasicTensor operator+(float x) const { return *this + BasicTensor(pool(), x); }
    BasicTensor operator-(float x) const { return *this - BasicTensor(pool(), x); }
    BasicTensor operator*(float x) const { return *this * BasicTensor(pool(), x); }
    BasicTensor operator/(float x) const { return *this / BasicTensor(pool(), x); }
    friend BasicTensor operator+(float x, const BasicTensor& y) { return y + x; }
    friend BasicTensor operator*(float x, const BasicTensor& y) { return y * x; }
    friend BasicTensor operator-(float x, const BasicTensor& y) { return BasicTensor(y.pool(), x) - y; }
    friend BasicTensor operator/(float x, const BasicTensor& y) { return BasicTensor(y.pool(), x) / y; }
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
    BasicTensor T() const;
    BasicTensor sum(std::optional<int> axis = {}, bool keepdims = false) const;
    BasicTensor mean() const;
    BasicTensor cross_entropy(const std::vector<std::size_t>& labels) const;

    void backward() const;
    void zero_grad();
    void step(float learning_rate);

private:
    struct Node;
    using Ptr = std::shared_ptr<Node>;
    using Backward = std::function<void(const Node&)>;
    BasicTensor result(Shape shape, Values values, std::vector<Ptr> parents,
                       Backward backward) const;
    enum class Binary { Add, Subtract, Multiply, Divide };
    BasicTensor binary(const BasicTensor& rhs, Binary op) const;
    template<class Function, class Derivative>
    BasicTensor unary(Function f, Derivative derivative) const;

    Ptr node_;
};

} // namespace ml
