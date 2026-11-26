#include "BasicTensor.h"
#include "Parallel.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace ml {
namespace {

std::size_t elements(const Shape& shape) {
    std::size_t size = 1;
    for (auto dim : shape) {
        if (!dim || size > Values().max_size() / dim)
            throw std::invalid_argument("invalid tensor dimensions");
        size *= dim;
    }
    return size;
}

// Right-align the input axes. A singleton axis has stride zero when expanded.
struct BroadcastIndex {
    Shape dims, strides;
    std::size_t period = 0;

    BroadcastIndex(const Shape& out, const Shape& in) : dims(out), strides(out.size()) {
        auto offset = out.size() - in.size();
        std::size_t stride = 1;
        bool contiguous = true;
        for (auto i = in.size(); i > 0; --i) {
            auto axis = offset + i - 1;
            if (in[i-1] != out[axis]) contiguous = false;
            if (in[i-1] != 1) strides[axis] = stride;
            stride *= in[i-1];
        }
        if (contiguous || stride == 1) period = stride;
    }
    std::size_t operator()(std::size_t i) const {
        if (period) return i % period;
        std::size_t index = 0;
        for (auto axis = dims.size(); axis > 0; --axis) {
            index += (i % dims[axis-1]) * strides[axis-1];
            i /= dims[axis-1];
        }
        return index;
    }
};

float sigmoid_value(float x) {
    const float e = std::exp(-std::abs(x));
    return x >= 0 ? 1 / (1 + e) : e / (1 + e);
}

float activation_value(Activation a, float x) {
    if (a == Activation::Sigmoid) return sigmoid_value(x);
    return a == Activation::ReLU ? std::max(0.f, x) : x;
}

// Derivative written in terms of the output y, so the pre-activation need not be kept.
// ReLU: y > 0 exactly when x > 0.
float activation_slope(Activation a, float y) {
    if (a == Activation::Sigmoid) return y * (1 - y);
    return a == Activation::ReLU ? float(y > 0) : 1.f;
}

std::string to_string(const Shape& shape) {
    std::string s = "{";
    for (std::size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + "}";
}

// Visits the flat range [begin, end) of a row-major matrix with `width` columns as
// row segments: visit(row, first_column, end_column).
template<class Visit>
void row_segments(std::size_t begin, std::size_t end, std::size_t width, Visit visit) {
    while (begin < end) {
        const std::size_t row = begin / width, first = begin % width;
        const std::size_t last = std::min(width, first + (end - begin));
        visit(row, first, last);
        begin += last - first;
    }
}

} // namespace

struct BasicTensor::Node {
    ndp::ForkJoinPool& pool;
    Shape shape;
    Values value, grad;
    std::vector<Ptr> parents;
    std::vector<std::size_t> parent_versions;
    Backward backward;
    std::size_t version = 0;

    Node(ndp::ForkJoinPool& p, Shape s, Values v)
        : pool(p), shape(std::move(s)), value(std::move(v)), grad(value.size(), 0) {
        if (elements(shape) != value.size()) throw std::invalid_argument("tensor size mismatch");
    }
};

BasicTensor::BasicTensor(ndp::ForkJoinPool& p, Shape s, Values v)
    : node_(std::make_shared<Node>(p, std::move(s), std::move(v))) {}
BasicTensor::BasicTensor(ndp::ForkJoinPool& p, float x) : BasicTensor(p, {}, {x}) {}
const Shape& BasicTensor::shape() const { return node_->shape; }
const Values& BasicTensor::value() const { return node_->value; }
const Values& BasicTensor::grad() const { return node_->grad; }
ndp::ForkJoinPool& BasicTensor::pool() const { return node_->pool; }

BasicTensor BasicTensor::result(Shape shape, Values values, std::vector<Ptr> parents,
                                Backward backward) const {
    BasicTensor out(pool(), std::move(shape), std::move(values));
    for (const auto& p : parents) out.node_->parent_versions.push_back(p->version);
    out.node_->parents = std::move(parents);
    out.node_->backward = std::move(backward);
    return out;
}

BasicTensor BasicTensor::binary(const BasicTensor& rhs, Binary op) const {
    if (&pool() != &rhs.pool()) throw std::invalid_argument("tensor pools differ");
    const auto& a = shape();
    const auto& b = rhs.shape();
    Shape dims(std::max(a.size(), b.size()));
    for (std::size_t i = 0; i < dims.size(); ++i) {
        auto x = i < a.size() ? a[a.size()-1-i] : 1;
        auto y = i < b.size() ? b[b.size()-1-i] : 1;
        if (x != y && x != 1 && y != 1) throw std::invalid_argument("incompatible broadcast");
        dims[dims.size()-1-i] = std::max(x, y);
    }
    BroadcastIndex ai(dims, a), bi(dims, b);
    Values values(elements(dims));
    constexpr float eps = 1e-12f;
    detail::parallel_for(pool(), values.size(), 1, [&](auto i) {
        float x = value()[ai(i)], y = rhs.value()[bi(i)];
        switch (op) {
            case Binary::Add: values[i] = x + y; break;
            case Binary::Subtract: values[i] = x - y; break;
            case Binary::Multiply: values[i] = x * y; break;
            case Binary::Divide: values[i] = x / (y + eps); break;
        }
    });
    return result(std::move(dims), std::move(values), {node_, rhs.node_},
        [ai, bi, op](const Node& out) {
            auto& left = *out.parents[0];
            auto& right = *out.parents[1];
            for (unsigned side = 0; side < 2; ++side) {
                auto& dest = side == 0 ? left : right;
                auto accumulate = [&](std::size_t i) {
                    auto j = ai(i), k = bi(i);
                    float g = out.grad[i];
                    if (op == Binary::Subtract && side == 1) g = -g;
                    if (op == Binary::Multiply) g *= side == 0 ? right.value[k] : left.value[j];
                    if (op == Binary::Divide) {
                        float d = right.value[k] + eps;
                        g = side == 0 ? g / d : -g * left.value[j] / (d * d);
                    }
                    dest.grad[side == 0 ? j : k] += g;
                };
                if (dest.grad.size() == out.grad.size())
                    detail::parallel_for(out.pool, out.grad.size(), 1, accumulate);
                else // Multiple output elements contribute to one input gradient.
                    for (std::size_t i = 0; i < out.grad.size(); ++i) accumulate(i);
            }
        });
}

BasicTensor BasicTensor::operator+(const BasicTensor& r) const { return binary(r, Binary::Add); }
BasicTensor BasicTensor::operator-(const BasicTensor& r) const { return binary(r, Binary::Subtract); }
BasicTensor BasicTensor::operator*(const BasicTensor& r) const { return binary(r, Binary::Multiply); }
BasicTensor BasicTensor::operator/(const BasicTensor& r) const { return binary(r, Binary::Divide); }

template<class Function, class Derivative>
BasicTensor BasicTensor::unary(Function f, Derivative derivative) const {
    Values values(value().size());
    detail::parallel_for(pool(), values.size(), 1, [&](auto i) { values[i] = f(value()[i], i); });
    return result(shape(), std::move(values), {node_}, [derivative](const Node& out) {
        auto& in = *out.parents[0];
        detail::parallel_for(out.pool, out.grad.size(), 1, [&](auto i) {
            in.grad[i] += derivative(in.value[i], out.value[i], i) * out.grad[i];
        });
    });
}

BasicTensor BasicTensor::pow(float p) const {
    return unary([p](float x, auto) { return std::pow(x, p); },
                 [p](float x, float, auto) { return p == 0 ? 0.f : p * std::pow(x, p-1); });
}
BasicTensor BasicTensor::log() const {
    return unary([](float x, auto) { return std::log(x + 1e-12f); },
                 [](float x, float, auto) { return 1 / (x + 1e-12f); });
}
BasicTensor BasicTensor::exp() const {
    return unary([](float x, auto) { return std::exp(x); }, [](float, float y, auto) { return y; });
}
BasicTensor BasicTensor::relu() const {
    return unary([](float x, auto) { return std::max(0.f, x); },
                 [](float x, float, auto) { return float(x > 0); });
}
BasicTensor BasicTensor::sigmoid() const {
    return unary([](float x, auto) { return sigmoid_value(x); },
                 [](float, float y, auto) { return y * (1 - y); });
}

BasicTensor BasicTensor::activate(const std::vector<Activation>& columns) const {
    if (shape().empty() || shape().back() != columns.size())
        throw std::invalid_argument("activation count must match output width");
    if (std::all_of(columns.begin(), columns.end(), [&](auto a) { return a == columns.front(); })) {
        switch (columns.front()) {
            case Activation::Linear: return *this;
            case Activation::ReLU: return relu();
            case Activation::Sigmoid: return sigmoid();
        }
    }
    return unary([&](float x, auto i) { return activation_value(columns[i % columns.size()], x); },
                 [columns](float, float y, auto i) {
                     return activation_slope(columns[i % columns.size()], y);
                 });
}

BasicTensor BasicTensor::matmul(const BasicTensor& rhs) const {
    const auto& a = shape();
    const auto& b = rhs.shape();
    if (&pool() != &rhs.pool() || a.empty() || a.size() > 2 || b.empty() || b.size() > 2 ||
        a.back() != b.front()) throw std::invalid_argument("invalid matmul operands");
    const std::size_t m = a.size() == 2 ? a[0] : 1;
    const std::size_t k = a.back();
    const std::size_t n = b.size() == 2 ? b[1] : 1;
    Shape dims;
    if (a.size() == 2) dims.push_back(m);
    if (b.size() == 2) dims.push_back(n);
    Values values(elements(dims));
    detail::parallel_for(pool(), values.size(), k, [&](auto i) {
        float sum = 0;
        for (std::size_t j = 0; j < k; ++j)
            sum += value()[(i/n)*k+j] * rhs.value()[j*n+i%n];
        values[i] = sum;
    });
    return result(std::move(dims), std::move(values), {node_, rhs.node_}, [m,k,n](const Node& out) {
        auto& left = *out.parents[0];
        auto& right = *out.parents[1];
        // dA = dC B^T, dB = A^T dC. Each dot product owns one gradient element.
        detail::parallel_for(out.pool, m*k, n, [&](auto i) {
            float sum = 0;
            auto row = i / k, inner = i % k;
            for (std::size_t col = 0; col < n; ++col)
                sum += out.grad[row*n+col] * right.value[inner*n+col];
            left.grad[i] += sum;
        });
        detail::parallel_for(out.pool, k*n, m, [&](auto i) {
            float sum = 0;
            auto inner = i / n, col = i % n;
            for (std::size_t row = 0; row < m; ++row)
                sum += left.value[row*k+inner] * out.grad[row*n+col];
            right.grad[i] += sum;
        });
    });
}

BasicTensor BasicTensor::dense(const BasicTensor& weights, const BasicTensor& bias,
                               const std::vector<Activation>& activations) const {
    const Shape& xs = shape();
    const Shape& ws = weights.shape();
    if (&pool() != &weights.pool() || &pool() != &bias.pool())
        throw std::invalid_argument("dense: x, weights and bias must share one pool");
    if (xs.empty() || xs.size() > 2 || ws.size() != 2 || xs.back() != ws[1] ||
        bias.shape() != Shape{ws[0]} || activations.size() != ws[0])
        throw std::invalid_argument("dense: expected x {in} or {batch, in}, weights {out, in}, bias {out} "
            "and out activations; got x " + to_string(xs) + ", weights " + to_string(ws) + ", bias " +
            to_string(bias.shape()) + ", " + std::to_string(activations.size()) + " activations");
    // Backward writes dX and dW in the same pass; that is race-free only if they are different arrays.
    if (node_ == weights.node_ || node_ == bias.node_)
        throw std::invalid_argument("dense: x must not be the weights or bias tensor");

    const std::size_t rows = xs.size() == 2 ? xs[0] : 1, in = ws[1], out = ws[0];
    Shape dims = xs.size() == 2 ? Shape{rows, out} : Shape{out};
    Values y(rows*out);
    const float* X = value().data();
    const float* W = weights.value().data();
    const float* B = bias.value().data();

    // Forward: y[r, j] = act_j(x[r, :] . W[j, :] + b[j]). Both rows are contiguous, and the
    // task that owns flat index r*out + j is the only writer of y[r, j].
    detail::parallel_for(pool(), y.size(), in, [&](std::size_t i) {
        const std::size_t r = i / out, j = i % out;
        const float* xr = X + r*in;
        const float* wj = W + j*in;
        float z = 0;
        for (std::size_t k = 0; k < in; ++k) z += xr[k] * wj[k];
        y[i] = activation_value(activations[j], z + B[j]);
    });

    return result(std::move(dims), std::move(y), {node_, weights.node_, bias.node_},
                  [rows, in, out, acts = activations](const Node& node) {
        Node& x = *node.parents[0];
        Node& w = *node.parents[1];
        Node& b = *node.parents[2];
        // dZ[r, j] = act_j'(y[r, j]) dY[r, j], recomputed where used instead of stored,
        // so the whole backward is one pass with one barrier at the end.
        auto dz = [&](std::size_t r, std::size_t j) {
            const std::size_t i = r*out + j;
            return activation_slope(acts[j], node.value[i]) * node.grad[i];
        };
        // Each task owns one slice of W's out*in elements (dW, plus db[j] for the slice
        // holding column 0 of row j) and one slice of x's rows*in elements (dX). The
        // slices are disjoint, so the three products are written without a lock.
        const unsigned tasks = detail::taskCount(
            node.pool, detail::saturatingProduct(2*rows*out, in), std::max(out, rows)*in);
        detail::runTasks(node.pool, tasks, [&](unsigned t) {
            std::vector<float> acc(in);

            // Step 1: dW[j, k] += sum_r dZ[r, j] x[r, k]; db[j] += sum_r dZ[r, j].
            const auto [w_begin, w_end] = detail::chunk(out*in, t, tasks);
            row_segments(w_begin, w_end, in, [&](std::size_t j, std::size_t k0, std::size_t k1) {
                std::fill(acc.begin(), acc.begin() + (k1 - k0), 0.f);
                float bias_sum = 0;
                for (std::size_t r = 0; r < rows; ++r) {
                    const float g = dz(r, j);
                    const float* xr = x.value.data() + r*in;
                    for (std::size_t k = k0; k < k1; ++k) acc[k - k0] += g * xr[k];
                    bias_sum += g;
                }
                for (std::size_t k = k0; k < k1; ++k) w.grad[j*in + k] += acc[k - k0];
                if (k0 == 0) b.grad[j] += bias_sum;
            });

            // Step 2: dX[r, k] += sum_j dZ[r, j] W[j, k].
            const auto [x_begin, x_end] = detail::chunk(rows*in, t, tasks);
            row_segments(x_begin, x_end, in, [&](std::size_t r, std::size_t k0, std::size_t k1) {
                std::fill(acc.begin(), acc.begin() + (k1 - k0), 0.f);
                for (std::size_t j = 0; j < out; ++j) {
                    const float g = dz(r, j);
                    const float* wj = w.value.data() + j*in;
                    for (std::size_t k = k0; k < k1; ++k) acc[k - k0] += g * wj[k];
                }
                for (std::size_t k = k0; k < k1; ++k) x.grad[r*in + k] += acc[k - k0];
            });
        });
    });
}

BasicTensor BasicTensor::T() const {
    if (shape().size() < 2) return *this;
    Shape dims(shape().rbegin(), shape().rend());
    auto transpose_index = [s = shape()](std::size_t i) {
        if (s.size() == 2) return (i % s[1]) * s[0] + i / s[1];
        std::size_t dest = 0;
        for (auto it = s.rbegin(); it != s.rend(); ++it) {
            dest = dest * *it + i % *it;
            i /= *it;
        }
        return dest;
    };
    Values values(value().size());
    detail::parallel_for(pool(), values.size(), 1, [&](auto i) { values[transpose_index(i)] = value()[i]; });
    return result(std::move(dims), std::move(values), {node_}, [transpose_index](const Node& out) {
        auto& in = *out.parents[0];
        detail::parallel_for(out.pool, in.grad.size(), 1, [&](auto i) {
            in.grad[i] += out.grad[transpose_index(i)];
        });
    });
}

BasicTensor BasicTensor::sum(std::optional<int> axis, bool keepdims) const {
    Shape dims = shape();
    std::size_t width = value().size(), stride = 1;
    if (axis) {
        int rank = static_cast<int>(dims.size());
        int d = *axis < 0 ? rank + *axis : *axis;
        if (d < 0 || d >= rank) throw std::invalid_argument("invalid reduction axis");
        width = dims[d];
        for (std::size_t j = d+1; j < dims.size(); ++j) stride *= dims[j];
        if (keepdims) dims[d] = 1;
        else dims.erase(dims.begin()+d);
    } else {
        dims.assign(keepdims ? shape().size() : 0, 1);
    }
    Values values(elements(dims));
    detail::parallel_for(pool(), values.size(), width, [&](auto i) {
        auto base = (i / stride) * width * stride + i % stride;
        double total = 0;
        for (std::size_t j = 0; j < width; ++j) total += value()[base + j*stride];
        values[i] = static_cast<float>(total);
    });
    return result(std::move(dims), std::move(values), {node_}, [width, stride](const Node& out) {
        auto& in = *out.parents[0];
        detail::parallel_for(out.pool, in.grad.size(), 1, [&](auto i) {
            in.grad[i] += out.grad[(i / stride / width) * stride + i % stride];
        });
    });
}

BasicTensor BasicTensor::mean() const {
    return sum() / static_cast<float>(value().size());
}

BasicTensor BasicTensor::cross_entropy(const std::vector<std::size_t>& labels) const {
    if (shape().size() != 2 || shape()[0] != labels.size())
        throw std::invalid_argument("cross_entropy expects a matrix and one label per row");
    const auto classes = shape()[1];
    for (auto y : labels) if (y >= classes) throw std::invalid_argument("invalid class label");
    Values gradient(value().size());
    std::vector<double> losses(labels.size());
    detail::parallel_for(pool(), labels.size(), classes, [&](auto r) {
        auto row = value().begin() + r*classes;
        const double max_logit = *std::max_element(row, row+classes);
        double denom = 0;
        for (std::size_t j = 0; j < classes; ++j) {
            double e = std::exp(row[j] - max_logit);
            denom += e;
            gradient[r*classes+j] = static_cast<float>(e);
        }
        losses[r] = (max_logit - row[labels[r]]) + std::log(denom);
        for (std::size_t j = 0; j < classes; ++j)
            gradient[r*classes+j] = (gradient[r*classes+j]/denom - (j == labels[r])) / labels.size();
    });
    float loss = std::accumulate(losses.begin(), losses.end(), 0.0) / labels.size();
    return result({}, {loss}, {node_}, [g = std::move(gradient)](const Node& out) {
        auto& in = *out.parents[0];
        detail::parallel_for(out.pool, g.size(), 1, [&](auto i) { in.grad[i] += g[i] * out.grad[0]; });
    });
}

void BasicTensor::backward() const {
    // Visit each shared node once, after all its parents have been discovered.
    std::vector<Node*> order;
    std::unordered_set<Node*> visited{node_.get()};
    std::vector<std::pair<Node*, std::size_t>> stack{{node_.get(), 0}};
    while (!stack.empty()) {
        auto& [node, next] = stack.back();
        if (next == node->parents.size()) {
            order.push_back(node);
            stack.pop_back();
        } else {
            auto i = next++;
            auto* parent = node->parents[i].get();
            if (parent->version != node->parent_versions[i])
                throw std::logic_error("parameter updated since forward pass");
            if (visited.insert(parent).second) stack.emplace_back(parent, 0);
        }
    }
    for (auto* node : order)
        if (!node->parents.empty()) std::fill(node->grad.begin(), node->grad.end(), 0.f);
    if (node_->parents.empty()) {
        for (float& g : node_->grad) g += 1;
        return;
    }
    std::fill(node_->grad.begin(), node_->grad.end(), 1.f);
    for (auto it = order.rbegin(); it != order.rend(); ++it)
        if ((*it)->backward) (*it)->backward(**it);
}

void BasicTensor::zero_grad() { std::fill(node_->grad.begin(), node_->grad.end(), 0.f); }
void BasicTensor::step(float lr) {
    if (!node_->parents.empty()) throw std::logic_error("SGD requires a leaf tensor");
    if (!std::isfinite(lr) || lr < 0) throw std::invalid_argument("invalid learning rate");
    detail::parallel_for(pool(), value().size(), 1, [&](auto i) { node_->value[i] -= lr * grad()[i]; });
    ++node_->version;
}

} // namespace ml
