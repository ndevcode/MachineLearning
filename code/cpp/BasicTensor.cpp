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

#include "BasicTensor.h"
#include "Parallel.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ml {
namespace {

thread_local bool gradEnabledOnThread = true;

std::size_t elementCount(const Shape& shape) {
    std::size_t size = 1;
    for (auto dim : shape) {
        if (!dim || size > Values().max_size() / dim)
            throw std::invalid_argument("tensor shape " + toString(shape) +
                                        " has a zero or overflowing dimension");
        size *= dim;
    }
    return size;
}

/*****************************************************************************************
 * Broadcast: index maps for a binary operation on right-aligned shapes. A size-1 or
 * missing input axis repeats across the output and has input stride 0. Ranges are walked
 * with an odometer, so the divisions happen once per range, never once per element.
 *****************************************************************************************/
class Broadcast {
public:
    Broadcast(const Shape& a, const Shape& b, const char* op) {
        const std::size_t rank = std::max(a.size(), b.size());
        const std::array<const Shape*, 2> inputs{&a, &b};
        dims_.resize(rank);
        outStrides_.resize(rank);
        for (unsigned side : {0u, 1u}) {
            const Shape& in = *inputs[side];
            inDims_[side].assign(rank, 1);
            inStrides_[side].assign(rank, 0);
            std::copy(in.begin(), in.end(), inDims_[side].begin() + std::ptrdiff_t(rank - in.size()));
        }
        for (std::size_t axis = 0; axis < rank; ++axis) {
            const std::size_t x = inDims_[0][axis], y = inDims_[1][axis];
            if (x != y && x != 1 && y != 1)
                throw std::invalid_argument(std::string(op) + ": cannot broadcast shapes " +
                                            toString(a) + " and " + toString(b));
            dims_[axis] = std::max(x, y);
        }
        std::size_t outStride = 1;
        std::array<std::size_t, 2> inStride{1, 1};
        for (std::size_t axis = rank; axis-- > 0;) {
            outStrides_[axis] = outStride;
            outStride *= dims_[axis];
            for (unsigned side : {0u, 1u}) {
                if (inDims_[side][axis] != 1) inStrides_[side][axis] = inStride[side];
                inStride[side] *= inDims_[side][axis];
            }
        }
        size_ = outStride;
        // An input with one element per output element reads index i for output i.
        for (unsigned side : {0u, 1u}) covers_[side] = inStride[side] == size_;
    }

    const Shape& shape() const { return dims_; }
    std::size_t size() const { return size_; }
    bool covers(unsigned side) const { return covers_[side]; }

    // visit(i, ia, ib) for each output index i in [begin, end), in order, with the
    // indexes ia and ib that it reads from inputs a and b.
    template<class Visit> void walk(std::size_t begin, std::size_t end, Visit visit) const {
        if (covers_[0] && covers_[1]) {
            for (std::size_t i = begin; i < end; ++i) visit(i, i, i);
            return;
        }
        const std::size_t rank = dims_.size(), last = rank - 1;  // rank >= 1 here
        std::vector<std::size_t> pos(rank);
        std::size_t rest = begin, ia = 0, ib = 0;
        for (std::size_t axis = rank; axis-- > 0;) {
            pos[axis] = rest % dims_[axis];
            rest /= dims_[axis];
            ia += pos[axis] * inStrides_[0][axis];
            ib += pos[axis] * inStrides_[1][axis];
        }
        const std::size_t stepA = inStrides_[0][last], stepB = inStrides_[1][last];
        for (std::size_t i = begin; i < end;) {
            // Innermost axis in one tight loop, then carry into the outer axes.
            const std::size_t run = std::min(end - i, dims_[last] - pos[last]);
            for (std::size_t t = 0; t < run; ++t) visit(i + t, ia + t*stepA, ib + t*stepB);
            i += run;
            ia += run*stepA;
            ib += run*stepB;
            pos[last] += run;
            for (std::size_t axis = last; axis > 0 && pos[axis] == dims_[axis]; --axis) {
                // Rewind this axis to 0 (its full extent was added) and step the next one out.
                ia = ia - dims_[axis]*inStrides_[0][axis] + inStrides_[0][axis - 1];
                ib = ib - dims_[axis]*inStrides_[1][axis] + inStrides_[1][axis - 1];
                pos[axis] = 0;
                ++pos[axis - 1];
            }
        }
    }

    // visit(i, ia, ib) for every output index i that reads element `index` of input `side`,
    // in increasing i. Lets one task own a repeated input's gradient element.
    template<class Visit> void readersOf(unsigned side, std::size_t index, Visit visit) const {
        std::size_t i = 0, ia = 0, ib = 0, rest = index;
        std::vector<std::size_t> freeAxes;  // axes along which this input repeats
        for (std::size_t axis = dims_.size(); axis-- > 0;) {
            const std::size_t dim = inDims_[side][axis];
            if (dim == 1) {
                if (dims_[axis] > 1) freeAxes.push_back(axis);
                continue;
            }
            const std::size_t coord = rest % dim;
            rest /= dim;
            i += coord * outStrides_[axis];
            ia += coord * inStrides_[0][axis];
            ib += coord * inStrides_[1][axis];
        }
        std::reverse(freeAxes.begin(), freeAxes.end());  // outermost first: increasing i
        std::vector<std::size_t> count(freeAxes.size(), 0);
        for (;;) {
            visit(i, ia, ib);
            for (std::size_t f = freeAxes.size();;) {
                if (f == 0) return;
                const std::size_t axis = freeAxes[--f];
                i += outStrides_[axis];
                ia += inStrides_[0][axis];
                ib += inStrides_[1][axis];
                if (++count[f] < dims_[axis]) break;
                i -= dims_[axis] * outStrides_[axis];
                ia -= dims_[axis] * inStrides_[0][axis];
                ib -= dims_[axis] * inStrides_[1][axis];
                count[f] = 0;
            }
        }
    }

private:
    Shape dims_, outStrides_;
    std::array<Shape, 2> inDims_, inStrides_;  // per input, aligned to the output's axes
    std::size_t size_ = 1;
    std::array<bool, 2> covers_{};
};

// Calls f(std::integral_constant<Op, op>{}) so kernels branch on the operator at compile
// time instead of once per element.
template<class Op, class F> void withOp(Op op, F f) {
    switch (op) {
        case Op::Add: f(std::integral_constant<Op, Op::Add>{}); break;
        case Op::Subtract: f(std::integral_constant<Op, Op::Subtract>{}); break;
        case Op::Multiply: f(std::integral_constant<Op, Op::Multiply>{}); break;
        case Op::Divide: f(std::integral_constant<Op, Op::Divide>{}); break;
    }
}

template<auto Op> float combine(float x, float y) {
    using E = decltype(Op);
    if constexpr (Op == E::Add) return x + y;
    else if constexpr (Op == E::Subtract) return x - y;
    else if constexpr (Op == E::Multiply) return x * y;
    else return x / y;
}

// Dot product with DOT_LANES independent partial sums combined in a fixed tree. GCC
// vectorizes it without -ffast-math: on 64x784x512 it ran ~6x faster than a sequential
// sum at -O2 and -O3 alike (16 lanes halved that at -O2). The result does not depend on
// the thread count.
constexpr std::size_t DOT_LANES = 8;
float dot(const float* a, const float* b, std::size_t n) {
    float lanes[DOT_LANES] = {};
    std::size_t k = 0;
    for (; k + DOT_LANES <= n; k += DOT_LANES)
        for (std::size_t l = 0; l < DOT_LANES; ++l) lanes[l] += a[k + l] * b[k + l];
    float tail = 0;
    for (; k < n; ++k) tail += a[k] * b[k];
    for (std::size_t width = DOT_LANES / 2; width; width /= 2)
        for (std::size_t l = 0; l < width; ++l) lanes[l] += lanes[l + width];
    return lanes[0] + tail;
}

// dest[t] += sum over q = 0..count-1, in order, of coef[q] * rows[q*stride + t], for t < width.
// The shared kernel of matmul forward, matmul dB, dense dW and dense dX.
//
// Each pass streams ROWS_PER_PASS input rows across the full width, so the accumulator is
// loaded and stored once per 4 rows instead of once per row, and the hardware prefetcher
// sees 4 contiguous streams. Every element still sums in q order from zero, so results are
// bitwise identical to a row-at-a-time loop. Measured against two alternatives (i7-13650HX,
// one thread; README): 32-column register blocks walk the rows in strided strips, and lost
// up to 63% at batch 1 in an SSE2 build, where those rows are read from memory once.
// 4 rows per pass beat 8, and beat the plain row-at-a-time loop by 15-40% at batch 64 and up.
constexpr std::size_t ROWS_PER_PASS = 4;
void accumulateRows(const float* coef, std::size_t count, const float* rows, std::size_t stride,
                    std::size_t width, float* dest) {
    thread_local Values accBuffer;
    if (accBuffer.size() < width) accBuffer.resize(width);
    float* acc = accBuffer.data();
    std::fill(acc, acc + width, 0.f);
    std::size_t q = 0;
    for (; q + ROWS_PER_PASS <= count; q += ROWS_PER_PASS) {
        const float* row[ROWS_PER_PASS];
        float c[ROWS_PER_PASS];
        for (std::size_t u = 0; u < ROWS_PER_PASS; ++u) {
            row[u] = rows + (q + u)*stride;
            c[u] = coef[q + u];
        }
        for (std::size_t t = 0; t < width; ++t) {
            float a = acc[t];
            for (std::size_t u = 0; u < ROWS_PER_PASS; ++u) a += c[u] * row[u][t];
            acc[t] = a;
        }
    }
    for (; q < count; ++q) {
        const float c = coef[q];
        const float* row = rows + q*stride;
        for (std::size_t t = 0; t < width; ++t) acc[t] += c * row[t];
    }
    for (std::size_t t = 0; t < width; ++t) dest[t] += acc[t];
}

// Per-thread scratch for coefficient columns, reused across calls instead of allocated.
float* scratch(std::size_t n) {
    thread_local Values buffer;
    if (buffer.size() < n) buffer.resize(n);
    return buffer.data();
}

float sigmoidValue(float x) {
    const float e = std::exp(-std::abs(x));
    return x >= 0 ? 1 / (1 + e) : e / (1 + e);
}

float activationValue(Activation a, float x) {
    if (a == Activation::Sigmoid) return sigmoidValue(x);
    return a == Activation::ReLU ? std::max(0.f, x) : x;
}

// Derivative written in terms of the output y, so the pre-activation need not be kept.
// ReLU: y > 0 exactly when x > 0.
float activationSlope(Activation a, float y) {
    if (a == Activation::Sigmoid) return y * (1 - y);
    return a == Activation::ReLU ? float(y > 0) : 1.f;
}

// Visits the flat range [begin, end) of a row-major matrix with `width` columns as
// row segments: visit(row, firstColumn, endColumn).
template<class Visit>
void rowSegments(std::size_t begin, std::size_t end, std::size_t width, Visit visit) {
    while (begin < end) {
        const std::size_t row = begin / width, first = begin % width;
        const std::size_t last = std::min(width, first + (end - begin));
        visit(row, first, last);
        begin += last - first;
    }
}

// visit(i, j) for each i in [begin, end) of the row-major index space `dims`, in order,
// where j = sum over axes of coordinate * strides[axis]. Like Broadcast::walk, it divides
// once per range and then steps an odometer.
template<class Visit>
void walkStrided(const Shape& dims, const Shape& strides, std::size_t begin, std::size_t end,
                 Visit visit) {
    if (begin >= end) return;
    const std::size_t rank = dims.size(), last = rank - 1;  // callers pass rank >= 1
    std::vector<std::size_t> pos(rank);
    std::size_t rest = begin, j = 0;
    for (std::size_t axis = rank; axis-- > 0;) {
        pos[axis] = rest % dims[axis];
        rest /= dims[axis];
        j += pos[axis] * strides[axis];
    }
    const std::size_t step = strides[last];
    for (std::size_t i = begin; i < end;) {
        const std::size_t run = std::min(end - i, dims[last] - pos[last]);
        for (std::size_t t = 0; t < run; ++t) visit(i + t, j + t*step);
        i += run;
        j += run*step;
        pos[last] += run;
        for (std::size_t axis = last; axis > 0 && pos[axis] == dims[axis]; --axis) {
            j = j - dims[axis]*strides[axis] + strides[axis - 1];
            pos[axis] = 0;
            ++pos[axis - 1];
        }
    }
}

// Overflow throws instead of passing inf on (where a zero gradient times inf becomes NaN).
void requireFinite(const char* op, const Values& result, const Shape& shape) {
    const auto bad = std::find_if(result.begin(), result.end(), [](float y) { return !std::isfinite(y); });
    if (bad != result.end())
        throw std::overflow_error(std::string(op) + ": element " + std::to_string(bad - result.begin()) +
                                  " of shape " + toString(shape) + " overflows float");
}

std::string notDifferentiable(const char* op, const Shape& shape) {
    return std::string(op) + ": tensor of shape " + toString(shape) +
           " does not require a gradient (created with Grad::None, computed only from such "
           "tensors, or built under NoGradGuard)";
}

} // namespace

std::string toString(const Shape& shape) {
    std::string text = "{";
    for (std::size_t i = 0; i < shape.size(); ++i) text += (i ? ", " : "") + std::to_string(shape[i]);
    return text + "}";
}

bool gradEnabled() { return gradEnabledOnThread; }
NoGradGuard::NoGradGuard() : previous_(gradEnabledOnThread) { gradEnabledOnThread = false; }
NoGradGuard::~NoGradGuard() { gradEnabledOnThread = previous_; }

struct BasicTensor::Node {
    ndp::ForkJoinPool& pool;
    Shape shape;
    Values value;
    Values grad;  // allocated only when requiresGrad
    bool requiresGrad;
    std::vector<Ptr> parents;
    std::vector<std::size_t> parentVersions;
    Backward backward;
    std::size_t version = 0;
    std::uint64_t visitMark = 0;  // the last backward() pass that reached this node

    Node(ndp::ForkJoinPool& p, Shape s, Values v, Grad g)
        : pool(p), shape(std::move(s)), value(std::move(v)), requiresGrad(g == Grad::Required) {
        const std::size_t expected = elementCount(shape);
        if (expected != value.size())
            throw std::invalid_argument("tensor shape " + toString(shape) + " needs " +
                                        std::to_string(expected) + " values, got " +
                                        std::to_string(value.size()));
        if (requiresGrad) grad.assign(value.size(), 0.f);
    }

    // Releasing the last handle to a long chain would otherwise recurse once per node
    // through ~shared_ptr and overflow the stack (300,000 nodes did, on an 8 MB stack).
    // Parents this node solely owns are unlinked onto a worklist and freed one at a time.
    ~Node() {
        std::vector<Ptr> pending = std::move(parents);
        while (!pending.empty()) {
            Ptr parent = std::move(pending.back());
            pending.pop_back();
            // use_count() == 1: this is the last owner, so nothing can be reading its parents.
            if (parent.use_count() == 1) {
                for (Ptr& grandparent : parent->parents) pending.push_back(std::move(grandparent));
                parent->parents.clear();
            }
        }
    }
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    Node(Node&&) = delete;
    Node& operator=(Node&&) = delete;
};

namespace {
// Distinguishes backward() passes so a node can record "visited" without a hash set.
std::atomic<std::uint64_t> backwardPasses{0};
} // namespace

BasicTensor::BasicTensor(ndp::ForkJoinPool& p, Shape s, Values v, Grad g)
    : node_(std::make_shared<Node>(p, std::move(s), std::move(v), g)) {}
BasicTensor::BasicTensor(ndp::ForkJoinPool& p, float x, Grad g) : BasicTensor(p, {}, {x}, g) {}
const Shape& BasicTensor::shape() const { return node_->shape; }
const Values& BasicTensor::value() const { return node_->value; }
bool BasicTensor::requiresGrad() const { return node_->requiresGrad; }
bool BasicTensor::isLeaf() const { return node_->parents.empty(); }
ndp::ForkJoinPool& BasicTensor::pool() const { return node_->pool; }

const Values& BasicTensor::grad() const {
    if (!node_->requiresGrad) throw std::logic_error(notDifferentiable("grad", shape()));
    return node_->grad;
}

BasicTensor BasicTensor::result(Shape shape, Values values, std::vector<Ptr> parents,
                                Backward backward) const {
    const bool record = gradEnabled() &&
        std::any_of(parents.begin(), parents.end(), [](const Ptr& p) { return p->requiresGrad; });
    BasicTensor out(pool(), std::move(shape), std::move(values), record ? Grad::Required : Grad::None);
    if (record) {
        for (const auto& p : parents) out.node_->parentVersions.push_back(p->version);
        out.node_->parents = std::move(parents);
        out.node_->backward = std::move(backward);
    }
    return out;
}

BasicTensor BasicTensor::binary(const BasicTensor& rhs, Binary op) const {
    static constexpr const char* NAMES[] = {"operator+", "operator-", "operator*", "operator/"};
    const char* name = NAMES[static_cast<int>(op)];
    if (&pool() != &rhs.pool())
        throw std::invalid_argument(std::string(name) + ": operands belong to different pools");
    const Broadcast map(shape(), rhs.shape(), name);
    if (op == Binary::Divide) {
        const Values& divisor = rhs.value();
        const auto zero = std::find(divisor.begin(), divisor.end(), 0.f);
        if (zero != divisor.end())
            throw std::domain_error("operator/: division by zero at element " +
                                    std::to_string(zero - divisor.begin()) + " of divisor shape " +
                                    toString(rhs.shape()));
    }

    Values values(map.size());
    const float* a = value().data();
    const float* b = rhs.value().data();
    withOp(op, [&](auto opTag) {
        constexpr Binary OP = decltype(opTag)::value;
        detail::parallelRanges(pool(), values.size(), 1, [&](std::size_t begin, std::size_t end) {
            map.walk(begin, end, [&](std::size_t i, std::size_t ia, std::size_t ib) {
                values[i] = combine<OP>(a[ia], b[ib]);
            });
        });
    });

    return result(map.shape(), std::move(values), {node_, rhs.node_}, [map, op](const Node& out) {
        Node& left = *out.parents[0];
        Node& right = *out.parents[1];
        withOp(op, [&](auto opTag) {
            constexpr Binary OP = decltype(opTag)::value;
            for (unsigned side : {0u, 1u}) {
                Node& dest = side == 0 ? left : right;
                if (!dest.requiresGrad) continue;
                // Contribution of output i, which read a[ia] and b[ib], to this side's gradient.
                auto partial = [&](std::size_t i, std::size_t ia, std::size_t ib) {
                    const float g = out.grad[i];
                    if constexpr (OP == Binary::Add) return g;
                    else if constexpr (OP == Binary::Subtract) return side == 0 ? g : -g;
                    else if constexpr (OP == Binary::Multiply)
                        return g * (side == 0 ? right.value[ib] : left.value[ia]);
                    else {
                        if (side == 0) return g / right.value[ib];
                        // -g a / d², in double: d² in float underflows for |d| < 1e-19 and
                        // overflows for |d| > 2e19 although the gradient itself may fit.
                        // Double covers the square of every finite float.
                        const double d = right.value[ib];
                        return static_cast<float>(-double(g) * double(left.value[ia]) / (d * d));
                    }
                };
                if (map.covers(side)) {  // one output per element: output i owns dest.grad[i]
                    detail::parallelRanges(out.pool, out.grad.size(), 1,
                                           [&](std::size_t begin, std::size_t end) {
                        map.walk(begin, end, [&](std::size_t i, std::size_t ia, std::size_t ib) {
                            dest.grad[i] += partial(i, ia, ib);
                        });
                    });
                } else {  // repeated input: each task owns whole gradient elements
                    const std::size_t readers = out.grad.size() / dest.grad.size();
                    detail::parallelFor(out.pool, dest.grad.size(), readers, [&](std::size_t j) {
                        float total = dest.grad[j];
                        map.readersOf(side, j, [&](std::size_t i, std::size_t ia, std::size_t ib) {
                            total += partial(i, ia, ib);
                        });
                        dest.grad[j] = total;
                    });
                }
            }
        });
    });
}

BasicTensor BasicTensor::operator+(const BasicTensor& r) const { return binary(r, Binary::Add); }
BasicTensor BasicTensor::operator-(const BasicTensor& r) const { return binary(r, Binary::Subtract); }
BasicTensor BasicTensor::operator*(const BasicTensor& r) const { return binary(r, Binary::Multiply); }
BasicTensor BasicTensor::operator/(const BasicTensor& r) const { return binary(r, Binary::Divide); }

template<class Function, class Derivative>
BasicTensor BasicTensor::unary(Function f, Derivative derivative) const {
    Values values(value().size());
    const float* x = value().data();
    detail::parallelFor(pool(), values.size(), 1, [&](std::size_t i) { values[i] = f(x[i]); });
    return result(shape(), std::move(values), {node_}, [derivative](const Node& out) {
        Node& in = *out.parents[0];
        detail::parallelFor(out.pool, out.grad.size(), 1, [&](std::size_t i) {
            in.grad[i] += derivative(in.value[i], out.value[i]) * out.grad[i];
        });
    });
}

BasicTensor BasicTensor::pow(float p) const {
    if (!std::isfinite(p)) throw std::invalid_argument("pow: exponent must be finite, got " + std::to_string(p));
    const Values& v = value();
    const bool integral = std::trunc(p) == p;
    // Outside the domain: NaN input, a negative base with a fractional exponent, or a pole
    // at 0 for a negative exponent. With a gradient, also 0 < p < 1 at 0: infinite slope.
    const bool slopeAtZero = p > 0 && p < 1 && gradEnabled() && requiresGrad();
    const auto bad = std::find_if(v.begin(), v.end(), [&](float x) {
        return std::isnan(x) || (x < 0 && !integral) || (x == 0 && (p < 0 || slopeAtZero));
    });
    if (bad != v.end())
        throw std::domain_error("pow: element " + std::to_string(bad - v.begin()) + " of shape " +
                                toString(shape()) + " is " + std::to_string(*bad) + ", where x^" +
                                std::to_string(p) + (slopeAtZero && *bad == 0 ? " has an infinite slope"
                                                                               : " is undefined"));
    auto y = unary([p](float x) { return std::pow(x, p); },
                   [p](float x, float) { return p == 0 ? 0.f : p * std::pow(x, p - 1); });
    requireFinite("pow", y.value(), shape());
    return y;
}
BasicTensor BasicTensor::log() const {
    const Values& v = value();
    const auto bad = std::find_if(v.begin(), v.end(), [](float x) { return !(x > 0); });
    if (bad != v.end())
        throw std::domain_error("log: element " + std::to_string(bad - v.begin()) + " of shape " +
                                toString(shape()) + " is " + std::to_string(*bad) +
                                "; log needs positive input");
    return unary([](float x) { return std::log(x); }, [](float x, float) { return 1 / x; });
}
BasicTensor BasicTensor::exp() const {
    auto result = unary([](float x) { return std::exp(x); }, [](float, float y) { return y; });
    requireFinite("exp", result.value(), shape());
    return result;
}
BasicTensor BasicTensor::relu() const {
    return unary([](float x) { return std::max(0.f, x); }, [](float x, float) { return float(x > 0); });
}
BasicTensor BasicTensor::sigmoid() const {
    return unary([](float x) { return sigmoidValue(x); }, [](float, float y) { return y * (1 - y); });
}

BasicTensor BasicTensor::activate(const std::vector<Activation>& columns) const {
    if (shape().empty() || shape().back() != columns.size())
        throw std::invalid_argument("activate: " + std::to_string(columns.size()) +
                                    " activations for shape " + toString(shape()) +
                                    "; need one per element of the last axis");
    if (std::all_of(columns.begin(), columns.end(), [&](auto a) { return a == columns.front(); })) {
        switch (columns.front()) {
            case Activation::Linear: return identity();  // never an alias of the input
            case Activation::ReLU: return relu();
            case Activation::Sigmoid: return sigmoid();
        }
    }
    // Mixed activations: walk rows so the column is a loop counter, not i % width.
    const std::size_t width = columns.size();
    Values values(value().size());
    const float* x = value().data();
    detail::parallelRanges(pool(), values.size(), 1, [&](std::size_t begin, std::size_t end) {
        rowSegments(begin, end, width, [&](std::size_t row, std::size_t c0, std::size_t c1) {
            for (std::size_t c = c0; c < c1; ++c)
                values[row*width + c] = activationValue(columns[c], x[row*width + c]);
        });
    });
    return result(shape(), std::move(values), {node_}, [columns](const Node& out) {
        Node& in = *out.parents[0];
        const std::size_t cols = columns.size();
        detail::parallelRanges(out.pool, out.grad.size(), 1, [&](std::size_t begin, std::size_t end) {
            rowSegments(begin, end, cols, [&](std::size_t row, std::size_t c0, std::size_t c1) {
                for (std::size_t c = c0; c < c1; ++c) {
                    const std::size_t i = row*cols + c;
                    in.grad[i] += activationSlope(columns[c], out.value[i]) * out.grad[i];
                }
            });
        });
    });
}

BasicTensor BasicTensor::matmul(const BasicTensor& rhs) const {
    const auto& a = shape();
    const auto& b = rhs.shape();
    if (&pool() != &rhs.pool()) throw std::invalid_argument("matmul: operands belong to different pools");
    if (a.empty() || a.size() > 2 || b.empty() || b.size() > 2 || a.back() != b.front())
        throw std::invalid_argument("matmul: expected {m, k} or {k} times {k, n} or {k}; got " +
                                    toString(a) + " x " + toString(b));
    const std::size_t m = a.size() == 2 ? a[0] : 1;
    const std::size_t k = a.back();
    const std::size_t n = b.size() == 2 ? b[1] : 1;
    Shape dims;
    if (a.size() == 2) dims.push_back(m);
    if (b.size() == 2) dims.push_back(n);
    Values values(m*n);
    const float* A = value().data();
    const float* B = rhs.value().data();
    // C[i, :] = sum_kk A[i, kk] B[kk, :]. Each task owns a slice of C and sweeps rows of B
    // (contiguous, vectorized across columns, four rows of B per pass); every
    // element still sums over kk in order.
    detail::parallelRanges(pool(), values.size(), k, [&](std::size_t begin, std::size_t end) {
        rowSegments(begin, end, n, [&](std::size_t i, std::size_t j0, std::size_t j1) {
            accumulateRows(A + i*k, k, B + j0, n, j1 - j0, values.data() + i*n + j0);
        });
    });
    return result(std::move(dims), std::move(values), {node_, rhs.node_}, [m, k, n](const Node& out) {
        Node& left = *out.parents[0];
        Node& right = *out.parents[1];
        const float* dC = out.grad.data();
        // Step 1: dA[i, kk] += dC[i, :] . B[kk, :], two contiguous rows.
        if (left.requiresGrad)
            detail::parallelRanges(out.pool, m*k, n, [&](std::size_t begin, std::size_t end) {
                rowSegments(begin, end, k, [&](std::size_t i, std::size_t k0, std::size_t k1) {
                    for (std::size_t kk = k0; kk < k1; ++kk)
                        left.grad[i*k + kk] += dot(dC + i*n, right.value.data() + kk*n, n);
                });
            });
        // Step 2: dB[kk, :] += sum_i A[i, kk] dC[i, :], swept a contiguous row of dC at a time.
        if (right.requiresGrad)
            detail::parallelRanges(out.pool, k*n, m, [&](std::size_t begin, std::size_t end) {
                rowSegments(begin, end, n, [&](std::size_t kk, std::size_t j0, std::size_t j1) {
                    float* column = scratch(m);  // A[:, kk], gathered once per segment
                    for (std::size_t i = 0; i < m; ++i) column[i] = left.value[i*k + kk];
                    accumulateRows(column, m, dC + j0, n, j1 - j0, right.grad.data() + kk*n + j0);
                });
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
            "and out activations; got x " + toString(xs) + ", weights " + toString(ws) + ", bias " +
            toString(bias.shape()) + ", " + std::to_string(activations.size()) + " activations");
    // Backward writes dX and dW in the same pass; that is race-free only if they are different arrays.
    if (node_ == weights.node_ || node_ == bias.node_)
        throw std::invalid_argument("dense: x must not be the weights or bias tensor");

    const std::size_t rows = xs.size() == 2 ? xs[0] : 1, in = ws[1], out = ws[0];
    Shape dims = xs.size() == 2 ? Shape{rows, out} : Shape{out};
    Values y(elementCount(dims));
    const float* X = value().data();
    const float* W = weights.value().data();
    const float* B = bias.value().data();

    // Forward: y[r, j] = act_j(x[r, :] . W[j, :] + b[j]). Both rows are contiguous, and
    // the task that owns flat index r*out + j is the only writer of y[r, j]. (Computing four
    // outputs per pass over x[r, :] measured no faster: x's row is already in L1, and at
    // batch 1 streaming W dominates.)
    detail::parallelRanges(pool(), y.size(), in, [&](std::size_t begin, std::size_t end) {
        rowSegments(begin, end, out, [&](std::size_t r, std::size_t j0, std::size_t j1) {
            for (std::size_t j = j0; j < j1; ++j)
                y[r*out + j] = activationValue(activations[j], dot(X + r*in, W + j*in, in) + B[j]);
        });
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
            return activationSlope(acts[j], node.value[i]) * node.grad[i];
        };
        // Input data (Grad::None) gets no dX: for the first layer that is half the work.
        const bool needParams = w.requiresGrad || b.requiresGrad;
        const std::size_t layerWork = detail::saturatingProduct(rows*out, in);
        const std::size_t work = detail::saturatingProduct(
            layerWork, std::size_t(needParams) + std::size_t(x.requiresGrad));
        const unsigned tasks =
            detail::taskCount(node.pool, work, detail::saturatingProduct(std::max(out, rows), in));
        // Each task owns one slice of W's out*in elements (dW, plus db[j] for the slice
        // holding column 0 of row j) and one slice of x's rows*in elements (dX). The
        // slices are disjoint, so the three products are written without a lock.
        detail::runTasks(node.pool, tasks, [&](unsigned t) {
            // Step 1: dW[j, k] += sum_r dZ[r, j] x[r, k]; db[j] += sum_r dZ[r, j].
            const auto [wBegin, wEnd] = detail::chunk(out*in, t, tasks);
            if (needParams)
                rowSegments(wBegin, wEnd, in, [&](std::size_t j, std::size_t k0, std::size_t k1) {
                    if (!w.requiresGrad && k0 != 0) return;  // bias only: just the slice owning db[j]
                    float* column = scratch(rows);           // dZ[:, j], computed once per slice
                    for (std::size_t r = 0; r < rows; ++r) column[r] = dz(r, j);
                    if (b.requiresGrad && k0 == 0) {
                        float biasSum = 0;
                        for (std::size_t r = 0; r < rows; ++r) biasSum += column[r];
                        b.grad[j] += biasSum;
                    }
                    if (w.requiresGrad)
                        accumulateRows(column, rows, x.value.data() + k0, in, k1 - k0,
                                       w.grad.data() + j*in + k0);
                });

            // Step 2: dX[r, k] += sum_j dZ[r, j] W[j, k].
            const auto [xBegin, xEnd] = detail::chunk(rows*in, t, tasks);
            if (x.requiresGrad)
                rowSegments(xBegin, xEnd, in, [&](std::size_t r, std::size_t k0, std::size_t k1) {
                    float* row = scratch(out);  // dZ[r, :], computed once per slice
                    for (std::size_t j = 0; j < out; ++j) row[j] = dz(r, j);
                    accumulateRows(row, out, w.value.data() + k0, in, k1 - k0, x.grad.data() + r*in + k0);
                });
        });
    });
}

BasicTensor BasicTensor::identity() const {
    return unary([](float x) { return x; }, [](float, float) { return 1.f; });
}

BasicTensor BasicTensor::transpose() const {
    // A new node even when nothing moves, so the result never aliases its input's gradient.
    if (shape().size() < 2) return identity();
    const Shape& s = shape();
    const std::size_t rank = s.size();
    Shape dims(s.rbegin(), s.rend());
    // Input axis a becomes output axis rank-1-a. Row-major strides of both layouts:
    // inStrides[a] = product of s[a+1..], outStrides[a] = product of s[..a-1] (the stride
    // of output axis rank-1-a).
    Shape inStrides(rank), outStrides(rank);
    for (std::size_t a = rank, stride = 1; a-- > 0;) { inStrides[a] = stride; stride *= s[a]; }
    for (std::size_t a = 0, stride = 1; a < rank; ++a) { outStrides[a] = stride; stride *= s[a]; }
    // Forward gathers: walk the output in order, reading input axis rank-1-q for output axis q.
    const Shape gather(inStrides.rbegin(), inStrides.rend());
    Values values(value().size());
    const float* x = value().data();
    detail::parallelRanges(pool(), values.size(), 1, [&](std::size_t begin, std::size_t end) {
        walkStrided(dims, gather, begin, end, [&](std::size_t o, std::size_t i) { values[o] = x[i]; });
    });
    return result(std::move(dims), std::move(values), {node_}, [s, outStrides](const Node& out) {
        Node& in = *out.parents[0];
        // Backward walks the input in order, so each task owns a contiguous slice of in.grad.
        detail::parallelRanges(out.pool, in.grad.size(), 1, [&](std::size_t begin, std::size_t end) {
            walkStrided(s, outStrides, begin, end, [&](std::size_t i, std::size_t o) {
                in.grad[i] += out.grad[o];
            });
        });
    });
}

BasicTensor BasicTensor::sum(std::optional<int> axis, bool keepDims) const {
    Shape dims = shape();
    std::size_t width = value().size(), stride = 1;
    if (axis) {
        const int rank = static_cast<int>(dims.size());
        const int d = *axis < 0 ? rank + *axis : *axis;
        if (d < 0 || d >= rank)
            throw std::invalid_argument("sum: axis " + std::to_string(*axis) +
                                        " is out of range for shape " + toString(shape()));
        const auto dAxis = static_cast<std::size_t>(d);
        width = dims[dAxis];
        for (std::size_t j = dAxis + 1; j < dims.size(); ++j) stride *= dims[j];
        if (keepDims) dims[dAxis] = 1;
        else dims.erase(dims.begin() + d);
    } else {
        dims.assign(keepDims ? shape().size() : 0, 1);
    }
    // The input is {outer, width, stride} with the reduced axis in the middle; the output is
    // {outer, stride}. Output (o, s) starts at input (o, 0, s) and steps by stride.
    const std::size_t outer = value().size() / (width * stride);
    Values values(elementCount(dims));
    const float* x = value().data();
    detail::parallelRanges(pool(), values.size(), width, [&](std::size_t begin, std::size_t end) {
        walkStrided({outer, stride}, {width*stride, 1}, begin, end, [&](std::size_t i, std::size_t base) {
            double total = 0;
            for (std::size_t j = 0; j < width; ++j) total += x[base + j*stride];
            values[i] = static_cast<float>(total);
        });
    });
    return result(std::move(dims), std::move(values), {node_}, [outer, width, stride](const Node& out) {
        Node& in = *out.parents[0];
        // Input (o, w, s) reads output (o, s): stride 0 along the reduced axis.
        detail::parallelRanges(out.pool, in.grad.size(), 1, [&](std::size_t begin, std::size_t end) {
            walkStrided({outer, width, stride}, {stride, 0, 1}, begin, end,
                        [&](std::size_t i, std::size_t o) { in.grad[i] += out.grad[o]; });
        });
    });
}

// Its own node rather than sum() / n: the sum would round to float before the division,
// so finite inputs whose mean fits (two FLT_MAX) overflowed, and float(n) is inexact
// above 2^24. The total is the same double, summed in the same order, as sum()'s.
BasicTensor BasicTensor::mean() const {
    const double count = static_cast<double>(value().size());  // never 0: Node rejects empty shapes
    const double total = std::accumulate(value().begin(), value().end(), 0.0);
    return result({}, {static_cast<float>(total / count)}, {node_}, [count](const Node& out) {
        Node& in = *out.parents[0];
        // Equal to the old out.grad / float(n) for n <= 2^24: rounding a float quotient
        // through double first never changes it (53 >= 2*24 + 2 bits).
        const auto share = static_cast<float>(double(out.grad[0]) / count);
        detail::parallelFor(out.pool, in.grad.size(), 1, [&](std::size_t i) { in.grad[i] += share; });
    });
}

BasicTensor BasicTensor::crossEntropy(const std::vector<std::size_t>& labels) const {
    if (shape().size() != 2 || shape()[0] != labels.size())
        throw std::invalid_argument("crossEntropy: expected logits {rows, classes} and one label "
                                    "per row; got shape " + toString(shape()) + " and " +
                                    std::to_string(labels.size()) + " labels");
    const std::size_t classes = shape()[1];
    for (std::size_t r = 0; r < labels.size(); ++r)
        if (labels[r] >= classes)
            throw std::invalid_argument("crossEntropy: label " + std::to_string(labels[r]) +
                                        " at row " + std::to_string(r) + " is not below " +
                                        std::to_string(classes) + " classes");
    // The softmax gradient falls out of the forward pass; keep it only if it will be used.
    const bool record = gradEnabled() && requiresGrad();
    const double rows = static_cast<double>(labels.size());
    Values gradient(record ? value().size() : 0);
    std::vector<double> losses(labels.size());
    detail::parallelFor(pool(), labels.size(), classes, [&](std::size_t r) {
        const auto row = value().begin() + std::ptrdiff_t(r*classes);
        const double maxLogit = *std::max_element(row, row + std::ptrdiff_t(classes));
        double denom = 0;
        for (std::size_t j = 0; j < classes; ++j) {
            const double e = std::exp(row[std::ptrdiff_t(j)] - maxLogit);
            denom += e;
            if (record) gradient[r*classes + j] = static_cast<float>(e);
        }
        losses[r] = (maxLogit - row[std::ptrdiff_t(labels[r])]) + std::log(denom);
        if (record)
            for (std::size_t j = 0; j < classes; ++j) {
                const double target = j == labels[r] ? 1.0 : 0.0;
                gradient[r*classes + j] = static_cast<float>((gradient[r*classes + j]/denom - target) / rows);
            }
    });
    const auto loss = static_cast<float>(std::accumulate(losses.begin(), losses.end(), 0.0) / rows);
    return result({}, {loss}, {node_}, [g = std::move(gradient)](const Node& out) {
        Node& in = *out.parents[0];
        detail::parallelFor(out.pool, g.size(), 1, [&](std::size_t i) { in.grad[i] += g[i] * out.grad[0]; });
    });
}

void BasicTensor::backward() const {
    if (!node_->requiresGrad) throw std::logic_error(notDifferentiable("backward", shape()));
    // Step 1: depth-first post-order over the nodes that need gradients, so each node comes
    // after everything it reads. Shared nodes are visited once: each pass stamps the nodes
    // it reaches with its own number. Concurrent passes touch disjoint graphs (graphs that
    // share a parameter must not run backward concurrently anyway), so plain stamps suffice.
    const std::uint64_t pass = backwardPasses.fetch_add(1, std::memory_order_relaxed) + 1;
    std::vector<Node*> order;
    node_->visitMark = pass;
    std::vector<std::pair<Node*, std::size_t>> stack{{node_.get(), 0}};
    while (!stack.empty()) {
        auto& [node, next] = stack.back();
        if (next == node->parents.size()) {
            order.push_back(node);
            stack.pop_back();
            continue;
        }
        const std::size_t i = next++;
        Node* parent = node->parents[i].get();
        if (parent->version != node->parentVersions[i])
            throw std::logic_error("backward: an input of shape " + toString(parent->shape) +
                                   " was changed by step() after this graph was built; "
                                   "run the forward pass again");
        if (parent->requiresGrad && parent->visitMark != pass) {
            parent->visitMark = pass;
            stack.emplace_back(parent, 0);
        }
    }
    // Step 2: intermediate gradients restart from zero; leaf gradients accumulate.
    for (Node* node : order)
        if (!node->parents.empty()) std::fill(node->grad.begin(), node->grad.end(), 0.f);
    if (node_->parents.empty()) {
        for (float& g : node_->grad) g += 1;
        return;
    }
    // Step 3: seed d(out)/d(out) = 1 and run the backward closures from the output down.
    std::fill(node_->grad.begin(), node_->grad.end(), 1.f);
    for (auto it = order.rbegin(); it != order.rend(); ++it)
        if ((*it)->backward) (*it)->backward(**it);
}

void BasicTensor::zeroGrad() {
    if (!node_->requiresGrad) throw std::logic_error(notDifferentiable("zeroGrad", shape()));
    std::fill(node_->grad.begin(), node_->grad.end(), 0.f);
}

void BasicTensor::step(float learningRate) {
    if (!node_->parents.empty())
        throw std::logic_error("step: SGD updates leaf tensors; this tensor of shape " +
                               toString(shape()) + " is the result of an operation");
    if (!node_->requiresGrad) throw std::logic_error(notDifferentiable("step", shape()));
    if (!std::isfinite(learningRate) || learningRate < 0)
        throw std::invalid_argument("step: learning rate must be finite and non-negative, got " +
                                    std::to_string(learningRate));
    detail::parallelFor(pool(), value().size(), 1, [&](std::size_t i) {
        node_->value[i] -= learningRate * node_->grad[i];
    });
    ++node_->version;
}

void BasicTensor::accumulateGrad(const Values& delta) {
    if (!node_->requiresGrad) throw std::logic_error(notDifferentiable("accumulateGrad", shape()));
    if (delta.size() != node_->grad.size())
        throw std::invalid_argument("accumulateGrad: " + std::to_string(delta.size()) +
                                    " values for a tensor of shape " + toString(shape()));
    detail::parallelFor(pool(), delta.size(), 1, [&](std::size_t i) { node_->grad[i] += delta[i]; });
}

} // namespace ml
