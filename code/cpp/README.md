# CPU autograd

A C++20 library based on the CUDA project's `BasicTensor`, `Module`, `Layer`, and `MLP`. Its kernels run on `ndp::ForkJoinPool`. Build from the repository root with CMake and a C++ compiler.

```sh
cmake -S code/cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp -j
ctest --test-dir build/cpp --output-on-failure
```

Every target builds with `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`. Add `-DCPU_AUTOGRAD_SANITIZE=address,undefined` or `-DCPU_AUTOGRAD_SANITIZE=thread` to build everything with sanitizers. Under WSL2, ThreadSanitizer needs ASLR off: `setarch -R ctest`.

## Training with the library

Link your executable to the CMake target `cpu_autograd`. A dense layer accepts a feature vector or a row-major batch shaped `{examples, features}`. Weights have shape `{outputs, inputs}` and the bias broadcasts across the batch.

```cpp
#include "Module.h"

int main() {
    ndp::ForkJoinPool pool(4);  // the calling thread plus three workers
    ml::MLP model(pool, 2, {8, 2});
    ml::BasicTensor x(pool, {3, 2}, {0, 1, 1, 0, 1, 1}, ml::Grad::None);  // data: no gradient
    model.zeroGrad();
    auto logits = model(x);
    auto loss = logits.crossEntropy({1, 1, 0});
    loss.backward();
    model.step(0.01f);

    ml::NoGradGuard inference;  // no graph, no gradient buffers
    auto prediction = model(x);
}
```

Tensor copies share the same graph node. `backward()` seeds each output element with one and accumulates leaf gradients. Call `zeroGrad()` before each independent training step. Keep the pool alive while using its tensors. Independent graphs can share a pool, but only one caller thread at a time gets the workers; a concurrent caller runs its kernels serially. Serialize training steps that share parameters.

Leaf tensors default to `Grad::Required`. Create input data and targets with `Grad::None`. An operation's result needs a gradient only when one of its operands does, and backward skips the work for operands that need none; for the first layer, that is half of the backward pass. While a `NoGradGuard` is alive on a thread, operations record no graph and allocate no gradient buffers. `grad()`, `zeroGrad()`, `step()` and `backward()` throw on a tensor that needs no gradient.

Errors fail loudly. Division by zero and `log` of a non-positive or NaN value throw `std::domain_error`, naming the element. They are never padded with an epsilon. Shape errors throw `std::invalid_argument` naming the shapes, for example `matmul: expected {m, k} or {k} times {k, n} or {k}; got {2, 3} x {2, 3}`.

Elementwise operations support trailing-axis broadcasting. Matrix multiplication accepts vectors and matrices. The library also provides `transpose`, sum by axis, mean, powers, log, exp, ReLU, sigmoid, mean softmax cross-entropy on logits (`crossEntropy`), and `dense`, a whole layer as one graph node.

### Data-parallel training

`MLP::backwardSharded(x, shards, loss)` splits the batch into `shards` row blocks. It runs each block's whole forward and backward as one task, on its own copy of the parameters. It then adds the block gradients to the parameters in block order. That is one parallel call per step, and the result depends on `shards` but not on the thread count. `loss(output, first, rows)` returns the mean loss of its block and must be safe to call from several threads.

On the i7-13650HX the two strategies split as follows (time per SGD step, 16 shards):

| Network, batch | Per-operation, best | Sharded, best |
|---|---|---|
| 784 → 16-8-16-10, 64 | 53 µs (16 threads) | 112 µs (16 threads) |
| 784 → 512-512-10, 128 | 1.8 ms (16 threads) | 12.8 ms (4 threads) |
| 32 → 16-8-16-10, 1024 | 281 µs (16 threads) | 106 µs (16 threads) |
| 64 → 64-64-10, 1024 | 647 µs (16 threads) | 351 µs (16 threads) |

Use sharding for narrow networks with large batches, where every layer is too small to split. When a wide first layer dominates, splitting each operation is faster. Sharding also copies the parameters once per thread and keeps one gradient per shard.

## Threading

Every kernel splits its output indexes into contiguous, disjoint slices, one per task. A task writes only its own slice, so no kernel takes a lock, and results are bitwise identical for any thread count. `ForkJoinPool::run(tasks, f)` runs task 0 on the calling thread and task `w` on worker `w`. To dispatch, it bumps an atomic ticket for each worker that has a task. To finish, it waits for an atomic countdown to reach zero. That release/acquire pair is the only synchronization, and the pool has no mutex, queue or per-task allocation. A dispatch to spinning workers costs about 0.3 µs with 4 threads and 1.4 µs with 16. Idle threads spin for 2 ms before sleeping on a futex, because waking a sleeping thread cost 40–260 µs under WSL2. Override the spin time with `-DNDP_SPIN_MICROSECONDS=`. Loops under about 16K multiply-adds run serially (`MIN_WORK_PER_TASK` in `Parallel.h`).

`Layer` uses `dense`: forward is one parallel pass over the outputs `y[r, j]`. Backward is one parallel pass in which each task owns a slice of `dW`, the matching `db` entries, and a slice of `dX`. That gives one barrier per layer in each direction, where the unfused `matmul(W.transpose()) + b` and `activate` path needs four forward and five backward. The tests check `dense` against that unfused path.

Every inner loop reads contiguous memory. Dot products (the `dense` forward and the input gradient of `matmul`) keep 8 independent partial sums, combined in a fixed order. GCC vectorizes that without `-ffast-math`, about 6× faster than a sequential sum at `-O2` and `-O3`, and the result is the same for every thread count. The other products (`matmul` forward, the weight gradients) accumulate along contiguous rows, so they vectorize across output elements without reordering any element's sum. Broadcasting walks index ranges with an odometer, dividing once per range rather than once per element. A broadcast operand's gradient is split by element, and each task sums that element's readers in output order.

`tests/ForkJoinPoolTest.cpp` covers the pool and the partitioning in `Parallel.h`. CTest builds it twice. `ForkJoinPoolSleepTest` compiles it with spinning disabled, so every wait goes through the futex sleep and wake handshake. `tests/AutogradTest.cpp` checks broadcasting and `matmul` against naive reference loops, `dense` against the unfused ops, gradients against finite differences, and `backwardSharded` against a single graph. It also checks that results are bitwise identical across 1, 3 and 4 threads.

## MNIST

```sh
build/cpp/mnist ./assets/mnist ./improved_network.dat 25 16 4
```

The arguments are the IDX directory, checkpoint, epochs, training batch size, and thread count (including the calling thread). The defaults are `./assets/mnist`, `improved_network.dat` inside that directory, 25 epochs, one example per update, and hardware concurrency. Supply uncompressed train and t10k IDX image/label pairs. Both `train-images.idx3-ubyte` and `train-images-idx3-ubyte` naming conventions work.

The example trains hidden layers of 16, 8, and 16 ReLU units with ten sigmoid outputs. SGD uses a learning rate of 0.01 and one-hot squared error summed across outputs and averaged across examples. The reported error is the mean per output. Starting in epoch two, each training image shifts by up to one pixel per axis with zero padding. Training splits each operation rather than calling `backwardSharded`, which measured twice as slow for this network. Evaluation runs under `NoGradGuard`.

An existing checkpoint goes straight to test-set evaluation. The binary layout matches the supplied sequential model's native 32-bit counts, per-neuron activation bytes, and float weights with each bias following its neuron's weights. A library MLP with linear outputs uses the additional activation code 2. Saving publishes a completed file using a hard link in the destination directory.

## License

Copyright 2026 N Dev (ndevcode)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
