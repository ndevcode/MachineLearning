# CPU autograd

A C++20 library based on the CUDA project's `BasicTensor`, `Module`, `Layer`, and `MLP`. Its kernels run on `ndp::ForkJoinPool`. Build from the repository root with CMake and a C++ compiler.

```sh
cmake -S code/cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp -j
ctest --test-dir build/cpp --output-on-failure
```

Without `-DCMAKE_BUILD_TYPE`, the build defaults to Release, because `-O0` would disable the vectorization the kernels rely on. The library compiles with `-ffp-contract=off`: whether the compiler fuses `a*b + c` into one FMA depends on `-march` and on loop shape, and so would the last bits of every result. With it off, results are bitwise identical at `-O2`, `-O3` and `-O3 -march=native`, and measured no slower under `-march=native`. Every target builds with `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`. Add `-DCPU_AUTOGRAD_SANITIZE=address,undefined` or `-DCPU_AUTOGRAD_SANITIZE=thread` to build everything with sanitizers. Under WSL2, ThreadSanitizer needs ASLR off: `setarch -R ctest`.

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

Tensor copies share the same graph node. `backward()` seeds each output element with one and accumulates leaf gradients. Call `zeroGrad()` before each independent training step. Keep the pool alive while using its tensors. Independent graphs, on any number of threads, can share a pool: each kernel claims whichever workers are idle. Serialize training steps that share parameters. Operations return new graph nodes, never aliases of their input, even when nothing moves (`activate` with all-`Linear` columns, `transpose` of a vector). Destroying a long graph is iterative, so chains of any length tear down without growing the stack.

Leaf tensors default to `Grad::Required`. Create input data and targets with `Grad::None`. An operation's result needs a gradient only when one of its operands does, and backward skips the work for operands that need none; for the first layer, that is half of the backward pass. While a `NoGradGuard` is alive on a thread, operations record no graph and allocate no gradient buffers. `grad()`, `zeroGrad()`, `step()` and `backward()` throw on a tensor that needs no gradient.

Errors fail loudly, naming the element; nothing is padded with an epsilon. These throw `std::domain_error`:

- division by zero;
- `log` of a value that is not positive, or of NaN;
- `pow` with a negative base and a fractional exponent;
- `pow` at zero with a negative exponent;
- `pow` at zero with 0 < p < 1, when a gradient is needed (the slope is infinite).

`exp` and `pow` results that overflow float throw `std::overflow_error`. Shape errors throw `std::invalid_argument` naming the shapes, for example `matmul: expected {m, k} or {k} times {k, n} or {k}; got {2, 3} x {2, 3}`.

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

Use sharding for narrow networks with large batches, where every layer is too small to split. When a wide first layer dominates, splitting each operation is faster. Sharding also copies the parameters once per thread and keeps one gradient per shard (`shards` × the parameter count, in floats) until the final sum. Summing per thread instead would save that memory, but the rounding would then depend on the thread count.

## Threading

Every kernel splits its output indexes into contiguous, disjoint slices, one per task. A task writes only its own slice, so no kernel takes a lock, and results are bitwise identical for any thread count. `ForkJoinPool::run(tasks, f)` runs task 0 on the calling thread, and hands each other task to an idle worker:

1. **Claim:** compare-and-swap on the worker's slot.
2. **Dispatch:** write the task into the slot and bump its atomic ticket.
3. **Complete:** the worker bumps the slot's done counter; the caller waits for that, then releases the slot.

Those release/acquire pairs are the only synchronization; there is no mutex, queue or per-task allocation. A task that finds no idle worker runs on the calling thread. Concurrent callers and nested runs therefore share whatever workers are free and never fall back to serial while a worker is idle. A thread waits only on workers it claimed, so the waits form a tree and cannot deadlock. If several tasks throw, the lowest-numbered task's exception reaches the caller, whichever finished first. A dispatch to spinning workers costs about 0.3 µs with 4 threads and 1.1 µs with 16. Idle threads spin for 2 ms before sleeping on a futex, because waking a sleeping thread cost 40–260 µs under WSL2. Override the spin time with `-DNDP_SPIN_MICROSECONDS=`. Loops under about 16K multiply-adds run serially (`MIN_WORK_PER_TASK` in `Parallel.h`).

`Layer` uses `dense`: forward is one parallel pass over the outputs `y[r, j]`. Backward is one parallel pass in which each task owns a slice of `dW`, the matching `db` entries, and a slice of `dX`. That gives one barrier per layer in each direction, where the unfused `matmul(W.transpose()) + b` and `activate` path needs four forward and five backward. The tests check `dense` against that unfused path.

Every inner loop reads contiguous memory. Dot products (the `dense` forward and the input gradient of `matmul`) keep 8 independent partial sums, combined in a fixed order. GCC vectorizes that without `-ffast-math`, about 6× faster than a sequential sum at `-O2` and `-O3`, and the result is the same for every thread count.

The other products (`matmul` forward and weight gradient, `dense` weight and input gradients) share one kernel. It streams 4 input rows per pass across the full output width, so the accumulator is loaded and stored once per 4 rows, and every element still sums its rows in order. Two alternatives were measured on one thread and rejected:

- 32-column register blocks walk the rows in strided strips. In an SSE2 build at batch 1, where those rows are read from memory once, they lost up to 63%.
- Computing 4 dense outputs per pass over the input row gained nothing, since that row already sits in L1.

Against the previous row-at-a-time loop, the current kernel is unchanged at batch 1. It cuts MNIST batch 64 from 145 to 119 µs, wide batch 128 from 16.8 to 12.2 ms, and a 256³ `matmul` forward plus backward from 4.0 to 2.4 ms, all bitwise identical.

Broadcasting, `transpose` and `sum` backward walk index ranges with an odometer, dividing once per range rather than once per element. A broadcast operand's gradient is split by element, and each task sums that element's readers in output order.

`tests/ForkJoinPoolTest.cpp` covers the pool and the partitioning in `Parallel.h`. CTest builds it twice. `ForkJoinPoolSleepTest` compiles it with spinning disabled, so every wait goes through the futex sleep and wake handshake. `tests/AutogradTest.cpp` covers:

- broadcasting, `matmul`, `transpose` and `sum` against naive reference loops;
- `dense` against the unfused ops, and gradients against finite differences;
- `backwardSharded` against a single graph;
- the domain and overflow errors, a 400,000-node teardown, and the no-aliasing rule;
- bitwise identical results across 1, 3 and 4 threads.

## MNIST

```sh
build/cpp/mnist ./assets/mnist ./improved_network.dat 25 16 4
```

The arguments are the IDX directory, checkpoint, epochs, training batch size, and thread count (including the calling thread). The defaults are `./assets/mnist`, `improved_network.dat` inside that directory, 25 epochs, one example per update, and hardware concurrency. Supply uncompressed train and t10k IDX image/label pairs. Both `train-images.idx3-ubyte` and `train-images-idx3-ubyte` naming conventions work.

The example trains hidden layers of 16, 8, and 16 ReLU units with ten sigmoid outputs. SGD uses a learning rate of 0.01 and one-hot squared error summed across outputs and averaged across examples. The reported error is the mean per output. The training order is reshuffled every epoch from a fixed seed, so runs are reproducible. Starting in epoch two, each training image shifts by up to one pixel per axis with zero padding. Training splits each operation rather than calling `backwardSharded`, which measured twice as slow for this network. Evaluation runs under `NoGradGuard`. With a batch of 1, no layer reaches the 16K multiply-add threshold for splitting, so training runs on one thread; threads measured slower at that size.

An existing checkpoint goes straight to test-set evaluation, with a note on stderr. If EPOCHS or BATCH_SIZE is given as well, the program fails rather than silently ignore them.

The checkpoint layout matches the supplied sequential model's files: 32-bit counts, per-neuron activation bytes, and float weights with each bias following its neuron's weights. It is little-endian, the byte order of the x86 machines that wrote those files, and big-endian hosts byte-swap. A library MLP with linear outputs uses the additional activation code 2. Saving publishes a completed file through a hard link in the destination directory and never overwrites an existing one. On filesystems without hard links (FAT, exFAT, some network shares), it checks that the destination is absent and then renames. That rename still publishes only a complete file, but a writer that creates the destination between the check and the rename can be overwritten.

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
