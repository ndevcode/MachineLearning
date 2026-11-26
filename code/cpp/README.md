# CPU autograd

A C++20 library based on the CUDA project's `BasicTensor`, `Module`, `Layer`, and `MLP`. Its kernels run on `ndp::ForkJoinPool`. Build from the repository root with CMake and a C++ compiler.

```sh
cmake -S code/cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp -j
ctest --test-dir build/cpp --output-on-failure
```

## Training with the library

Link your executable to the CMake target `cpu_autograd`. A dense layer accepts a feature vector or a row-major batch shaped `{examples, features}`. Weights have shape `{outputs, inputs}` and the bias broadcasts across the batch.

```cpp
#include "Module.h"

int main() {
    ndp::ForkJoinPool pool(4);  // the calling thread plus three workers
    ml::MLP model(pool, 2, {8, 2});
    ml::BasicTensor x(pool, {3, 2}, {0, 1, 1, 0, 1, 1});
    model.zero_grad();
    auto logits = model(x);
    auto loss = logits.cross_entropy({1, 1, 0});
    loss.backward();
    model.step(0.01f);
}
```

Tensor copies share the same graph node. `backward()` seeds each output element with one and accumulates leaf gradients. Call `zero_grad()` before each independent training step. Keep the pool alive while using its tensors. Independent graphs can share a pool, but only one caller thread at a time gets the workers; a concurrent caller runs its kernels serially. Serialize training steps that share parameters.

Elementwise operations support trailing-axis broadcasting. Matrix multiplication accepts vectors and matrices. The library also provides transpose, sum by axis, mean, powers, log, exp, ReLU, sigmoid, mean softmax cross-entropy on logits, and `dense`, a whole layer as one graph node.

## Threading

Every kernel splits its output indexes into contiguous, disjoint slices, one per task. A task writes only its own slice, so no kernel takes a lock, and results are bitwise identical for any thread count. `ForkJoinPool::run(tasks, f)` runs task 0 on the calling thread and task `w` on worker `w`. To dispatch, it bumps an atomic ticket for each worker that has a task. To finish, it waits for an atomic countdown to reach zero. That release/acquire pair is the only synchronization, and the pool has no mutex, queue or per-task allocation. A dispatch to spinning workers costs about 0.3 µs with 4 threads and 1.4 µs with 16. Idle threads spin for 2 ms before sleeping on a futex, because waking a sleeping thread cost 40–260 µs under WSL2. Override the spin time with `-DNDP_SPIN_MICROSECONDS=`. Loops under about 16K multiply-adds run serially (`kMinWorkPerTask` in `Parallel.h`).

`Layer` uses `dense`: forward is one parallel pass over the outputs `y[r, j]`. Backward is one parallel pass in which each task owns a slice of `dW`, the matching `db` entries, and a slice of `dX`. That gives one barrier per layer in each direction, where the unfused `matmul(W.T()) + b` and `activate` path needs four forward and five backward. The tests check `dense` against that unfused path.

`tests/ForkJoinPoolTest.cpp` covers the pool and the partitioning in `Parallel.h`. CTest builds it twice. `ForkJoinPoolSleepTest` compiles it with spinning disabled, so every wait goes through the futex sleep and wake handshake. Run the suite under `-fsanitize=thread` to check the memory ordering. Under WSL2, ThreadSanitizer needs ASLR off: `setarch -R ctest`.

## MNIST

```sh
build/cpp/mnist ./assets/mnist ./improved_network.dat 25 16 4
```

The arguments are the IDX directory, checkpoint, epochs, training batch size, and thread count (including the calling thread). The defaults are `./assets/mnist`, `improved_network.dat` inside that directory, 25 epochs, one example per update, and hardware concurrency. Supply uncompressed train and t10k IDX image/label pairs. Both `train-images.idx3-ubyte` and `train-images-idx3-ubyte` naming conventions work.

The example trains hidden layers of 16, 8, and 16 ReLU units with ten sigmoid outputs. SGD uses a learning rate of 0.01 and one-hot squared error summed across outputs and averaged across examples. The reported error is the mean per output. Starting in epoch two, each training image shifts by up to one pixel per axis with zero padding.

An existing checkpoint goes straight to test-set evaluation. The binary layout matches the supplied sequential model's native 32-bit counts, per-neuron activation bytes, and float weights with each bias following its neuron's weights. A library MLP with linear outputs uses the additional activation code 2. Saving publishes a completed file using a hard link in the destination directory.
