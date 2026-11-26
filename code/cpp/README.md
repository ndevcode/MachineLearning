# CPU autograd

A C++17 library based on the CUDA project's `BasicTensor`, `Module`, `Layer`, and `MLP`. The matrix and activation kernels use the ndplace thread pool. Build from the repository root with CMake and a C++ compiler.

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
    ndp::ThreadPool pool(4);
    ml::MLP model(pool, 2, {8, 2});
    ml::BasicTensor x(pool, {3, 2}, {0, 1, 1, 0, 1, 1});
    model.zero_grad();
    auto logits = model(x);
    auto loss = logits.cross_entropy({1, 1, 0});
    loss.backward();
    model.step(0.01f);
}
```

Tensor copies share the same graph node. `backward()` seeds each output element with one and accumulates leaf gradients. Call `zero_grad()` before each independent training step. Keep the pool alive while using its tensors. Independent graphs can share a pool. Serialize training steps that share parameters.

Elementwise operations support trailing-axis broadcasting. Matrix multiplication accepts vectors and matrices. The library also provides transpose, sum by axis, mean, powers, log, exp, ReLU, sigmoid, and mean softmax cross-entropy on logits.

## MNIST

```sh
build/cpp/mnist ./assets/mnist ./improved_network.dat 25 16 4
```

The arguments are the IDX directory, checkpoint, epochs, training batch size, and worker count. The defaults are `./assets/mnist`, `improved_network.dat` inside that directory, 25 epochs, one example per update, and hardware concurrency. Supply uncompressed train and t10k IDX image/label pairs. Both `train-images.idx3-ubyte` and `train-images-idx3-ubyte` naming conventions work.

The example trains hidden layers of 16, 8, and 16 ReLU units with ten sigmoid outputs. SGD uses a learning rate of 0.01 and one-hot squared error summed across outputs and averaged across examples. The reported error is the mean per output. Starting in epoch two, each training image shifts by up to one pixel per axis with zero padding.

An existing checkpoint goes straight to test-set evaluation. The binary layout matches the supplied sequential model's native 32-bit counts, per-neuron activation bytes, and float weights with each bias following its neuron's weights. A library MLP with linear outputs uses the additional activation code 2. Saving publishes a completed file using a hard link in the destination directory.
