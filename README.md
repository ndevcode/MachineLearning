# Machine Learning and Deep Learning

This repository is a collection of machine-learning programs I have written to understand the
algorithms from the ground up. The work falls into two halves: from-scratch NumPy
implementations where the forward *and* backward passes are derived and written by hand (no
autograd), and PyTorch reimplementations of landmark architectures that share a small training
harness. Several of the from-scratch pieces exist to make the mathematics explicit rather than
to be fast. Numerical-stability practices — softmax max-subtraction, guarded logarithms, and
gradient and weight clipping — are applied throughout, and the models are validated on MNIST and
an English–Spanish corpus.

**Running.** Each script is self-contained and guarded by `if __name__ == "__main__":`, so
importing a module never triggers training or file I/O. Dataset locations are set by a path
constant near the top of each file; the DCGAN instead takes them on the command line
(`--data-path`, `--out-path`, `--log-path`).

---

## From scratch (NumPy, hand-derived gradients)

These implement backpropagation by hand and are the core of the repository.

- [N-layer network](code/nlayerMay2023.py) - A configurable fully-connected network for MNIST
  digit recognition with forward and backward propagation written by hand in NumPy.
  Hyperparameters are bundled in a validated config object, activations (tanh / ReLU) are
  selected through a dispatch table, and dropout and L2 regularization are included; the log
  term is guarded against non-positive inputs so the cross-entropy stays finite. It reaches
  roughly 92% train and 89.5% test accuracy, and can plot training error across experiments.

- [CNN forward and backward](code/CNN_fwd_bck_pass.py) - Convolution and pooling forward and
  backward passes derived from first principles in NumPy. The backward pass routes gradients
  through the max-pool argmax and accumulates the convolution gradient window by window, so the
  whole chain is explicit with no autograd.

- [GAN (NumPy)](code/gan_numpyonly.py) - A generator/discriminator pair trained to produce sine
  waves with manually-derived backpropagation. Activations and their derivatives are held in a
  dispatch table, and training uses gradient clipping and weight clipping for stability.

- [Transformer (NumPy)](code/transformer_numpy.py) - An encoder with multi-head self-attention
  and a decoder with masked self-attention and encoder–decoder (cross) attention, all in NumPy.
  The softmax subtracts the row maximum before exponentiating. This is an illustrative,
  readable version rather than a trained model.

## PyTorch — vision architectures

Reimplementations from the papers, sharing the training harness below.

- [AlexNet](code/alexnet.py) - The AlexNet convolutional classifier.

- [Inception v1](code/inceptionv1.py) - GoogLeNet inception blocks, each concatenating 1×1,
  3×3, 5×5, and pooling branches; the auxiliary classifiers are omitted.

- [ResNet-18](code/resnet18.py) - Residual `BasicBlock`s with a projection shortcut (a 1×1
  convolution matching the block stride followed by BatchNorm, with no activation on the
  shortcut so it stays linear).

- [YOLO](code/yolo_cnn.py) - A simplified YOLO detector whose loss is implemented from scratch:
  boxes are matched to the responsible predictor by IoU (argmax + scatter), width and height
  are regressed under a square-root transform so large and small boxes are weighted comparably,
  and the λ_coord / λ_noobj terms balance localization against the no-object confidence. Custom
  IoU and non-maximum suppression are included.

## Transformers

- [NMT — English→Spanish](code/nmt_final.py) - An encoder–decoder translation transformer
  composed from an attention head, a multi-head wrapper, and encoder / decoder / cross-attention
  blocks built through a factory of attention variants. Training is teacher-forced in parallel;
  inference decodes greedily, one token at a time. A fused multi-head projection and a KV cache
  for decoding are left as future work.

- [Vision Transformer](code/vision_transformer.py) - Images are unfolded into patches,
  projected, prefixed with a learnable CLS token and positional encoding, and classified from
  the CLS output. The encoder stack itself uses PyTorch's built-in `TransformerEncoderLayer`
  rather than a hand-written one.

## Generative Adversarial Networks

- [DCGAN (PyTorch)](code/dgcan1.py) - A deep-convolutional GAN generating MNIST samples, with a
  transposed-convolution generator and a convolutional discriminator. It checkpoints the
  generator, discriminator, and both optimizer states so training can resume, and takes its data
  / output / log paths on the command line.

- [GAN (NumPy)](code/gan_numpyonly.py) - The from-scratch counterpart; see above.

## Training infrastructure

A small, reusable harness drives the PyTorch models, separating four concerns: `MLSystem` owns
the train/validation step (built as closures), the device, and the epoch loop; `TrainingLogger`
observes events for logging and checkpointing; `EarlyStopping` tracks the best validation metric
and stops on patience exhaustion; and `ExperimentRunner` composes the three and runs a seeded
experiment. Checkpoints save the model *and* optimizer state so training can resume. The fullest
worked example is the [MNIST classifier](code/mnist_linear_conv.py), which drives
fully-connected, convolutional, and custom weight/bias layers through the harness. The harness
is currently duplicated across the PyTorch scripts; consolidating it into a single shared module
is left as future work.
