# C Convolutional Neural Network (MNIST)

> Pure-C implementation of a tiny Convolutional Neural Network trained on the classic **MNIST** handwritten-digits dataset – no external ML libraries, just math and standard C.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  ![Language: C](https://img.shields.io/badge/language-C-blue)

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Dataset](#dataset)
7. [Results](#results)
8. [Project Structure](#project-structure)
9. [Contributing](#contributing)
10. [License](#license)

---

## Overview
This project was built to **demystify** Convolutional Neural Networks by showing that you can train one with nothing more than the C standard library. The code is intentionally kept small and linear so that every line can be followed with a debugger or printed to the terminal for educational purposes.

*Written for fun*

## Features

### How CNNs Work
Convolutional Neural Networks (CNNs) are specialized neural networks designed for processing structured grid data like images. The key idea is to automatically learn spatial hierarchies of features through backpropagation by using three main architectural ideas:

1. **Local Receptive Fields**: Neurons are connected only to a local region of the input volume (unlike fully-connected layers).
2. **Shared Weights**: The same filter (set of weights) is used across different positions of the input.
3. **Spatial Subsampling**: Pooling layers reduce the spatial size of the representation.

### Implementation Details
This project implements a simple yet complete CNN in pure C with the following components:

- **Pure C Implementation**: No external ML libraries - just standard C and math.
- **Network Architecture**:
  - Input: 28×28 grayscale MNIST digits (normalized to [0,1])
  - Convolutional Layer: 8 filters (5×5), stride 1, valid padding
  - ReLU Activation: Introduces non-linearity (max(0, x))
  - Max Pooling: 2×2 window, stride 2 (reduces spatial dimensions)
  - Flatten: Converts 3D feature maps to 1D vector
  - Dense Layer: Fully-connected layer with 10 output units (one per digit)
  - Softmax: Outputs probability distribution over 10 classes

- **Training**:
  - Forward & Backward passes implemented from scratch
  - Cross-entropy loss for multi-class classification
  - Stochastic Gradient Descent (SGD) with configurable learning rate
  - Mini-batch processing for efficient training

- **Key Optimizations**:
  - Memory-efficient tensor operations
  - In-place computations where possible
  - Loop unrolling for critical sections
  - Cache-friendly memory access patterns

- **Performance**:
  - ~90% accuracy on MNIST test set after 5 epochs
  - Fast inference time (<1ms per image on modern CPUs)
  - Minimal memory footprint (~1MB for the entire network)

---

## Educational Sections

### Layer Structure
A CNN is organized as a sequence of specialized layers, each transforming the data in a unique way:
1. **Convolutional Layer:** The first layer receives the raw 28×28 pixel image. It applies 8 different 5×5 filters, each scanning the image and producing a feature map that highlights certain patterns (like edges or shapes). Each filter is trained to detect a different feature.
2. **ReLU Activation:** After convolution, the ReLU function is applied to every value, replacing negatives with zero. This introduces non-linearity, enabling the network to model complex relationships and ignore uninformative signals.
3. **Pooling Layer:** Each of the 8 feature maps is downsampled using 2×2 max-pooling. This reduces the width and height by half, which shrinks computation and makes the network less sensitive to small shifts or distortions in the input.
4. **Flatten Layer:** The pooled feature maps (now 8 smaller 2D arrays) are flattened into a single 1D vector, preparing the data for the dense layer.
5. **Dense (Fully-Connected) Layer:** This layer connects every input from the flattened vector to each of the 10 output neurons (one per digit). It combines all learned features to make a final prediction.
6. **Softmax:** The final scores are passed through the softmax function, which turns them into probabilities that sum to 1, making it easy to interpret the network’s confidence for each digit.

### Convolutions
Convolution is a mathematical operation where a small filter (kernel) is slid across the input image. At each position, the filter multiplies its weights by the underlying pixel values and sums the result, producing a single output value. This process is repeated for all positions, generating a feature map. In this project:
- Each filter is 5×5 and initialized with random values (He initialization).
- The same filter weights are used everywhere in the image (weight sharing), allowing the network to detect the same pattern regardless of its position.
- Multiple filters are used in parallel, so the network can learn to detect many different features.
- The output of the convolution is a set of feature maps, each highlighting where a specific feature appears in the image.
- This operation is efficient and helps the network generalize well to new data.

### Pooling
Pooling layers reduce the size of feature maps while preserving the most important information. Max-pooling, used here, divides each feature map into non-overlapping 2×2 blocks and takes the maximum value from each block. This has several benefits:
- Reduces the number of parameters and computations in the network.
- Provides a form of translation invariance, so small shifts in the input do not drastically change the output.
- Helps prevent overfitting by summarizing the presence of features in local patches rather than their exact locations.
- In this project, after pooling, each feature map is half as wide and tall, making the network faster and more robust.

### Backpropagation
Backpropagation is the learning algorithm that enables neural networks to improve with experience. It works by computing how much each weight in the network contributed to the final error (loss), then adjusting the weights to reduce that error. In this project:
- After a forward pass, the loss is computed using cross-entropy, comparing the predicted probabilities to the true label.
- Gradients of the loss are computed layer by layer, moving backwards from the output to the input (hence "back" propagation).
- The gradients for each parameter (filter weights, dense weights, biases) are calculated explicitly in C, making the process transparent and educational.
- Each weight is updated using Stochastic Gradient Descent: new_weight = old_weight - learning_rate × gradient.
- This iterative process gradually tunes the network to make more accurate predictions over time.

---

## Quick Start
```bash
# clone the repo
$ git clone https://github.com/<your-user>/CNN-main.git && cd CNN-main/CNN-main

# build (works on Linux, macOS, WSL, MinGW, etc.)
$ gcc -Wall -Wextra -O3 main.c lib/*.c -o cnn -lm

# run
$ ./cnn
```
> For Windows users with MSVC, replace the `gcc` call with the equivalent `cl` command.

## Installation
1. Ensure you have a C compiler (GCC ≥ 9 or MSVC ≥ 2019).
2. (Optional) Download the MNIST dataset into the `MNIST/` folder *(see below).*
3. Compile:
   ```bash
   gcc -Wall -Wextra -g -O3 main.c lib/*.c -o cnn -lm
   ```

## Usage
```
./cnn [epochs] [learning_rate]
# Defaults: epochs=5, lr=0.005
```
Example:
```
./cnn 3 0.01
```

During training you will see per-epoch loss & accuracy printed to stdout.

## Dataset
The code expects the four raw MNIST ubyte files inside the local `MNIST/` directory:
* `train-images-idx3-ubyte`
* `train-labels-idx1-ubyte`
* `t10k-images-idx3-ubyte`
* `t10k-labels-idx1-ubyte`

If they are missing, download them from Yann LeCun’s website:
<http://yann.lecun.com/exdb/mnist/>

## Results
| Epochs | Learning Rate | Average Loss | Accuracy |
|-------:|--------------:|-------------:|---------:|
| 1      | 0.005         | 0.4156       | 86 %     |
| 5      | 0.005         | 0.2888       | 91 %     |

*(Your mileage may vary based on compiler optimizations & CPU.)*

## Project Structure

### Core Files
- **`main.c`** - Entry point that initializes the network, loads MNIST data, and runs the training loop.
- **`lib/convolution.c`** - Implements 2D convolution with He-initialized filters, forward pass, and backpropagation through the convolutional layer.
- **`lib/pooling.c`** - Handles 2×2 max-pooling operations that reduce spatial dimensions while preserving important features.
- **`lib/dense.c`** - Fully-connected layer implementation with weight matrices and bias terms, including forward pass and gradient updates.
- **`lib/backprop.c`** - Contains backpropagation logic, gradient calculations, and weight updates for both convolutional and dense layers.
- **`lib/import.c`** - Loads MNIST dataset files (IDX format) and converts them into usable in-memory arrays with proper normalization.

### Header Files (in `lib/`)
- **`convolution.h`** - Defines the ConvLayer struct and function prototypes for convolution operations.
- **`pooling.h`** - Interface for max-pooling functionality.
- **`dense.h`** - Dense layer structure and function declarations.
- **`output.h`** - Softmax activation and cross-entropy loss calculations.
- **`import.h`** - MNIST data loading function declarations.

### Data
- **`MNIST/`** - Directory containing the MNIST dataset files (not included in repo):
  - `train-images-idx3-ubyte` - Training images
  - `train-labels-idx1-ubyte` - Training labels
  - `t10k-images-idx3-ubyte` - Test images
  - `t10k-labels-idx1-ubyte` - Test labels

## Contributing
Contributions are very welcome! Feel free to open issues or PRs for:
- bug fixes
- performance improvements
- portability fixes (Windows/MSVC, ARM, etc.)
- documentation enhancements

Please follow the existing code style: 4-space indentation, snake_case variables, and plenty of comments.

## License
This project is released under the **Unlicense**