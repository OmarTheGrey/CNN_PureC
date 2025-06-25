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

*Written for learning & fun – not production.*

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
```
CNN-main/
├── MNIST/            # dataset files (not committed)
├── lib/              # CNN layer & utility implementations
│   ├── convolution.c
│   ├── pooling.c
│   ├── dense.c
│   ├── backprop.c
│   ├── import.c      # reads MNIST ubyte format
│   └── ...
├── main.c            # program entry – sets up network & training loop
└── README.md         # you are here
```

## Contributing
Contributions are very welcome! Feel free to open issues or PRs for:
- bug fixes
- performance improvements
- portability fixes (Windows/MSVC, ARM, etc.)
- documentation enhancements

Please follow the existing code style: 4-space indentation, snake_case variables, and plenty of comments.

## License
This project is released under the **Unlicense**