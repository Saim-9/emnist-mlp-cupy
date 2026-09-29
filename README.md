# EMNIST Neural Network From Scratch (NumPy/CuPy)

An attempt to learn how neural networks actually work by building one from scratch — no PyTorch, no TensorFlow, no Keras. Everything is implemented manually using CuPy, so training runs on GPU, with a EMNIST data pipeline and training loop built around it.

## What's Implemented

- Dense (fully connected) layers, with forward and backward passes
- ReLU and Softmax activations
- Categorical cross-entropy loss, with a combined Softmax + Cross-Entropy backward pass 
- Dropout 
- L1 and L2 regularization
- Adam optimizer (with bias correction and learning rate decay)
- SGD optimizer (with momentum)

All operations run on CuPy arrays (GPU).

## Training on EMNIST

Trained on the EMNIST-balanced dataset (47 handwritten character classes).

**Data pipeline:** pixel values normalized to `[0, 1]`, with a stratified validation split (10% per class) so every class is represented proportionally in training and validation.

**Architecture:**

```
Input (784)  →  Dense(784 → 512)  →  ReLU  →  Dropout
             →  Dense(512 → 256)  →  ReLU  →  Dropout
             →  Dense(256 → 47)   →  Softmax + Cross-Entropy
```

- Optimizer: Adam (learning rate 0.001, decay 1e-5)
- Batch size: 128
- Regularization: L2 weight decay (1e-4) + dropout (0.2), tuned by comparing train/validation accuracy gap across several configurations
- Dropout disabled during validation/test evaluation

**Final test accuracy: ~86%**

An MLP on EMNIST-balanced is expected to plateau in the mid-to-high 80s. Since this is a plain MLP, it flattens the image into 784 independent pixel values and has no way to represent spatial structure, which is why the acc of MLPs plateaus in the mid-to-high 80s.Convolutional networks outperform MLPs on image data, a natural next step beyond this project, which i may or may not implement.

## Requirements

- Python 3.10+
- [CuPy](https://cupy.dev/) (matched to your CUDA version, e.g. `cupy-cuda12x`) — requires an NVIDIA GPU
- NumPy

```bash
pip install numpy cupy-cuda12x
```

## Dataset

Not included in this repo (the CSVs exceed GitHub's file size limits). Download `emnist-balanced-train.csv` and `emnist-balanced-test.csv` (e.g. from [Kaggle's EMNIST dataset](https://www.kaggle.com/datasets/crawford/emnist)) and place both in the project folder.

## How to Run

1. Clone this repository.
2. Install the requirements above.
3. Download the EMNIST-balanced CSV files into the project folder.
4. Open `emnist-nn.ipynb` and run the cells in order: load/preprocess data, create the validation split, move data to GPU, run training, then run the final test evaluation cell once.

## What I Learned

- How forward and backward propagation work at the level of individual matrix operations, across dense layers, activations, and loss functions.
- How dropout and L1/L2 regularization actually work, and how they affect training versus how they're implemented under the hood.
- How to diagnose overfitting versus underfitting from the train/validation gap, and tune regularization through controlled experiments rather than guessing.
- Why the loss value being optimized and the loss value being logged can diverge, since the optimizer only ever acts on gradients.
- Why a held-out test set has to stay untouched during tuning, and what it actually confirms once training is done.
- The architectural ceiling of MLPs on image data, and why convolutional networks are the natural next step.


Many of the foundational concepts here were learned from the NNFS book. 
---

Built by [Saim-9](https://github.com/Saim-9)
