# 🖥️ Computer Vision Lab 6-1 — Perceptrons, PyTorch & the Training Loop

## 🎯 Objectives

In this lab, you will:

1. In this lab, we will be introduced to the **single-layer perceptron**, and extend it to a **multi-layer perceptron (MLP)**.
2. Learn **PyTorch** fundamentals: tensors, `autograd`, and how a gradient actually flows backward and updates a weight verified by hand against the chain rule.
3. Write a **`dataset.py`**: a `Dataset` subclass from scratch, for the IRIS dataset.
4. Write a **`model.py`**: single-layer and multi-layer perceptrons, and read the `forward()` pass end to end.
5. Write a **`train.py`**: the training loop, **TensorBoard** logging, **checkpointing** (`last.pt` / `best.pt`), and **early stopping**.
6. Train both models on IRIS and compare what the hidden layer actually buys you.
7. Implement a perceptron **from scratch in NumPy**, and verify your hand-derived gradients against PyTorch's.

> ⚠️ **Note on this lab's shift:** Labs 1–5 were entirely **classical** — every feature was hand-designed by you, and every "model" was fit by scikit-learn in one call. From this lab on, the model **learns its own weights by gradient descent**, and you own the training loop.

> 📌 **Lab 6 comes in two parts.** This sheet (6-1) teaches the PyTorch machinery on **IRIS** — a 150-sample, 4-feature dataset small enough that nothing hides behind long runtimes. [Lab 6-2](./labsheet6-2.md) then applies that same machinery to **images**: converting MNIST into LBP features, comparing them against raw pixels, and tuning the training mechanics. **6-2 imports the `dataset.py`, `model.py` and `train.py` you write here, unchanged** — so finish this sheet first.

> 💡 **Why IRIS, in a computer vision course?** Precisely *because* it isn't images. This lab is about the training loop itself — autograd, optimizers, checkpointing — and IRIS lets you see all of it clearly without waiting on feature extraction or wondering whether a bad result is your model or your features. The images arrive in 6-2, once the machinery is second nature.

> 💡 A companion notebook, [`labsheet6-1.ipynb`](./labsheet6-1.ipynb), contains everything below in runnable order. Set up the environment and kernel as in Section 0, then work through the notebook.

---

## 0️⃣ Prerequisites

### Step 1 — Create the `cv-dl` environment

Following the same pattern as Lab 1's `cv_gui`, deep-learning work gets its **own** environment rather than being piled into `cv-env`:

```bash
conda create -n cv-dl python=3.10
conda activate cv-dl

# CPU-only PyTorch is entirely sufficient for this lab - the models are tiny.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Everything else this lab uses, including Jupyter
pip install scikit-learn scikit-image matplotlib tensorboard opencv-python jupyter ipykernel
```

> ⚠️ **Run those two `pip install` lines separately, exactly as written.** The first uses `--index-url` to point pip at PyTorch's own package server. That flag **replaces** PyPI for the whole command, so if you append `tensorboard` or `scikit-learn` to it, pip will look for them on PyTorch's server, not find them, and the entire install aborts.

> 💡 **Why CPU-only?** The biggest model in this lab is a 160→128→10 MLP. On a dataset this size it trains in seconds on a CPU, and the GPU wheels are several GB. If you already have a CUDA-capable GPU set up, the code below is device-agnostic and will use it — but you gain almost nothing here.

### Step 2 — Register the environment as a Jupyter kernel

Installing Jupyter *into* the environment is not enough — the notebook also has to be told that this environment exists as a kernel it can run against:

```bash
conda activate cv-dl
python -m ipykernel install --user --name cv-dl --display-name "Python (cv-dl)"
```

You only ever do this once per environment.

### Step 3 — Open the notebook and select the kernel

```bash
conda activate cv-dl
cd <wherever you cloned this repo>/labsheets
jupyter notebook labsheet6-1.ipynb
```

Then **select the kernel**:

- **Jupyter Notebook / Lab:** menu → *Kernel* → *Change kernel* → **Python (cv-dl)**
- **VS Code:** click the kernel name in the top-right of the notebook → *Select Another Kernel* → *Python Environments* → **cv-dl**

> ⚠️ **This step is the one people skip, and it produces a confusing failure.** If the notebook is running on your base/default kernel instead of `cv-dl`, the very first cell fails with `ModuleNotFoundError: No module named 'torch'` — even though you just installed torch successfully. The packages are fine; the notebook is simply talking to a different Python. Always check the kernel name shown in the corner of the notebook before debugging an import error.

### Step 4 — Verify

Run this as the first cell. The `sys.executable` line is the one that actually proves you are on the right kernel — it should contain `cv-dl` in the path:

```python
import sys
import torch, torchvision, skimage

print("python:     ", sys.executable)        # <- must contain 'cv-dl'
print("torch:      ", torch.__version__)
print("torchvision:", torchvision.__version__)
print("skimage:    ", skimage.__version__)
print("CUDA available:", torch.cuda.is_available())  # False is fine for this lab
```

> 💡 Everything in this lab runs comfortably on a laptop CPU. There is no cloud runtime to disconnect, nothing to re-download each session, and your `checkpoints/` and `runs/` simply stay on disk between sessions.

### Files you will build

Unlike previous labs (one script per exercise), this lab builds a small **project**, because that's how real training code is organized. Lab 6-2 then extends these same files rather than starting over:

```
lab6/
├── dataset.py      # IrisDataset, written from scratch      (Section 3)
├── model.py        # SingleLayerPerceptron, MLP              (Section 4)
├── train.py        # loop, TensorBoard, checkpoints, early stopping  (Section 5)
├── run_lab6.py     # driver: which configs to train          (Section 6)
├── runs/           # TensorBoard logs (one subfolder per run)
└── checkpoints/    # one subfolder per run, each with last.pt and best.pt
```

Each run gets the **same name** in both `runs/` and `checkpoints/`, so a curve you're looking at in TensorBoard and the weights that produced it are trivially matched up:

```
runs/mnist_lbp_mlp_adam_lr1e-3/          <- tensorboard logs
checkpoints/mnist_lbp_mlp_adam_lr1e-3/
├── last.pt                              <- every epoch, for resuming
└── best.pt                              <- only when val loss improves
```

> 💡 In the notebook, these files are created by `%%writefile model.py` cells — the notebook genuinely writes each file to disk and then imports it as a module. So you end up with a real, runnable project directory, not just notebook cells, and the same files work from a plain terminal with `python run_lab6.py`.

---

## 1️⃣ Theory: From Lab 4's SVM to a Perceptron

### You have already trained a single-layer network

In Lab 4 you trained this:

```python
svm = SGDClassifier(loss="hinge", random_state=RANDOM_STATE, learning_rate="optimal")
```

and the labsheet noted that hinge loss trained by stochastic gradient descent **is** a linear SVM, just fit iteratively instead of solved exactly.

Look at what that object actually is. It holds one weight matrix `W` and one bias vector `b`. It predicts by computing `W·x + b` and taking the largest entry. It updates `W` by gradient descent on a loss.

That is a **single-layer perceptron**. You have already built one — it was just wearing a scikit-learn name tag. Everything new in this lab is about (a) owning the training loop instead of calling `.partial_fit()`, and (b) stacking more than one layer.

### Representation 1 — the single-layer perceptron

Start with **one output unit**. It takes every input, multiplies each by its own weight, sums them, adds a bias, and emits one number:

```
   ONE NEURON (output unit j)

        x₁ ●───── w₁ⱼ ─────┐
                            ╲
        x₂ ●───── w₂ⱼ ───────►  ⊕  ────►  zⱼ
                            ╱    ▲
        x₃ ●───── w₃ⱼ ─────┘     │
           ⋮                     bⱼ
        x_d ●──── w_dⱼ ────┘

                 zⱼ = Σᵢ wᵢⱼ·xᵢ + bⱼ
```

A **single-layer perceptron** is just `C` of those units side by side — one per class — all reading the same inputs. Every input connects to every output, which is exactly what "fully connected" (`nn.Linear`) means:

```
   SINGLE-LAYER PERCEPTRON        d inputs  ->  C classes

        INPUT                         OUTPUT (logits)

        x₁ ●──┐
              ├──────────►  ⊕  ──►  z₁  ┐
        x₂ ●──┤                          │
              ├──────────►  ⊕  ──►  z₂  ├──► softmax ──► p ──► argmax ──► ŷ
        x₃ ●──┤                          │
              ├──────────►  ⊕  ──►  z_C ┘
        x_d ●─┘
                W (d×C), b (1×C)
```

**Forward pass.** Per-element, and in matrix form — the matrix form is what the code actually computes, for a whole batch of `N` samples at once:

```
   scalar :  zⱼ = Σᵢ₌₁..d  wᵢⱼ·xᵢ + bⱼ            for j = 1..C

   matrix :  Z = X·W + b
   shapes :  (N×C) = (N×d)·(d×C) + (1×C)

   softmax:  pⱼ = exp(zⱼ) / Σₖ exp(zₖ)           logits -> probabilities
   loss   :  L  = −(1/N) Σₙ log p[n, yₙ]          cross-entropy
```

> 🔎 **This is exactly `nn.Linear(d, C)`.** When you write `nn.Linear(160, 10)` in Section 4, PyTorch allocates precisely the `W` and `b` above (it stores `W` transposed, as `10×160`), and the forward pass computes `X·W + b`. Nothing else is hidden inside it.

### Representation 2 — the multi-layer perceptron

Insert a **hidden layer** between input and output, with a non-linearity after it:

```
   MULTI-LAYER PERCEPTRON       d inputs -> h hidden -> C classes

     INPUT            HIDDEN LAYER              OUTPUT LAYER
                   a⁽¹⁾ = ReLU(z⁽¹⁾)                z⁽²⁾

     x₁ ●──┐         ┌──► ⊕ ─ReLU─► ●──┐
           ├─────────┤                  ├──────►  ⊕  ──► z₁
     x₂ ●──┤         ├──► ⊕ ─ReLU─► ●──┤
           ├─────────┤                  ├──────►  ⊕  ──► z₂
     x₃ ●──┤         ├──► ⊕ ─ReLU─► ●──┤
           ├─────────┤                  ├──────►  ⊕  ──► z_C
     x_d ●─┘         └──► ⊕ ─ReLU─► ●──┘

            W⁽¹⁾ (d×h)                   W⁽²⁾ (h×C)
            b⁽¹⁾ (1×h)                   b⁽²⁾ (1×C)
```

**Forward pass:**

```
   Z⁽¹⁾ = X·W⁽¹⁾ + b⁽¹⁾          (N×h)     hidden pre-activations
   A⁽¹⁾ = ReLU(Z⁽¹⁾)             (N×h)     ReLU(z) = max(0, z)
   Z⁽²⁾ = A⁽¹⁾·W⁽²⁾ + b⁽²⁾       (N×C)     output logits
   P    = softmax(Z⁽²⁾)          (N×C)
```

### Why the ReLU is load-bearing

Remove the non-linearity and the two layers collapse algebraically:

```
   Z⁽²⁾ = (X·W⁽¹⁾ + b⁽¹⁾)·W⁽²⁾ + b⁽²⁾
        =  X·(W⁽¹⁾W⁽²⁾) + (b⁽¹⁾W⁽²⁾ + b⁽²⁾)
        =  X·W′ + b′                          ... a SINGLE linear layer
```

Two stacked `Linear` layers with nothing between them are **exactly as expressive as one**, just slower to train. Internalize this now — it is why every architecture you meet from here on has activations between its layers.

With the ReLU in place, the hidden layer learns an intermediate representation and the output layer draws a linear boundary *in that learned space*, which is a curved boundary back in the original input space.

### Weight calculation: how the weights actually change

Training is one rule applied over and over. Every parameter moves **against** its gradient, scaled by the learning rate `η`:

```
   W ← W − η · ∂L/∂W
   b ← b − η · ∂L/∂b
```

Everything else is working out what `∂L/∂W` is for each layer.

**Single layer (softmax + cross-entropy).** These two combine so that the gradient at the logits is simply "predicted minus true":

```
   Δ      = ∂L/∂Z = (P − Y) / N        Y = one-hot targets,  (N×C)
   ∂L/∂W  = Xᵀ·Δ                        (d×C)
   ∂L/∂b  = Σₙ Δ[n,:]                   (1×C)
```

**Multi-layer — backpropagation.** The same idea applied layer by layer, from the output backwards. Each layer passes a gradient back through its own weights, and the ReLU gates it:

```
   OUTPUT LAYER
   Δ⁽²⁾     = (P − Y) / N                           (N×C)
   ∂L/∂W⁽²⁾ = A⁽¹⁾ᵀ·Δ⁽²⁾                            (h×C)
   ∂L/∂b⁽²⁾ = Σₙ Δ⁽²⁾[n,:]                          (1×C)

   HIDDEN LAYER   (propagate back, then gate through the ReLU)
   Δ⁽¹⁾     = (Δ⁽²⁾·W⁽²⁾ᵀ) ⊙ ReLU′(Z⁽¹⁾)            (N×h)
              where  ReLU′(z) = 1 if z > 0, else 0
                     ⊙         = elementwise product
   ∂L/∂W⁽¹⁾ = Xᵀ·Δ⁽¹⁾                               (d×h)
   ∂L/∂b⁽¹⁾ = Σₙ Δ⁽¹⁾[n,:]                          (1×h)
```

Then apply the same update rule to all four parameter blocks.

> 🔎 **This is precisely what `loss.backward()` computes.** You will verify that claim numerically on one neuron in Section 2, and again on a full layer in this lab's Student Task — where you implement these equations in NumPy and assert your `∂L/∂W` matches PyTorch's `.grad` to floating-point tolerance.

### Worked example: one weight update, by hand

Two inputs, two classes, one sample. `x = [1, 2]`, true class `y = 0`, learning rate `η = 0.1`:

```
   W = [0.10  0.30]      b = [0.0  0.0]
       [0.20  0.40]

   FORWARD
   z₁ = 0.10·1 + 0.20·2 = 0.5
   z₂ = 0.30·1 + 0.40·2 = 1.1
   p  = softmax([0.5, 1.1]) = [0.3543, 0.6457]
   L  = −log(p₁) = −log(0.3543) = 1.0375

   BACKWARD        (true class is 0, so Y = [1, 0])
   Δ = P − Y = [0.3543 − 1,  0.6457 − 0] = [−0.6457,  0.6457]

   ∂L/∂W = xᵀ·Δ = [1] · [−0.6457, 0.6457] = [−0.6457   0.6457]
                  [2]                        [−1.2913   1.2913]
   ∂L/∂b = [−0.6457,  0.6457]

   UPDATE   W ← W − 0.1·∂L/∂W
   W_new = [0.10−0.1(−0.6457)   0.30−0.1(0.6457)]  =  [0.1646   0.2354]
           [0.20−0.1(−1.2913)   0.40−0.1(1.2913)]     [0.3291   0.2709]
   b_new = [0.0646,  −0.0646]
```

Read what happened. The model put only 35% probability on the correct class, so **every weight feeding class 0 went up and every weight feeding class 1 went down**. And `x₂ = 2` moved twice as far as `x₁ = 1` — a larger input means that weight was twice as responsible for the error, so it takes twice the correction.

Verify it against PyTorch yourself. This should print `True`:

```python
import numpy as np
import torch
import torch.nn.functional as F

x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([0])
W = torch.tensor([[0.10, 0.30], [0.20, 0.40]], requires_grad=True)
b = torch.tensor([[0.0, 0.0]], requires_grad=True)

loss = F.cross_entropy(x @ W + b, y)   # softmax + cross-entropy, fused
loss.backward()

print("loss :", round(loss.item(), 4))                      # 1.0375
print("dL/dW:\n", W.grad.numpy())
print("dL/db:", b.grad.numpy())

expected_dW = np.array([[-0.6457, 0.6457], [-1.2913, 1.2913]])
print("matches the hand calculation:",
      np.allclose(W.grad.numpy(), expected_dW, atol=1e-4))
```

### Single vs multi layer, summarized

|                     | Single-layer perceptron                   | Multi-layer perceptron (MLP)                     |
| ------------------- | ----------------------------------------- | ------------------------------------------------ |
| Parameters          | one`W`, one `b`                           | one`W`, `b` per layer                            |
| Decision boundary   | linear (hyperplane)                       | non-linear                                       |
| Needs activations?  | no (output layer only)                    | **yes**, or it collapses to a single layer       |
| Dropout applicable? | no (nothing to drop between)              | yes, between hidden layers                       |
| Equivalent to       | logistic / softmax regression, linear SVM | —                                                |

**In this lab, you will train on both types of networks and compare.** When a single layer already does nearly as well as an MLP, that tells you the problem is close to linearly separable in whatever feature space you built — which is itself a useful finding about your *features*, not just your model.

---

## 2️⃣ PyTorch Essentials: Tensors, Autograd, and the Update

### Tensors

A `torch.Tensor` is a NumPy array that additionally knows how to compute its own gradients. Most NumPy idioms carry straight over:

```python
import torch
import numpy as np

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.zeros(2, 3)
c = torch.from_numpy(np.array([1.0, 2.0]))   # shares memory with the NumPy array

print(a.shape, a.dtype, a.device)
print(a.numpy())          # back to NumPy (CPU tensors only)
```

Two dtype rules that cause most beginner errors in this lab:

- Features must be **`float32`**. NumPy defaults to `float64`; `nn.Linear` expects `float32` and will raise a dtype mismatch.
- Class labels must be **`int64`** (`torch.long`). `nn.CrossEntropyLoss` wants **class indices**, not one-hot vectors, and not floats.

### Autograd: watch a gradient get computed

This is the single most important cell in this lab. We build a single neuron, run it forward, call `.backward()`, and then check PyTorch's answer against the chain rule computed by hand.

```python
import torch

# One neuron: two inputs, two weights, one bias. requires_grad=True tells PyTorch
# to record every operation involving these tensors, so it can walk backward later.
x = torch.tensor([2.0, 3.0])
w = torch.tensor([0.5, -1.0], requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# ---- forward pass ----
z = (w * x).sum() + b      # z = w1*x1 + w2*x2 + b
y_pred = torch.sigmoid(z)
y_true = torch.tensor(1.0)
loss = (y_pred - y_true) ** 2

print(f"z       = {z.item():.6f}")
print(f"y_pred  = {y_pred.item():.6f}")
print(f"loss    = {loss.item():.6f}")

# ---- backward pass ----
loss.backward()            # populates w.grad and b.grad

print(f"w.grad  = {w.grad}")
print(f"b.grad  = {b.grad.item():.6f}")
```

Now, the same gradient by hand. The chain rule for this graph is:

```
dloss/dw = dloss/dy_pred  ×  dy_pred/dz  ×  dz/dw
         = 2(y_pred - y_true)  ×  y_pred(1 - y_pred)  ×  x
```

```python
with torch.no_grad():
    manual_dz = 2 * (y_pred - y_true) * y_pred * (1 - y_pred)
    manual_dw = manual_dz * x
    manual_db = manual_dz

print(f"manual dw = {manual_dw}")
print(f"manual db = {manual_db.item():.6f}")

assert torch.allclose(w.grad, manual_dw), "gradients disagree!"
assert torch.allclose(b.grad, manual_db), "gradients disagree!"
print("autograd matches the hand-computed chain rule")
```

That assert passing is the whole point: `loss.backward()` is not magic, it is the chain rule applied mechanically over the graph of operations you just wrote. Everything else in this lab is the same step, repeated over bigger graphs.

### The update step

A gradient tells you which direction increases the loss. So step the *opposite* way, scaled by the learning rate:

```python
LR = 0.1
with torch.no_grad():          # weight updates must not themselves be recorded
    w -= LR * w.grad
    b -= LR * b.grad
```

In real code an **optimizer** does this for you, which also lets you swap SGD for Adam without touching the loop.

### The five-line ritual

Every training loop you will ever write in PyTorch contains these five lines, in this order:

```python
optimizer.zero_grad()          # 1. clear gradients from the previous step
logits = model(features)       # 2. forward pass
loss = criterion(logits, labels)   # 3. compute the loss
loss.backward()                # 4. backward pass -> fills .grad on every parameter
optimizer.step()               # 5. apply the update using those .grad values
```

> ⚠️ **Why `zero_grad()` exists, and the bug you get without it.** PyTorch **accumulates** gradients into `.grad` rather than overwriting them. If you forget `zero_grad()`, step 4 *adds* this batch's gradient to every previous batch's gradient, so your effective gradient keeps growing without bound. The model still trains — badly, erratically, and often it converges to something mediocre instead of crashing. There is no error message. This is the most common PyTorch bug, and the fact that it fails *silently* is exactly what makes it worth memorizing now.
>
> (The accumulation behaviour is deliberate, not an oversight — it's what lets you simulate a large batch on a small GPU by summing gradients over several small batches before stepping once.)

### `model.train()` vs `model.eval()`, and `torch.no_grad()`

Two different things that are easy to confuse:

- **`model.train()` / `model.eval()`** switch the *behaviour of certain layers*. Dropout is active in `train()` and disabled in `eval()`. (BatchNorm, which we don't use here, also changes behaviour.) For a model with no dropout, these do nothing at all — which is why forgetting them stays invisible until you add dropout in Lab 6-2.
- **`torch.no_grad()`** stops PyTorch recording the computation graph. Use it around all evaluation and inference: it's faster and uses less memory. It does **not** disable dropout.

You need both when evaluating:

```python
model.eval()
with torch.no_grad():
    logits = model(x_val)
```

---

## 3️⃣ `dataset.py` — Writing a `Dataset` from Scratch

### Why `Dataset` and `DataLoader` exist

PyTorch splits data handling into two objects with cleanly separated jobs:

- **`Dataset`** answers exactly two questions: *how many samples are there* (`__len__`) and *give me sample number `i`* (`__getitem__`). It knows nothing about batching or shuffling.
- **`DataLoader`** wraps a `Dataset` and handles batching, shuffling, and parallel loading with worker processes. It knows nothing about your data format.

That separation is why you can swap IRIS for MNIST without touching your training loop at all — which is exactly what you'll do in Lab 6-2.

### IRIS

IRIS is 150 flowers, 4 numeric measurements each (sepal length/width, petal length/width), 3 species. It is not a vision dataset and it ships with scikit-learn, not torchvision — which makes it perfect for this section, because there is no ready-made loader and you have to write the `Dataset` yourself.

It's also a useful case for the single-vs-multi-layer question: `setosa` is cleanly linearly separable from the other two, while `versicolor` and `virginica` overlap slightly. So a single layer gets *most* of the way, and you can see exactly how much the hidden layer buys.

```python
"""dataset.py - IRIS"""

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class IrisDataset(Dataset):
    """A Dataset over in-memory feature/label arrays.

    This is the minimal, complete Dataset: store the data, answer __len__,
    answer __getitem__. Everything else (batching, shuffling) is the
    DataLoader's job, not ours.
    """

    def __init__(self, features, labels):
        # float32 for the features (what nn.Linear expects) and int64 for the
        # labels (what nn.CrossEntropyLoss expects as CLASS INDICES).
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def build_iris_datasets(val_size=0.2, test_size=0.2, random_state=42):
    """Returns (train_ds, val_ds, test_ds, scaler, class_names, n_features, n_classes).

    Note the three-way split: train to fit weights, val to choose the best
    checkpoint and to trigger early stopping, test touched ONCE at the end.
    """
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_size + test_size, stratify=y, random_state=random_state
    )
    relative_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, stratify=y_temp, random_state=random_state
    )

    # Fit the scaler on TRAIN ONLY, then apply it to val/test - exactly the same
    # rule as Lab 4's "fit the visual vocabulary on training descriptors only".
    # Fitting on all the data leaks val/test statistics into training.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (
        IrisDataset(X_train, y_train),
        IrisDataset(X_val, y_val),
        IrisDataset(X_test, y_test),
        scaler,
        # Plain python str, NOT numpy.str_ - these get written into the
        # checkpoint, and torch.load's weights_only=True default rejects
        # numpy scalar types. See the checkpointing warning in Section 6.
        [str(name) for name in data.target_names],
        X_train.shape[1],   # 4 features
        len(data.target_names),  # 3 classes
    )
```

> ⚠️ **The scaler is part of your model.** If you save `best.pt` but lose the `StandardScaler` that was fitted alongside it, the checkpoint is useless — feeding raw centimetres into weights trained on standardized inputs produces confident nonsense. Section 5 saves the scaler **inside** the checkpoint for exactly this reason. This is the direct descendant of Lab 4's `joblib.dump(scaler, "caltech256_scaler.joblib")`.

---

## 4️⃣ `model.py` — Single-Layer and Multi-Layer Perceptrons

Every PyTorch model subclasses `nn.Module` and implements two methods: `__init__` (declare the layers that hold parameters) and `forward` (say how data flows through them). You never call `forward()` yourself — you call `model(x)`, and `nn.Module.__call__` dispatches to `forward()` after running PyTorch's internal hooks.

```python
"""model.py"""

import torch.nn as nn


class SingleLayerPerceptron(nn.Module):
    """One nn.Linear, nothing else: logits = W·x + b.

    This is the direct PyTorch equivalent of Lab 4's SGDClassifier - one
    weight matrix, one bias, a linear decision boundary.
    """

    def __init__(self, in_features, num_classes):
        super().__init__()          # ALWAYS call this first in an nn.Module
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: (batch_size, in_features)  ->  returns (batch_size, num_classes)
        # Returns RAW LOGITS. No softmax here - see the warning below.
        return self.fc(x)


class MLP(nn.Module):
    """A multi-layer perceptron: one or more hidden layers, each followed by
    a ReLU non-linearity and (optionally) dropout."""

    def __init__(self, in_features, num_classes, hidden_sizes=(128,), dropout=0.0):
        super().__init__()

        layers = []
        prev = in_features
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            # Without a non-linearity here, stacked Linear layers collapse
            # algebraically into a single Linear layer (Section 1).
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden

        # Output layer: no activation, no dropout - raw logits.
        layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(name, in_features, num_classes, hidden_sizes=(128,), dropout=0.0):
    if name == "single":
        return SingleLayerPerceptron(in_features, num_classes)
    if name == "mlp":
        return MLP(in_features, num_classes, hidden_sizes, dropout)
    raise ValueError(f"Unknown model: {name}")
```

> ⚠️ **Do not put a softmax in `forward()`.** `nn.CrossEntropyLoss` already applies `LogSoftmax` internally — it is `LogSoftmax + NLLLoss` fused into one numerically-stable operation. If you apply softmax yourself *and* use `CrossEntropyLoss`, you get a **double softmax**: the gradients are squashed, the model trains far more slowly and plateaus at a worse accuracy — but **it does not error, and it does not obviously look broken**. It just quietly underperforms. Return raw logits; apply softmax only when you explicitly want probabilities to display (as in Lab 6-2's `predict.py`).

### Reading the forward pass end to end

Trace a batch through `MLP(160, 10, hidden_sizes=(128,), dropout=0.2)` — a 160-d input and 10 classes, which is the shape you'll meet in Lab 6-2 (IRIS is simply 4-d in, 3 classes out):

| Step | Operation               | Shape in     | Shape out    | Parameters                  |
| ---- | ----------------------- | ------------ | ------------ | --------------------------- |
| 1    | input batch             | —           | `(B, 160)` | —                          |
| 2    | `nn.Linear(160, 128)` | `(B, 160)` | `(B, 128)` | `W: 128×160`, `b: 128` |
| 3    | `nn.ReLU()`           | `(B, 128)` | `(B, 128)` | none                        |
| 4    | `nn.Dropout(0.2)`     | `(B, 128)` | `(B, 128)` | none (train only)           |
| 5    | `nn.Linear(128, 10)`  | `(B, 128)` | `(B, 10)`  | `W: 10×128`, `b: 10`   |
| 6    | output logits           | `(B, 10)`  | —           | —                          |

Total: `128×160 + 128 + 10×128 + 10` = `20,480 + 128 + 1,280 + 10` = **21,898** learnable parameters. Don't take that on trust — check it yourself:

```python
model = MLP(160, 10, hidden_sizes=(128,), dropout=0.2)
print(model)
print("trainable parameters:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))
```

Steps 2–5 happen forward when you call `model(x)`. `loss.backward()` then walks the exact same chain **in reverse** — step 5, then 4, then 3, then 2 — depositing a gradient into `.grad` on each of those weight matrices and bias vectors along the way. `optimizer.step()` then reads those `.grad` values and updates each parameter. That is the entire end-to-end path, and it is the same chain rule you verified by hand in Section 2.

---

## 5️⃣ `train.py` — Training Loop, TensorBoard, Checkpointing, Early Stopping

One parameterized training function, so every experiment in Lab 6-2 is a config change rather than a copy-pasted loop.

```python
"""train.py - one parameterized training run, logged to TensorBoard and checkpointed."""

import os
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import build_model


class OneHotMSELoss(nn.Module):
    """MSE against a one-hot target. Included deliberately as the WRONG tool
    for classification, so you can watch it underperform in Lab 6-2."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.mse = nn.MSELoss()

    def forward(self, logits, targets):
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)
        return self.mse(torch.softmax(logits, dim=1), one_hot)


def build_criterion(name, num_classes):
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()      # expects RAW LOGITS + class indices
    if name == "hinge":
        return nn.MultiMarginLoss()       # multiclass hinge == Lab 4's linear SVM
    if name == "mse":
        return OneHotMSELoss(num_classes)
    raise ValueError(f"Unknown loss: {name}")


def build_optimizer(name, params, lr, weight_decay):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run the model over a loader and return (mean_loss, accuracy).

    model.eval() matters here: it disables dropout. @torch.no_grad() is a
    separate thing - it stops the graph being recorded (faster, less memory).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def train(config, train_ds, val_ds, test_ds=None, scaler=None, class_names=None):
    """Train one configuration end to end. Returns a summary dict."""

    run_name = config["run_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = os.path.join("runs", run_name)
    ckpt_dir = os.path.join("checkpoints", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    model = build_model(
        config["model"], config["in_features"], config["num_classes"],
        hidden_sizes=config.get("hidden_sizes", (128,)),
        dropout=config.get("dropout", 0.0),
    ).to(device)

    criterion = build_criterion(config["loss"], config["num_classes"])
    optimizer = build_optimizer(
        config["optimizer"], model.parameters(),
        config["lr"], config.get("weight_decay", 0.0),
    )

    # Log the model graph to TensorBoard's GRAPHS tab - a clickable picture of
    # the forward pass you traced by hand in Section 4.
    sample_batch = next(iter(train_loader))[0].to(device)
    writer.add_graph(model, sample_batch)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = -1
    best_state = None
    patience_counter = 0
    start_time = time.perf_counter()

    for epoch in range(1, config["epochs"] + 1):
        # ---------------- train one epoch ----------------
        model.train()                      # dropout ON
        running_loss, correct, total = 0.0, 0, 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            # ---- the five-line ritual (Section 2) ----
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)

            # Optional L1 regularization. Unlike L2 (which lives in the
            # optimizer as weight_decay), L1 is not built in - you just add
            # the term to the loss, because a loss is only an expression
            # autograd knows how to differentiate.
            l1_lambda = config.get("l1_lambda", 0.0)
            if l1_lambda > 0:
                loss = loss + l1_lambda * sum(p.abs().sum() for p in model.parameters())

            loss.backward()
            optimizer.step()
            # ------------------------------------------

            running_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # ---------------- TensorBoard ----------------
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("generalization_gap", train_acc - val_acc, epoch)

        # Weight and GRADIENT histograms. .grad still holds the last batch's
        # gradients at this point - zero_grad() only runs at the top of the
        # next iteration. The gradient histograms are what let you SEE a
        # learning rate explode or a gradient vanish in Lab 6-2.
        for name, param in model.named_parameters():
            writer.add_histogram(f"weights/{name}", param, epoch)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad, epoch)

        # ---------------- checkpointing ----------------
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            # The optimizer state carries momentum / Adam moment estimates.
            # Resume without it and the loss visibly jumps as they rebuild.
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": config,
            "class_names": class_names,
            # Save the fitted scaler WITH the weights - see Section 3's warning.
            "scaler_mean": None if scaler is None else scaler.mean_.tolist(),
            "scaler_scale": None if scaler is None else scaler.scale_.tolist(),
        }
        torch.save(checkpoint, os.path.join(ckpt_dir, "last.pt"))

        # ---------------- best checkpoint + early stopping ----------------
        # NOTE: selection is on VALIDATION loss, never training loss. Val loss
        # and val accuracy can disagree - loss often starts rising from
        # overconfidence while accuracy still creeps up. Pick one, state which.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc      # accuracy AT the best-loss epoch
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(checkpoint, os.path.join(ckpt_dir, "best.pt"))
            marker = "  <- best"
        else:
            patience_counter += 1
            marker = ""

        print(f"[{run_name}] epoch {epoch:3d}/{config['epochs']}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}{marker}")

        if patience_counter >= config.get("patience", config["epochs"]):
            print(f"[{run_name}] early stopping at epoch {epoch} "
                  f"(no val improvement for {patience_counter} epochs). "
                  f"best.pt is from epoch {best_epoch}.")
            break

    elapsed = time.perf_counter() - start_time

    # ---------------- final test evaluation, using the BEST weights ----------------
    # Restore best.pt rather than keeping whatever the last epoch produced -
    # this is the entire reason for tracking a best checkpoint at all.
    test_loss = test_acc = None
    if test_ds is not None and best_state is not None:
        model.load_state_dict(best_state)
        test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"[{run_name}] TEST (from best.pt, epoch {best_epoch}): "
              f"loss={test_loss:.4f} acc={test_acc:.4f}")

    # hparams tab: makes every run across BOTH halves of Lab 6 comparable
    # in one sortable table
    writer.add_hparams(
        {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))},
        {"hparam/best_val_loss": best_val_loss,
         "hparam/best_val_acc": best_val_acc,
         "hparam/test_acc": test_acc if test_acc is not None else 0.0},
    )
    writer.close()

    return {
        "run_name": run_name, "best_epoch": best_epoch,
        "best_val_loss": best_val_loss, "best_val_acc": best_val_acc,
        "test_acc": test_acc, "seconds": elapsed,
    }
```

### Resuming from a checkpoint

```python
def load_checkpoint(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt
```

> ⚠️ **Save a dict, never the model object.** `torch.save(model)` pickles the class definition itself, so the file breaks the moment you edit `model.py`. Saving `state_dict()` saves only the tensors, which load into any compatible architecture.
>
> ⚠️ **`torch.load` defaults to `weights_only=True`** in PyTorch 2.6+. Plain primitives (ints, floats, strings, lists, tensors) load fine, which is why the config above sticks to them and stores the scaler as two lists rather than pickling the `StandardScaler` object. If you add something exotic to the checkpoint, you'll need `weights_only=False` — and should only do that for files you produced yourself.

---

## 6️⃣ Running It: IRIS, Single Layer vs MLP

```python
"""run_lab6.py - minimal driver"""

from dataset import build_iris_datasets
from train import train

iris_train, iris_val, iris_test, scaler, iris_classes, n_feat, n_cls = build_iris_datasets()

base_iris = dict(
    in_features=n_feat, num_classes=n_cls, batch_size=16, epochs=100,
    loss="cross_entropy", optimizer="adam", lr=1e-2, weight_decay=0.0,
    patience=15,
)

train({**base_iris, "run_name": "iris_single", "model": "single"},
      iris_train, iris_val, iris_test, scaler, iris_classes)

train({**base_iris, "run_name": "iris_mlp", "model": "mlp", "hidden_sizes": (32,)},
      iris_train, iris_val, iris_test, scaler, iris_classes)
```

### Viewing TensorBoard

Inline in the notebook (run this in a cell):

```python
%load_ext tensorboard
%tensorboard --logdir runs
```

Or as a standalone server, in a second terminal with `cv-dl` active:

```bash
tensorboard --logdir runs
# then open http://localhost:6006
```

Tabs worth knowing: **SCALARS** (loss/accuracy curves, all runs overlaid — tick the runs in the left sidebar to compare), **GRAPHS** (the forward pass from `add_graph`), **HISTOGRAMS/DISTRIBUTIONS** (weights and gradients per epoch), **HPARAMS** (every run as one sortable row).

### Task

1. Run both configurations. For each, report the best epoch, best validation loss, and test accuracy.
2. **Single vs multi layer.** How much does the hidden layer actually buy you on IRIS? IRIS is a useful case here: `setosa` is cleanly linearly separable from the other two species, while `versicolor` and `virginica` overlap slightly — so a single layer should get *most* of the way on its own.

   > ⚠️ **IRIS's test split is only 30 samples**, so a single sample is worth 3.3% of the score. It is entirely normal for the MLP to score *below* the single-layer model on IRIS test accuracy in a given run — that is noise, not evidence that the hidden layer hurt. Look at the validation curves in TensorBoard, and re-run with a few different `random_state` values before concluding anything. **Knowing when a difference is too small to mean anything is part of the skill.**
   >
3. **Watch the checkpointing work.** Note which epoch `best.pt` came from versus which epoch training actually stopped at. The gap between those two numbers is exactly what early stopping plus best-checkpointing bought you — without it you would have shipped the final epoch's weights, which were measurably worse on validation.
4. Open the **GRAPHS** tab in TensorBoard and compare the single-layer and MLP graphs against the forward pass you traced by hand in Section 4. Then open **HISTOGRAMS** and watch how the weight distributions move over training.
5. Deliberately break it: comment out `optimizer.zero_grad()` in `train.py` and re-run. The model still trains — note how the loss curve differs, and how easy this would have been to miss. Put the line back afterwards.

---

## 🧪 Student Task (Evaluative) — A Perceptron from Scratch in NumPy

You have now used `loss.backward()` and `optimizer.step()` many times. This task is about proving to yourself that you could have written them.

**Implement a single-layer perceptron entirely in NumPy** — no `torch.nn`, no autograd, no optimizer — and train it on IRIS.

You must write, by hand:

1. **Forward pass** — `z = X·W + b`, then softmax to get class probabilities.
2. **Loss** — cross-entropy between the predicted probabilities and the one-hot true labels.
3. **Backward pass** — the analytic gradients `dL/dW` and `dL/db`. (For softmax + cross-entropy together, `dL/dz` simplifies to `(probabilities - one_hot) / batch_size` — derive this, don't just use it. The derivation is the point.)
4. **Update** — `W -= lr * dW`, `b -= lr * db`.
5. A **mini-batch training loop** over the same IRIS split your PyTorch version used.

**Requirements:**

1. **Verify your gradients against PyTorch.** Build a `SingleLayerPerceptron` with the *same* initial weights as your NumPy version, run one forward and backward pass on the *same* batch, and assert that your hand-computed `dW` and `db` match `model.fc.weight.grad` and `model.fc.bias.grad` to within floating-point tolerance. **Show this assert passing** — this is the core deliverable, and it is the same verification you did on one neuron in Section 2, now at full layer scale.
2. Train your NumPy perceptron to convergence on IRIS and plot its training loss curve beside the PyTorch `iris_single` run's curve from Section 6. They should be close; explain any difference you see (weight initialization, batch ordering, and floating-point accumulation are the usual suspects).
3. Report final test accuracy for both and compare.
4. **Extend to one hidden layer** (NumPy MLP: `X → W1 → ReLU → W2 → softmax`). You'll need to backpropagate through the ReLU — note that its gradient is simply `1` where the input was positive and `0` elsewhere. Verify these gradients against PyTorch too.

**Some things to think about:**

- What happens to your gradients if you initialize all weights to zero instead of randomly? Try it. Why does the network fail to learn, and what does that tell you about why random initialization is necessary?
- Your NumPy version has no `zero_grad()`. Why not — what is it about writing the update manually that makes the problem disappear?

> 📌 As with every previous lab, no reference solution is included. Build it from Section 2's hand-verified gradient, which is this task in miniature.

---

---

## ➡️ Next

Continue to **[Lab 6-2 — Image Features &amp; Training Mechanics](./labsheet6-2.md)**, which reuses the `dataset.py`, `model.py` and `train.py` you just wrote — unchanged — and points them at images.

---
