# 🖥️ Computer Vision Lab 6-2 — Image Features & Training Mechanics

## 🎯 Objectives

In this lab, you will:

1. Extend your `dataset.py` with a converter that turns **MNIST images into LBP feature vectors**.
2. Compare that hand-designed representation against **raw pixels**, with everything else held fixed.
3. Build an **overfitting sandbox** — a setup small enough that regularization visibly does something.
4. Compare **loss functions**: cross-entropy, multiclass hinge (Lab 4's SVM objective), and MSE as the deliberate wrong answer.
5. Compare **optimizers**, and see why optimizer and learning rate are *not* independent knobs.
6. Diagnose a bad **learning rate** from its gradient histograms.
7. Apply **regularization** (L2 via `weight_decay`, L1 added by hand) and **dropout**, and measure what each closes.
8. Understand why **`model.eval()` is not optional**, by watching the bug happen.
9. Load a saved `best.pt` and run **inference**, then run a systematic **feature × architecture study**.

> 📌 **This lab continues directly from [Lab 6-1](./labsheet6-1.md).** It reuses the `dataset.py`, `model.py` and `train.py` you wrote there — **unchanged**, apart from adding two new `Dataset` classes in Section 1. Every experiment below is a *config change*, not new training code. If you haven't finished 6-1, do that first.

> 💡 A companion notebook, [`labsheet6-2.ipynb`](./labsheet6-2.ipynb), contains everything below in runnable order. Use the same `cv-dl` environment and kernel you set up in Lab 6-1 Section 0.

---

## 0️⃣ Prerequisites

- The **`cv-dl`** conda environment and its Jupyter kernel, from Lab 6-1 Section 0. Nothing new to install.
- **`dataset.py`, `model.py` and `train.py` from Lab 6-1**, in the same folder as this notebook.

```bash
conda activate cv-dl
cd <your lab6 folder>
jupyter notebook labsheet6-2.ipynb
```

Then select the **Python (cv-dl)** kernel, exactly as in Lab 6-1.

> 💡 **Working locally means your Lab 6-1 files are simply still there** — nothing was wiped between sessions, and `checkpoints/` and `runs/` from 6-1 are still on disk, so TensorBoard will show your IRIS runs alongside the new ones in this lab.
>
> If you no longer have the three files (different machine, or you cleaned the folder), the notebook's setup cells can re-create them — they are byte-identical to Lab 6-1's. Run those cells only if the import check below fails.

### Check you're ready

```python
import sys
print("python:", sys.executable)        # <- must contain 'cv-dl'

from dataset import build_iris_datasets
from model import build_model
from train import train
print("Lab 6-1 modules imported OK")
```

If `sys.executable` doesn't contain `cv-dl`, you're on the wrong kernel — fix that first (Lab 6-1 Section 0, Step 3). If the imports fail with `ModuleNotFoundError: No module named 'dataset'`, the three files aren't in this folder; run the setup cells in the notebook to re-create them.

### Everything here is a config change

Lab 6-1 deliberately built `train()` as a **parameterized function** rather than a hard-coded loop:

```python
train(config, train_ds, val_ds, test_ds=None, scaler=None, class_names=None)
```

Every experiment in this lab changes one key in `config` and gives the run a new `run_name`. All runs land in the same `runs/` tree, so you can overlay them in a single TensorBoard and compare directly. **If you find yourself copy-pasting a training loop, stop and reread Lab 6-1 Section 5** — that's the signal you've misread how `train()` is meant to be used.

Start TensorBoard now, before training anything, and refresh it as runs appear:

```python
%load_ext tensorboard
%tensorboard --logdir runs
```

Locally, in a second terminal with `cv-dl` active:

```bash
tensorboard --logdir runs
# then open http://localhost:6006
```

---

## 1️⃣ Extending `dataset.py`: Converting MNIST into LBP Features

### There is no "LBP dataset" — we derive it

MNIST ships raw 28×28 grayscale images and nothing else. `MNISTLBPDataset` below is a **converter**, not a loader of something that already exists: it takes the raw images and computes a Local Binary Pattern feature vector for each one.

You met LBP in Lab 4 (suggested for the Caltech-256 feature-engineering task) and Lab 5 (suggested for superpixel features). Here it finally gets used as the *primary* representation, feeding a network.

> 💡 **Add the code below to the same `dataset.py` you wrote in Lab 6-1.** `IrisDataset` and `build_iris_datasets()` stay exactly as they are — you're appending, not replacing.

> 🔎 **This transform is *stateless*, and that matters.** In Lab 4, the BoVW vocabulary had to be fitted on training descriptors only, or information about val/test would leak into the features. LBP has **no learned parameters** — it's a fixed function of the pixels — so train, val and test all go through the identical function and there is nothing to leak. Getting into the habit of asking "does this transform need fitting, and if so on what?" is worth more than either answer.

### How LBP becomes a feature vector

For each pixel, LBP compares it to its `P` neighbours on a circle of radius `R`, producing a `P`-bit code. With `method="uniform"` and `P=8`, those codes collapse into **10 distinct values** (the 9 "uniform" patterns plus one bucket for everything else). So an LBP code map of a 28×28 image is a 28×28 array of values in `0..9`.

A histogram of that whole map would be a 10-dimensional vector — and that is **far too little to separate 10 digit classes**. Worse, a global histogram discards *where* each pattern occurred, which for handwritten digits is almost all of the information. A 6 and a 9 contain nearly identical local patterns; what distinguishes them is where those patterns sit.

The fix is the same one that makes LBP work for face recognition: compute the code map once over the whole image, then **split it into a grid of cells and histogram each cell separately**, concatenating the results.

```
28×28 uint8 image
   └─> local_binary_pattern(P=8, R=1, "uniform")  ->  28×28 code map, values 0..9
        └─> split into a 4×4 grid of 7×7 cells
             └─> bincount each cell into 10 bins, L1-normalize
                  └─> concatenate  ->  4 × 4 × 10 = 160-d feature vector
```

> ⚠️ Compute LBP **once over the whole image**, then grid the resulting code map. Do **not** crop the image into cells and run LBP separately on each — at `R=1` every cell would lose a 1-pixel border, throwing away information and introducing artificial edges at every cell boundary.

```python
"""dataset.py (continued) - MNIST, as LBP features and as raw pixels.

Append this to the dataset.py you wrote in Lab 6-1.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from skimage.feature import local_binary_pattern

LBP_P = 8        # number of circular neighbours

LBP_R = 1        # radius, in pixels
LBP_BINS = 10    # uniform LBP with P=8 produces P+2 = 10 distinct codes
LBP_GRID = 4     # 4x4 grid of cells -> 4*4*10 = 160-d feature


def load_mnist_arrays(root="./data", train=True):
    """Returns (images, targets) as a (N, 28, 28) uint8 array and an int64 tensor.

    Primary path is torchvision, whose MNIST exposes .data and .targets
    directly - so we skip PIL and ToTensor() and hand numpy straight to skimage.

    Fallback: torchvision's MNIST mirrors go down regularly, and the original
    yann.lecun.com URL now returns 404 permanently. Rather than leaving you
    stuck, we pull the identical data from OpenML if the download fails.
    """
    try:
        base = MNIST(root=root, train=train, download=True)
        return base.data.numpy(), base.targets.clone()
    except Exception as exc:
        print(f"[mnist] torchvision download failed ({type(exc).__name__}); "
              f"falling back to OpenML...")
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                            as_frame=False, parser="auto")
        X = X.astype(np.uint8).reshape(-1, 28, 28)
        y = y.astype(np.int64)
        # OpenML ships one combined set; the canonical split is first 60k / last 10k.
        span = slice(0, 60000) if train else slice(60000, 70000)
        return X[span], torch.tensor(y[span])


def lbp_feature(image_u8, P=LBP_P, R=LBP_R, grid=LBP_GRID, n_bins=LBP_BINS):
    """One 28x28 uint8 image -> a (grid*grid*n_bins,)-d float32 feature vector.

    Stateless: no fitting, no parameters learned from data. Train, val and
    test all go through this identical function.
    """
    codes = local_binary_pattern(image_u8, P, R, method="uniform")

    h, w = codes.shape
    row_edges = np.linspace(0, h, grid + 1).astype(int)
    col_edges = np.linspace(0, w, grid + 1).astype(int)

    blocks = []
    for i in range(grid):
        for j in range(grid):
            cell = codes[row_edges[i]:row_edges[i + 1],
                         col_edges[j]:col_edges[j + 1]].astype(np.int64).ravel()
            # bincount is much faster than np.histogram for small integer codes
            hist = np.bincount(cell, minlength=n_bins)[:n_bins].astype(np.float32)
            total = hist.sum()
            if total > 0:
                hist /= total   # L1-normalize: a distribution, not a raw count
            blocks.append(hist)

    return np.concatenate(blocks)


class MNISTLBPDataset(Dataset):
    """MNIST converted to grid-LBP feature vectors.

    The conversion is done ONCE in __init__ and cached to disk, not recomputed
    in __getitem__ - see the note below on why that matters.
    """

    def __init__(self, root="./data", train=True, cache_dir="./cache",
                 limit=None, random_state=42):
        split = "train" if train else "test"
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"mnist_lbp_{split}_P{LBP_P}_R{LBP_R}_g{LBP_GRID}.pt"
        )

        if os.path.exists(cache_path):
            blob = torch.load(cache_path)
            features, targets = blob["features"], blob["targets"]
            print(f"[MNISTLBPDataset] loaded cached {split} features from {cache_path}")
        else:
            images, targets = load_mnist_arrays(root=root, train=train)

            print(f"[MNISTLBPDataset] converting {len(images)} {split} images to LBP "
                  f"features (one-time, then cached)...")
            features = torch.tensor(
                np.stack([lbp_feature(im) for im in images]), dtype=torch.float32
            )
            torch.save({"features": features, "targets": targets}, cache_path)
            print(f"[MNISTLBPDataset] cached to {cache_path}")

        # Subsample AFTER caching, so changing `limit` never triggers a reconversion.
        if limit is not None and limit < len(targets):
            generator = torch.Generator().manual_seed(random_state)
            keep = torch.randperm(len(targets), generator=generator)[:limit]
            features, targets = features[keep], targets[keep]

        self.features = features
        self.targets = targets
        self.n_features = features.shape[1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class MNISTRawDataset(Dataset):
    """MNIST as flattened, scaled raw pixels (784-d).

    This is the CONTROL for the LBP experiment - without it, an LBP accuracy
    number has nothing to be compared against and means very little.
    """

    def __init__(self, root="./data", train=True, limit=None, random_state=42):
        images, targets = load_mnist_arrays(root=root, train=train)
        features = torch.tensor(
            images.reshape(len(images), -1).astype(np.float32) / 255.0
        )

        if limit is not None and limit < len(targets):
            generator = torch.Generator().manual_seed(random_state)
            keep = torch.randperm(len(targets), generator=generator)[:limit]
            features, targets = features[keep], targets[keep]

        self.features = features
        self.targets = targets
        self.n_features = features.shape[1]   # 784

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
```

> ⚠️ **MNIST downloads fail surprisingly often.** The original `yann.lecun.com/exdb/mnist/` URL now returns **404 permanently**, and torchvision's S3 mirror intermittently refuses connections. If you see `RuntimeError: Error downloading train-images-idx3-ubyte.gz`, that is the mirror, not your code. `load_mnist_arrays()` above handles this by falling back to OpenML automatically — the data is identical, it just arrives as one combined 70,000-image array that we re-split into the canonical first-60k / last-10k.

> ⏱️ **Why convert in `__init__` and not `__getitem__`?** Computing LBP inside `__getitem__` is the "obvious" design — it looks like a normal transform. But `__getitem__` runs **once per sample per epoch**, so a 30-epoch run would recompute the identical features 30 times, and the CPU-bound conversion becomes the bottleneck while the model sits idle waiting for batches. Converting once in `__init__` and caching to disk means you pay for it one time, ever.
>
> For calibration: the conversion itself runs at roughly **0.2 ms per image** on a typical CPU, so converting the full 60,000-image training split takes on the order of **10–20 seconds** — once. (The *first* run also downloads MNIST, which adds its own time on top; that's the download, not the conversion.) Loading the cached tensor on every subsequent run takes a small fraction of a second — if yours is still taking tens of seconds every time, you are recomputing instead of hitting the cache.
>
> **Measure the lazy-vs-precomputed difference yourself rather than taking our word for it** — see Task 2 below.

### Task

1. Visualize what LBP is actually seeing. Pick a few MNIST digits, display the original next to its LBP code map (`imshow` the `codes` array), and next to a bar plot of its 160-d feature vector. Do two images of the same digit have visibly similar feature vectors? Do two different digits differ?
2. **Measure the lazy-vs-precomputed difference.** Write a variant of `MNISTLBPDataset` that calls `lbp_feature()` inside `__getitem__` instead of precomputing, then time one full epoch with each version (`time.perf_counter()`, as in Lab 5). Report both numbers and the ratio.
3. Vary `LBP_GRID` (try `1`, `2`, `4`, `7`) and note the resulting feature dimension each time. Keep these variants — you'll train on them in this lab's Student Task.

---

## 2️⃣ Running It: LBP Features vs Raw Pixels

Same `train()`, same models, same everything — only the **representation** changes.

```python
"""run_lab6_mnist.py"""

import torch
from torch.utils.data import random_split

from dataset import MNISTLBPDataset, MNISTRawDataset
from train import train

# The first MNISTLBPDataset construction downloads MNIST and runs the one-time
# LBP conversion; every later run loads the cached tensor instantly.
lbp_train = MNISTLBPDataset(train=True)
lbp_test = MNISTLBPDataset(train=False)
# Carve a validation split out of the training data - MNIST ships only train/test.
lbp_val, lbp_train = random_split(
    lbp_train, [6000, len(lbp_train) - 6000],
    generator=torch.Generator().manual_seed(42),
)

base_mnist = dict(
    num_classes=10, batch_size=128, epochs=30,
    loss="cross_entropy", optimizer="adam", lr=1e-3, weight_decay=0.0,
    patience=5,
)

train({**base_mnist, "run_name": "mnist_lbp_single", "model": "single",
       "in_features": 160}, lbp_train, lbp_val, lbp_test)

train({**base_mnist, "run_name": "mnist_lbp_mlp", "model": "mlp",
       "in_features": 160, "hidden_sizes": (128,)}, lbp_train, lbp_val, lbp_test)

# The control: identical everything, raw 784-d pixels instead of 160-d LBP.
raw_train = MNISTRawDataset(train=True)
raw_test = MNISTRawDataset(train=False)
raw_val, raw_train = random_split(
    raw_train, [6000, len(raw_train) - 6000],
    generator=torch.Generator().manual_seed(42),
)

train({**base_mnist, "run_name": "mnist_raw_single", "model": "single",
       "in_features": 784}, raw_train, raw_val, raw_test)

train({**base_mnist, "run_name": "mnist_raw_mlp", "model": "mlp",
       "in_features": 784, "hidden_sizes": (128,)}, raw_train, raw_val, raw_test)
```

### Task

1. Run all four configurations. For each, report the best epoch, best validation loss, and test accuracy.
2. **Single vs multi layer**, on each representation. Where a single layer nearly matches the MLP, what does that tell you about how linearly separable those features already are?
3. **LBP vs raw pixels.** Which representation wins, by how much, and at what feature dimension? The LBP feature is 160-d against raw pixels' 784-d — so "which is better" and "which is better *per dimension*" are two different questions, and you should answer both.
4. Before writing up (3), think about what each representation actually encodes. LBP compresses the image into a description of *local texture patterns*, binned by region. Raw pixels keep everything, including exact position, but know nothing about structure. MNIST digits are near-binary strokes on a flat background — so how much genuine *texture* is there for LBP to find, and how much of its performance is really the `4×4` grid preserving *where* things are? This lab's student task varies the grid size specifically to separate those two effects. **Report what you measure, not what you expected — and don't assume the hand-designed feature has to lose.**

---

## 3️⃣ Training Mechanics: Loss, Optimizer, Learning Rate, Regularization, Dropout

Every experiment here is a config change to the same `train()` function, and every run lands in the same `runs/` tree, so you can overlay them all in one TensorBoard.

### 3.0 The overfitting sandbox

> ⚠️ **Read this before running the regularization and dropout experiments.** On full MNIST (54k training images) a small MLP barely overfits — train and val curves stay glued together, so weight decay and dropout change nothing visible and the comparison is worthless. To study overfitting you need a setup that actually overfits: **subsample the training set to ~1000 images**.

```python
sandbox_train = MNISTLBPDataset(train=True, limit=1000)
sandbox_val = MNISTLBPDataset(train=False, limit=2000)

sandbox = dict(
    in_features=160, num_classes=10, model="mlp", hidden_sizes=(256, 256),
    batch_size=32, epochs=60, loss="cross_entropy", optimizer="adam",
    lr=1e-3, patience=60,   # patience disabled, so you can SEE the overfit happen
)
```

A wide MLP on 1000 samples will drive training accuracy to ~1.0 while validation stalls and val loss turns back upward. That visible gap is what the next three experiments try to close.

### 3.1 Loss function

| Loss | PyTorch | Notes |
|---|---|---|
| Cross-entropy | `nn.CrossEntropyLoss()` | The default for classification. Expects **raw logits** and **class indices**. |
| Multiclass hinge | `nn.MultiMarginLoss()` | This *is* Lab 4's linear SVM objective. With `model="single"`, this run reproduces Lab 4's `SGDClassifier(loss="hinge")` in PyTorch. |
| MSE on one-hot | `OneHotMSELoss` | The wrong tool, included to see *how* it's wrong. |

Run all three on the same model and dataset, changing only `loss`.

> ⚠️ **Compare these runs on *accuracy*, never on loss.** Cross-entropy, hinge and MSE produce numbers on completely different scales — an MSE run will report a far smaller "loss" than a cross-entropy run while being a much worse classifier. The loss value is only meaningful *within* one loss function, for watching a single run converge. This is why Lab 6-1's `train()` returns `best_val_acc` alongside `best_val_loss`, and it is a mistake that is very easy to make when a results table has a tidy "loss" column inviting you to sort it.

Expect cross-entropy and hinge to land reasonably close, with MSE learning visibly more slowly. Pay attention to what the *loss* column does while that happens — it is entirely possible for the MSE run to report the **lowest loss of the three while being the worst classifier of the three**, which is the trap above in its most concrete form.

Think about why MSE struggles here: MSE's gradient through a softmax becomes very small when a prediction is confidently *wrong*, so the model barely corrects its worst mistakes — exactly the opposite of what you want. Cross-entropy's gradient, by contrast, grows with how wrong the prediction is.

### 3.2 Optimizer

Run `sgd`, `sgd_momentum`, `adam`, each at its own sensible learning rate. Compare **epochs to reach a target validation accuracy** and **curve smoothness**, not just the final number — on problems this small everything converges eventually, and comparing only endpoints hides the entire difference.

Then run **Adam at SGD's learning rate** (`adam`, `lr=0.1`) and **SGD at Adam's** (`sgd`, `lr=1e-3`). The point: optimizer and learning rate are **not independent knobs**. Adam adapts a per-parameter step size, so it wants a much smaller base rate than plain SGD; swapping one without the other breaks both.

### 3.3 Learning rate

Three runs on the same everything else:

| `lr` | Expected behaviour |
|---|---|
| `1e-5` | Loss barely moves — the model is learning, just unusably slowly |
| `1e-3` | Healthy convergence (with Adam) |
| `1.0` | Training collapses — accuracy drops to roughly chance (~0.1 for 10 classes) and stays there. With other optimizers or a larger rate still you may instead see the loss go to `NaN` outright. |

> 🔎 A collapsed run and a NaN run are both failures, but they *look* different, and the collapsed one is the more dangerous of the two: there is no error, no crash, and a loss number that still prints. Only the accuracy — and the gradient histograms — reveal that nothing is being learned.

Open **HISTOGRAMS** and look at `grads/*` for each run. The `lr=1.0` run shows gradients exploding across orders of magnitude; the `lr=1e-5` run shows them steady but producing no movement in the weights. **Include screenshots of the gradient histograms in your report** — being able to diagnose a bad learning rate from a gradient histogram is the skill this section is actually teaching.

### 3.4 Regularization (L2 / weight decay, and L1)

In PyTorch, **L2 regularization is an optimizer argument**, not a term you add to the loss:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

This surprises people who learned it as `loss + λ‖w‖²`. Mathematically it's the same thing — the derivative of `λ‖w‖²` is just `2λw`, which the optimizer adds directly to the gradient rather than routing it through the loss.

**L1 is not built in**, which is why Lab 6-1's `train()` adds it manually. That three-line block is worth staring at: it demonstrates that a loss is just an expression autograd can differentiate, so you can extend it with any differentiable term you like.

> 💡 `Adam(weight_decay=)` folds L2 into the gradient, while `AdamW` applies *decoupled* weight decay directly to the weights. The two behave differently in practice and `AdamW` is generally preferred with Adam-family optimizers. Both are available in `build_optimizer` if you want to compare.

On the sandbox config, sweep `weight_decay ∈ {0, 1e-4, 1e-2, 1e-1}`. Watch `generalization_gap` in TensorBoard: it should narrow as you increase it, then at the high end the model underfits and *both* curves get worse.

### 3.5 Dropout — and why `model.eval()` is not optional

Dropout randomly zeroes a fraction `p` of activations during training, so the network can't lean on any single unit. Note that it needs a hidden layer to act on, so **dropout is inherently an MLP technique** — there is nothing to drop in a single-layer perceptron.

Sweep `dropout ∈ {0.0, 0.2, 0.5}` on the sandbox config.

**Then do this experiment**, which is the most valuable one in the section:

```python
"""eval_mode_demo.py - why model.eval() is not optional."""
import torch
from torch.utils.data import DataLoader
from dataset import MNISTLBPDataset
from model import build_model

# NOTE: we deliberately do NOT reuse train.py's evaluate() here, because it
# calls model.eval() internally - which would silently fix the very bug we
# are trying to demonstrate. This helper leaves the mode exactly as set.
@torch.no_grad()
def accuracy_leaving_mode_alone(model, loader):
    correct = total = 0
    for features, labels in loader:
        correct += (model(features).argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return correct / total


train_ds = MNISTLBPDataset(train=True, limit=1000)
val_ds = MNISTLBPDataset(train=False, limit=2000)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)

model = build_model("mlp", 160, 10, hidden_sizes=(256, 256), dropout=0.5)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for _ in range(10):
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()

model.train()   # BUG: dropout is STILL ACTIVE during evaluation
print("train() mode:", [round(accuracy_leaving_mode_alone(model, val_loader), 4)
                        for _ in range(5)])

model.eval()    # correct: dropout disabled
print("eval()  mode:", [round(accuracy_leaving_mode_alone(model, val_loader), 4)
                        for _ in range(5)])
```

In `train()` mode the accuracy is **lower, and different every single time** — you are randomly deleting half the network on every forward pass, so the same model on the same data gives a different answer each run. In `eval()` mode it is higher and identical every time.

> ⚠️ **Note the trap this demo had to avoid, because you will hit it too.** Lab 6-1's `evaluate()` starts with `model.eval()`. If you call *that* function to run this experiment, both halves print the same numbers and the bug appears not to exist — the helper quietly repaired it. Any time you try to reproduce a bug and it refuses to appear, check whether something downstream is fixing it for you.

Forgetting `model.eval()` is a bug you will hit for years. Seeing it misbehave once is worth more than reading the rule ten times.

### Task — the experiment matrix

Run the matrix below (all on the sandbox config unless noted), and present the results as **one table plus the overlaid TensorBoard screenshots**:

| # | What varies | Values |
|---|---|---|
| 1 | `loss` | `cross_entropy`, `hinge`, `mse` |
| 2 | `optimizer` | `sgd`, `sgd_momentum`, `adam` (each at a sensible `lr`) |
| 3 | optimizer/LR mismatch | `adam@0.1`, `sgd@1e-3` |
| 4 | `lr` | `1e-5`, `1e-3`, `1.0` |
| 5 | `weight_decay` | `0`, `1e-4`, `1e-2`, `1e-1` |
| 6 | `dropout` | `0.0`, `0.2`, `0.5` |
| 7 | `model.eval()` bug | the five-repeat comparison above |

For each group, state **which setting you would ship and why**, in one sentence. Then re-enable early stopping (`patience=5`) on your best configuration and report which epoch `best.pt` came from versus which epoch training stopped at — the difference between those two numbers is the concrete thing early stopping and best-checkpointing bought you.

---

## 4️⃣ Inference: Loading `best.pt`

Every labsheet in this course ends with a "load the saved model, run it on one input, display the result" script — Lab 4's version used `joblib.load`. Here's Lab 6's, loading a checkpoint you trained in Section 2.

```python
"""predict.py - classify one hardcoded input using a saved best.pt"""

import cv2
import torch
import numpy as np

from model import build_model
from dataset import lbp_feature

CHECKPOINT_PATH = "checkpoints/mnist_lbp_mlp/best.pt"   # hardcoded - change as needed
IMAGE_PATH = "my_digit.png"                             # a 28x28-ish grayscale digit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
config = ckpt["config"]
print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
      f"(val_loss={ckpt['val_loss']:.4f}, val_acc={ckpt['val_acc']:.4f})")

# Rebuild the SAME architecture the checkpoint was trained with, then load the
# weights into it. This is why we saved `config` alongside the state_dict.
model = build_model(
    config["model"], config["in_features"], config["num_classes"],
    hidden_sizes=tuple(config.get("hidden_sizes", (128,))),
    dropout=config.get("dropout", 0.0),
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()          # disables dropout - NOT optional (Section 3.5)

# ---- preprocess exactly as the training data was preprocessed ----
gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError(f"Could not read {IMAGE_PATH}")
gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

features = torch.tensor(lbp_feature(gray), dtype=torch.float32).unsqueeze(0).to(device)

# If this checkpoint was trained on standardized features (e.g. IRIS), apply
# the SAME scaler that was fitted at training time and saved with the weights.
if ckpt.get("scaler_mean") is not None:
    mean = torch.tensor(ckpt["scaler_mean"], dtype=torch.float32, device=device)
    scale = torch.tensor(ckpt["scaler_scale"], dtype=torch.float32, device=device)
    features = (features - mean) / scale

with torch.no_grad():
    logits = model(features)
    # Softmax belongs HERE, at display time - not in forward() (Lab 6-1 Section 4).
    probabilities = torch.softmax(logits, dim=1)[0]

predicted = int(probabilities.argmax())
confidence = float(probabilities[predicted])
class_names = ckpt.get("class_names") or [str(i) for i in range(config["num_classes"])]

print(f"Predicted: {class_names[predicted]}  (confidence {confidence:.3f})")
print("\nTop 3:")
top3 = torch.topk(probabilities, k=min(3, len(class_names)))
for score, idx in zip(top3.values.tolist(), top3.indices.tolist()):
    print(f"  {class_names[idx]:12s} {score:.3f}")

display = cv2.cvtColor(cv2.resize(gray, (280, 280), interpolation=cv2.INTER_NEAREST),
                        cv2.COLOR_GRAY2BGR)
cv2.putText(display, f"{class_names[predicted]} ({confidence:.2f})", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
cv2.imshow("Prediction", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

> 💡 **Inside a Jupyter notebook, use `matplotlib` instead of `cv2.imshow()`.** `cv2.imshow()` opens a separate OS window that frequently hangs the notebook kernel; matplotlib renders inline. The companion notebook does it that way. The script above uses `cv2.imshow()`, which is the right choice when you run it as a plain script from the terminal — `python predict.py`.

---

## 🧪 Student Task (Evaluative) — Feature & Architecture Study

A systematic study of how **feature representation** and **model capacity** interact, using the infrastructure you built above.

### The grid

Train every combination of:

- **Representation:** raw pixels (784-d), LBP `grid=1` (10-d), LBP `grid=2` (40-d), LBP `grid=4` (160-d), LBP `grid=7` (490-d)
- **Model:** single-layer perceptron, MLP with one hidden layer, MLP with two hidden layers

That's 15 runs. Each is a config change — if you find yourself copy-pasting a training loop, go back and reread Lab 6-1 Section 5.

**Requirements:**

1. **Results table:** representation, feature dimension, model, best epoch, test accuracy, wall-clock training time, and **one-time feature extraction time**. Report extraction time separately from training time — a representation that trains fast but takes ten minutes to compute is not actually fast.
2. **Tune the learning rate per representation**, at least coarsely. A 10-d input and a 784-d input do not want the same learning rate, and comparing an untuned run against a tuned one is not a fair comparison.
3. **Overlay all 15 runs in one TensorBoard** and include screenshots of the SCALARS and HPARAMS tabs.
4. Apply the mechanics from Section 3 (early stopping, and whichever regularization helped) consistently across all runs, and say what you applied.
5. **Analysis** — answer these specifically:
   - How does accuracy scale with LBP grid size? Is there a point of diminishing returns, or does it keep improving? What does that say about how much *spatial* information the grid is adding versus how much *texture* information LBP captured in the first place?
   - Where does the extra hidden layer help most, and where does it barely matter? Form a hypothesis about what that implies about the linear separability of each representation.
   - **Does any LBP variant beat raw pixels — and if so, at what feature dimension?** Compare on accuracy *and* on accuracy-per-dimension. Whichever way it comes out, explain it in terms of what LBP actually measures (local texture micro-patterns, binned per region) versus what distinguishes MNIST digits (stroke shape and position). A clearly-argued result earns full marks in either direction; an unexplained one does not.
   - **Separate the two effects.** Going from `grid=1` to `grid=7` changes the feature in two ways at once: it adds spatial information, *and* it increases dimensionality. Design a comparison that tells you which one is responsible for the improvement. (One idea among several: compare an LBP variant against a raw-pixel baseline cut down to a similar dimension, e.g. by downsampling the image.)
6. Submit the `best.pt` for your highest-accuracy configuration, along with a `predict.py` run (Section 4) demonstrating it on an image you provide yourself.

**Some things to think about:**

- LBP `grid=1` gives you a **10-dimensional** feature for a **10-class** problem. Before you run it, predict what will happen and why. Then run it.
- Look at where *all* your configurations plateau, not just which one wins. Both a good hand-designed feature and raw pixels, fed to a flat fully-connected network, tend to top out in the same region — and neither gets close to what a convolutional model reaches on MNIST. The gap between your best number here and ~0.99 is not a tuning failure; it is the limitation of treating an image as an unstructured vector of numbers, and it is exactly what the next stage of this course addresses.

> 📌 No reference solution is included in this lab sheet.

---
