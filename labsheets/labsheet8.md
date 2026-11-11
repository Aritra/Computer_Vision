# 🖥️ Computer Vision Lab 8 — Convolutional Neural Networks on MNIST

## 🎯 Objectives

In this lab, you will:

1. Understand **convolution** as a layer: kernels, stride, padding, dilation, and the output-size formula.
2. Implement a **2-D convolution from scratch** and verify it against `nn.Conv2d`.
3. See why **parameter sharing** and **translation equivariance** are the real advantages over Lab 6's MLP — not just "more layers".
4. Compute and reason about the **receptive field** of a stack of convolutions.
5. Build a configurable **CNN** and train it on MNIST with the same parameterized `train()` discipline as Labs 6 and 7.
6. Compare **pooling vs strided convolution**, **kernel size vs depth**, and the effect of **BatchNorm**.
7. Use **data augmentation**, and measure what it actually buys on a **shifted** test set.
8. **Visualize** learned filters and feature maps, and see what the network is responding to.

> 📌 **This lab answers the question Lab 6-2 ended on.** There you fed MNIST to a flat fully-connected network — as raw pixels and as LBP features — and found that both plateau well short of what MNIST allows. The closing note said the gap "is the limitation of treating an image as an unstructured vector of numbers". This lab removes that limitation.

> 📌 **New files, same habits.** Lab 6's models take `(batch, features)`; convolutions take `(batch, channels, height, width)`. So this lab introduces `img_dataset.py`, `cnn_model.py` and `img_train.py`, built in the same style as Lab 7's `seq_*` trio. Nothing in Labs 6 or 7 is modified. **Lab 9 reuses `img_dataset.py` and `img_train.py` unchanged.**

## 📖 Background: Why Convolution

A fully-connected layer on a 28×28 image first **flattens** it to a 784-vector. Two things are destroyed by that flatten:

1. **Locality.** Pixel `(5, 5)` and pixel `(5, 6)` are neighbours, but after flattening they are just dimensions 145 and 146 — no more related than dimensions 145 and 700. The network has to *learn* that adjacency from data.
2. **Translation structure.** A `7` in the top-left and the same `7` shifted three pixels right are completely different 784-vectors. Every weight that learned to recognise the first one is useless for the second.

A **convolutional layer** fixes both by construction. It slides a small **kernel** (say 3×3) across the image, computing a weighted sum of each local neighbourhood:

```
    output[i, j] = Σ_m Σ_n  kernel[m, n] · input[i + m, j + n]   + bias
```

Three consequences, and they are the whole point:

| Property | What it means | Why it matters |
|---|---|---|
| **Local connectivity** | each output sees only a small patch | wires locality in, instead of learning it |
| **Parameter sharing** | the *same* kernel is used at every position | an edge detector learned in one corner works everywhere |
| **Translation equivariance** | shift the input, the feature map shifts the same way | a shifted `7` produces shifted features, not unrelated ones |

Parameter sharing is also why convolutions are so cheap. One 3×3 kernel producing 32 channels from a 1-channel image has `32 × (1×3×3 + 1) = 320` parameters, **regardless of image size**. A dense layer mapping 784 pixels to 32 units needs `784 × 32 + 32 = 25,120`. You will verify both numbers in Section 1.

> 🔗 **What this does *not* give you.** Convolution is equivariant to translation, not **invariant** to it, and it is not equivariant to rotation or scale at all. Pooling adds a little local invariance; the rest has to come from data augmentation or from the classifier head. Section 5.4 measures exactly this.

---

## 0️⃣ Prerequisites

- The **`cv-dl`** conda environment and its Jupyter kernel from Lab 6-1. Nothing new to install — `torch` and `torchvision` are already there.

```bash
conda activate cv-dl
cd <your lab8 folder>
jupyter notebook labsheet8.ipynb
```

Then select the **Python (cv-dl)** kernel, as in Lab 6-1.

### Check you're ready

```python
import sys, torch, torchvision
print("python     :", sys.executable)        # must contain 'cv-dl'
print("torch      :", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA       :", torch.cuda.is_available())   # False is fine
```

Everything here trains on a CPU. A full CNN run on MNIST takes a couple of minutes; the small-subset experiments take seconds.

Start TensorBoard before training anything:

```python
%load_ext tensorboard
%tensorboard --logdir runs
```

### Files you will build

```
lab8/
├── img_dataset.py   # MNIST as (1, 28, 28) images, + augmentation + a shifted test set
├── cnn_model.py     # SimpleCNN, ManualConv2d, receptive_field
├── img_train.py     # the training loop, TensorBoard, checkpointing, early stopping
├── runs/            # TensorBoard logs
└── checkpoints/     # one folder per run, each with best.pt
```

---

## 1️⃣ Convolution Arithmetic, Parameters and Receptive Fields

### 1.1 Output size

For one spatial dimension, with input size `n`, kernel `k`, padding `p`, stride `s` and dilation `d`:

```
    out = floor( (n + 2p − d·(k − 1) − 1) / s ) + 1
```

Two cases are worth memorising, because almost every architecture uses them:

| Setting | Effect | Example (n = 28) |
|---|---|---|
| `k=3, p=1, s=1` | **size preserved** | 28 → 28 |
| `k=3, p=1, s=2` | **halved** | 28 → 14 |

> ⚠️ **"Same" padding only works for odd kernels.** With `k=3, p=1` the output matches the input exactly. With an even kernel there is no integer `p` that preserves size symmetrically, which is why 3×3 and 5×5 are everywhere and 4×4 is rare.

### 1.2 Parameter sharing, counted

A `Conv2d(in_ch, out_ch, k)` layer has

```
    out_ch × (in_ch × k × k)   weights   +   out_ch   biases
```

**and that number does not depend on the image size at all.** A dense layer's parameter count is proportional to the number of pixels. You will verify this in the Task below; the contrast is the single strongest argument for convolution.

### 1.3 Receptive field

The **receptive field** of an output unit is the region of the *input* that can influence it. One 3×3 conv sees 3×3. Stack a second 3×3 on top and each unit now sees 5×5 of the original. For a stack of layers with kernels `k_i` and strides `s_i`:

```
    RF = 1 + Σ_i ( (k_i − 1) × Π_{j<i} s_j )
```

This is why **two stacked 3×3 convolutions are usually preferred to one 5×5**: the receptive field is the same (5×5), but the parameter count is lower (`2 × 9 = 18` vs `25` per channel pair) *and* there is an extra non-linearity in between. Depth buys you receptive field more cheaply than width buys it.

> 💡 **A CNN cannot classify what it cannot see.** If the receptive field of your final feature map is smaller than the digit, the network is deciding from a fragment. On 28×28 MNIST this is easy to satisfy; on larger images it is a real design constraint.

### 1.4 `img_dataset.py`

```python
"""img_dataset.py - MNIST as images, with augmentation and a shifted test set.

Lab 6 flattened MNIST to 784-vectors. Convolutions need the spatial layout,
so everything here keeps the (1, 28, 28) shape. Lab 9 reuses this file.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081     # the standard MNIST statistics


def load_mnist_arrays(root="./data", train=True):
    """Returns (images uint8 (N, 28, 28), targets int64).

    torchvision exposes .data and .targets directly, so we skip PIL entirely.
    Falls back to OpenML if the mirrors are down - the same guard as Lab 6-2,
    because yann.lecun.com now 404s permanently.
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
        span = slice(0, 60000) if train else slice(60000, 70000)
        return X[span], torch.tensor(y[span])


class MNISTImageDataset(Dataset):
    """MNIST as normalized (1, 28, 28) float tensors.

    shift_max : if > 0, every image is randomly translated by up to this many
                pixels when it is fetched. Used BOTH as training augmentation
                and (with a fixed seed) to build a shifted TEST set, which is
                how Section 5.4 measures translation robustness honestly.
    """

    def __init__(self, root="./data", train=True, limit=None, shift_max=0,
                 fixed_shift=False, random_state=42):
        images, targets = load_mnist_arrays(root=root, train=train)
        x = torch.tensor(images, dtype=torch.float32) / 255.0
        x = (x - MNIST_MEAN) / MNIST_STD
        self.images = x[:, None]                     # (N, 1, 28, 28)
        self.targets = targets
        self.shift_max = shift_max
        self.fixed_shift = fixed_shift

        if limit is not None and limit < len(self.targets):
            g = torch.Generator().manual_seed(random_state)
            keep = torch.randperm(len(self.targets), generator=g)[:limit]
            self.images, self.targets = self.images[keep], self.targets[keep]

        if shift_max > 0 and fixed_shift:
            # Apply ONCE, deterministically: a fixed, reproducible test set.
            g = torch.Generator().manual_seed(random_state)
            shifts = torch.randint(-shift_max, shift_max + 1,
                                   (len(self.targets), 2), generator=g)
            self.images = torch.stack([
                _translate(img, int(dx), int(dy))
                for img, (dx, dy) in zip(self.images, shifts)
            ])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.shift_max > 0 and not self.fixed_shift:
            # Fresh random shift each epoch -> this is AUGMENTATION.
            dx, dy = torch.randint(-self.shift_max, self.shift_max + 1, (2,))
            img = _translate(img, int(dx), int(dy))
        return img, self.targets[idx]


def _translate(img, dx, dy):
    """Shift a (1, H, W) tensor by (dx, dy) pixels, filling with the
    normalized value of a black pixel (NOT zero - see the warning below)."""
    pad = (0 - MNIST_MEAN) / MNIST_STD
    out = torch.full_like(img, pad)
    h, w = img.shape[-2:]
    xs_src = slice(max(0, -dx), min(w, w - dx))
    xs_dst = slice(max(0, dx), min(w, w + dx))
    ys_src = slice(max(0, -dy), min(h, h - dy))
    ys_dst = slice(max(0, dy), min(h, h + dy))
    out[:, ys_dst, xs_dst] = img[:, ys_src, xs_src]
    return out
```

> ⚠️ **Fill with the right value, not with zero.** After normalisation a black background pixel is `(0 − 0.1307)/0.3081 ≈ −0.424`, **not** `0`. Padding a shifted image with literal zeros paints a grey frame the network can detect, and it will cheerfully learn "grey border ⇒ shifted" instead of becoming robust. This kind of silent cue is the most common way an augmentation experiment lies to you.

### Task

1. **Verify the output-size formula.** For `n = 28`, compute by hand and then check with a real tensor:

   ```python
   import torch
   from torch import nn
   for k, p, s in [(3, 1, 1), (3, 0, 1), (3, 1, 2), (5, 2, 1), (5, 0, 1), (7, 3, 1)]:
       out = nn.Conv2d(1, 1, k, stride=s, padding=p)(torch.zeros(1, 1, 28, 28))
       print(k, p, s, "->", tuple(out.shape[-2:]))
   ```

2. **Count the parameter-sharing win.** Compare `nn.Conv2d(1, 32, 3, padding=1)` against `nn.Linear(28*28, 32)`. Report both counts and the ratio. Then repeat the comparison imagining a 224×224 input: which count changes, and which does not?
3. **Receptive field.** Using the formula in 1.3, compute the receptive field of three stacked `k=3, s=1` convolutions. Then of `conv3 → maxpool2 → conv3`. Verify your answer with `receptive_field()` from Section 2.
4. Explain in two sentences why two 3×3 convolutions are usually preferred to one 5×5.

---

## 2️⃣ `cnn_model.py`: A Configurable CNN

```python
"""cnn_model.py - a configurable CNN, a from-scratch conv, and RF arithmetic."""

import torch
import torch.nn.functional as F
from torch import nn


def receptive_field(layers):
    """Receptive field of one output unit, for a stack of (kernel, stride) pairs.

    RF = 1 + sum_i (k_i - 1) * prod_{j<i} s_j
    Pooling counts as a layer too: maxpool 2x2 stride 2 is (2, 2).
    """
    rf, jump = 1, 1
    for k, s in layers:
        rf += (k - 1) * jump
        jump *= s
    return rf


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SimpleCNN(nn.Module):
    """(B, 1, 28, 28) -> (B, num_classes).

    channels   : one conv block per entry, e.g. (32, 64)
    norm       : "none" | "batch"
    downsample : "max" | "avg" | "stride" | "none"
    head       : "flatten" (classic) | "gap" (global average pool)
    Returns raw logits. Softmax at display time only - as in Labs 6 and 7.
    """

    def __init__(self, in_channels=1, num_classes=10, channels=(32, 64),
                 kernel_size=3, norm="none", downsample="max", dropout=0.0,
                 head="flatten", head_hidden=128, input_size=28):
        super().__init__()
        self.layer_spec = []                       # for receptive_field()
        blocks, prev = [], in_channels
        for ch in channels:
            stride = 2 if downsample == "stride" else 1
            pad = kernel_size // 2                 # "same" padding for odd k
            blocks.append(nn.Conv2d(prev, ch, kernel_size, stride=stride, padding=pad))
            self.layer_spec.append((kernel_size, stride))
            if norm == "batch":
                blocks.append(nn.BatchNorm2d(ch))
            blocks.append(nn.ReLU())
            if downsample == "max":
                blocks.append(nn.MaxPool2d(2)); self.layer_spec.append((2, 2))
            elif downsample == "avg":
                blocks.append(nn.AvgPool2d(2)); self.layer_spec.append((2, 2))
            prev = ch
        self.features = nn.Sequential(*blocks)

        # Work out the flattened width by running one dummy tensor through,
        # rather than deriving it by hand and getting it wrong after an edit.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            feat = self.features(dummy)
        self.feature_shape = tuple(feat.shape[1:])

        if head == "gap":
            # Global average pool: one number per channel, so the head size is
            # independent of the input resolution.
            self.pool = nn.AdaptiveAvgPool2d(1)
            n_flat = prev
        elif head == "flatten":
            self.pool = nn.Identity()
            n_flat = int(torch.tensor(self.feature_shape).prod())
        else:
            raise ValueError(f"unknown head {head!r}")

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_flat, head_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

    def feature_maps(self, x):
        """Activations after every block - for the visualisation in 5.5."""
        maps, h = [], x
        for layer in self.features:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                maps.append(h.detach())
        return maps


class FlatMLP(nn.Module):
    """Lab 6's MLP, taking the SAME (B, 1, 28, 28) input so the comparison
    in Section 4 is controlled: identical data, identical training loop."""

    def __init__(self, num_classes=10, hidden=(128,), dropout=0.0, input_size=28):
        super().__init__()
        layers, prev = [nn.Flatten()], input_size * input_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- from scratch
class ManualConv2d(nn.Module):
    """A 2-D convolution written out with unfold, to show what Conv2d does.

    Parameter layout mirrors nn.Conv2d, so you can copy weights across and
    check the outputs agree - the same exercise as Lab 7's ManualLSTM.

    The trick: im2col. F.unfold rips out every k*k patch and lays it in a
    column, turning convolution into one big matrix multiply.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.k, self.stride, self.padding = kernel_size, stride, padding
        self.out_channels = out_channels
        scale = 1.0 / (in_channels * kernel_size * kernel_size) ** 0.5
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size).uniform_(-scale, scale))
        self.bias = nn.Parameter(torch.empty(out_channels).uniform_(-scale, scale))

    def forward(self, x):
        B, _, H, W = x.shape
        out_h = (H + 2 * self.padding - self.k) // self.stride + 1
        out_w = (W + 2 * self.padding - self.k) // self.stride + 1
        cols = F.unfold(x, self.k, padding=self.padding, stride=self.stride)
        out = self.weight.view(self.out_channels, -1) @ cols      # (B, out_ch, L)
        out = out + self.bias[None, :, None]
        return out.view(B, self.out_channels, out_h, out_w)
```

### Task

1. **Verify `ManualConv2d` against `nn.Conv2d`.** Build both with the same shape, copy `weight` and `bias` across, run the same input through, and assert with `torch.allclose(..., atol=1e-5)`.
2. Print `model.feature_shape` for `channels=(32, 64)` with `downsample="max"` and with `downsample="stride"`. Are they the same? Should they be?
3. Use `receptive_field(model.layer_spec)` on a 2-block and a 3-block CNN. At what depth does the receptive field first cover the whole 28×28 image?
4. Compare `head="flatten"` against `head="gap"` on parameter count. Where did the parameters go?

---

## 3️⃣ `img_train.py`: The Training Loop

Same shape as Lab 7's `seq_train.py` — one config dict, TensorBoard, early stopping, best-checkpointing. The only additions are an optional **LR scheduler** and logging of **images** so you can see what the model is actually being fed.

| Config key | Meaning | Default |
|---|---|---|
| `run_name` | log/checkpoint folder | required |
| `model` | `"cnn"` or `"mlp"` | `"cnn"` |
| `channels`, `kernel_size`, `norm`, `downsample`, `head`, `dropout` | CNN architecture | `(32,64)`, 3, `"none"`, `"max"`, `"flatten"`, 0.0 |
| `optimizer`, `lr`, `weight_decay` | optimisation | `"adam"`, `1e-3`, 0.0 |
| `scheduler` | `None` or `"cosine"` | `None` |
| `batch_size`, `epochs`, `patience`, `seed` | loop control | 128, 10, 3, 0 |

```python
"""img_train.py - one parameterized training run for image models."""

import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cnn_model import SimpleCNN, FlatMLP, count_params


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(config):
    """Everything in Labs 8 and 9 goes through here, so a run is fully
    described by its config dict - which is what makes the experiment
    tables in Section 5 honest."""
    if config["model"] == "mlp":
        return FlatMLP(num_classes=config.get("num_classes", 10),
                       hidden=config.get("hidden", (128,)),
                       dropout=config.get("dropout", 0.0))
    if config["model"] == "cnn":
        return SimpleCNN(
            num_classes=config.get("num_classes", 10),
            channels=tuple(config.get("channels", (32, 64))),
            kernel_size=config.get("kernel_size", 3),
            norm=config.get("norm", "none"),
            downsample=config.get("downsample", "max"),
            dropout=config.get("dropout", 0.0),
            head=config.get("head", "flatten"),
            head_hidden=config.get("head_hidden", 128),
        )
    raise ValueError(f"unknown model {config['model']!r}")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (loss, accuracy). Calls model.eval() itself - as in Lab 6-1.

    That matters more here than it did in Lab 6: BatchNorm, like Dropout,
    behaves differently in train and eval mode (Section 5.3).
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += criterion(out, y).item() * len(y)
        correct += (out.argmax(dim=1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def build_optimizer(name, params, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


def train(config, train_ds, val_ds, test_ds=None, model=None):
    """Pass `model` to train something build_model doesn't know about -
    which is exactly how Lab 9 reuses this file for a Vision Transformer."""
    torch.manual_seed(config.get("seed", 0))
    device = get_device()

    model = (model if model is not None else build_model(config)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config.get("optimizer", "adam"), model.parameters(),
                                config.get("lr", 1e-3), config.get("weight_decay", 0.0))

    epochs = config.get("epochs", 10)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if config.get("scheduler") == "cosine" else None)

    bs = config.get("batch_size", 128)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    run_name = config["run_name"]
    writer = SummaryWriter(os.path.join("runs", run_name))
    ckpt_dir = os.path.join("checkpoints", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best.pt")

    # Log one batch of inputs so you can SEE what the model is being fed -
    # augmentation bugs are far easier to spot in a picture than in a number.
    sample, _ = next(iter(train_loader))
    writer.add_images("inputs/train", sample[:32] * 0.3081 + 0.1307, 0)

    best_val, best_epoch, bad = float("inf"), 0, 0
    history, t0 = [], time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(y)
            correct += (out.argmax(dim=1) == y).sum().item()
            n += len(y)
        if scheduler is not None:
            scheduler.step()

        train_loss, train_acc = run_loss / n, correct / n
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("generalization_gap", train_acc - val_acc, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc))

        if val_loss < best_val:
            best_val, best_epoch, bad = val_loss, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                        "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc},
                       ckpt_path)
        else:
            bad += 1
            if bad >= config.get("patience", 3):
                break

    elapsed = time.perf_counter() - t0
    stopped = epoch

    # Test with the BEST weights, not whatever the last epoch happened to leave.
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    result = dict(run_name=run_name, best_epoch=best_epoch, stopped_epoch=stopped,
                  best_val_loss=best_val, val_acc=ckpt["val_acc"],
                  train_time=elapsed, n_params=count_params(model), history=history)
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=512)
        result["test_loss"], result["test_acc"] = evaluate(
            model, test_loader, criterion, device)

    writer.add_hparams({k: str(v) for k, v in config.items()},
                       {"hparam/val_acc": ckpt["val_acc"],
                        "hparam/test_acc": result.get("test_acc", float("nan"))})
    writer.close()
    return result
```

> 💡 **`model=` is the extension point.** `build_model` only knows about `"cnn"` and `"mlp"`, but `train()` accepts a ready-made module. Lab 9 passes a Vision Transformer straight in and reuses this entire file unchanged — exactly the way Lab 6-2 reused Lab 6-1's `train.py`.

---

## 4️⃣ Worked Example: MLP vs CNN, Same Data, Same Loop

The controlled comparison Lab 6-2 set up and could not finish.

```python
"""run_cnn_example.py - the comparison Lab 6-2 pointed at."""

import torch
from torch.utils.data import random_split

from img_dataset import MNISTImageDataset
from img_train import train

full_train = MNISTImageDataset(train=True)
test_ds = MNISTImageDataset(train=False)
val_ds, train_ds = random_split(
    full_train, [6000, len(full_train) - 6000],
    generator=torch.Generator().manual_seed(42))

base = dict(epochs=8, patience=3, batch_size=128, optimizer="adam", lr=1e-3)

runs = {
    "mlp_128":    dict(model="mlp", hidden=(128,)),
    "mlp_512":    dict(model="mlp", hidden=(512, 256)),
    "cnn_small":  dict(model="cnn", channels=(16,)),
    "cnn_32_64":  dict(model="cnn", channels=(32, 64)),
    "cnn_gap":    dict(model="cnn", channels=(32, 64), head="gap"),
}

from cnn_model import SimpleCNN, count_params

print(f"{'run':12s} {'conv':>8s} {'head':>8s} {'total':>9s} {'test_acc':>9s} {'time_s':>8s}")
for name, cfg in runs.items():
    res = train({**base, "run_name": f"ex_{name}", **cfg}, train_ds, val_ds, test_ds)
    # Split the parameter count: how much is feature extraction, how much is
    # the classifier on top? For a CNN these are wildly different numbers.
    if cfg["model"] == "cnn":
        m = SimpleCNN(channels=tuple(cfg["channels"]), head=cfg.get("head", "flatten"))
        conv = sum(p.numel() for p in m.features.parameters())
        head = sum(p.numel() for p in m.classifier.parameters())
    else:
        conv, head = 0, res["n_params"]
    print(f"{name:12s} {conv:8d} {head:8d} {res['n_params']:9d} "
          f"{res['test_acc']:9.4f} {res['train_time']:8.1f}")
```

### What to look for

- The **CNN should beat both MLPs**, including `cnn_small`, whose convolutional stack is a *single* 1→16 layer.
- **Look hard at the `conv` and `head` columns before concluding anything about efficiency.** `cnn_small`'s convolution has about **160 parameters**; its flatten head has about **400,000**. The feature extractor is almost free and the classifier is almost everything. So a CNN with a `flatten` head is *not* automatically a small model — and any claim that "the CNN wins with fewer parameters" has to say *which* parameters it means.
- `cnn_gap` replaces the flatten head with global average pooling and drops to roughly **28,000** parameters — a 15× reduction — but **loses several points of accuracy** at this budget. That is a real trade, not a free win: averaging over the 7×7 map throws away *where* each feature fired, and for digit shapes position matters.
- The margin on plain MNIST will look **modest** (a point or two between the better models). MNIST is nearly saturated; both families do well on centred, clean digits. Do not oversell it.
- **The honest version of this comparison is Section 5.4**, on shifted digits, where the architectures separate properly.

> 🔎 **The real question this table raises.** If the convolution is only 160 parameters, is the CNN winning because of *convolution*, or just because it has a big dense head like the MLP does? The exercise asks you to design a comparison that separates parameter *sharing* from parameter *count* — this table is where that question comes from, and `cnn_gap` is one end of the answer.

> ⚠️ **MNIST is an easy, nearly-solved dataset, and every architecture looks good on it.** Treat small differences here with the same suspicion Lab 7 taught you to treat small differences in market data: run more than one seed before believing a gap.

---

## 5️⃣ Experiments

Every experiment is a config change with a new `run_name`, overlaid in TensorBoard.

### 5.1 Kernel size and depth

Sweep, on the full training set:

| # | What varies | Values |
|---|---|---|
| 1 | `kernel_size` | 3, 5, 7 |
| 2 | `channels` | `(16,)`, `(32,64)`, `(32,64,128)` |
| 3 | `head` | `flatten`, `gap` |

**Task**

1. One table: config, parameter count, receptive field (`receptive_field(model.layer_spec)`), best epoch, test accuracy, training time.
2. Does a 5×5 kernel beat two stacked 3×3 layers at a comparable receptive field? Compare at matched **parameter count**, not matched depth.
3. `head="gap"` removes most of the classifier's parameters. How much accuracy does that cost, and what do you get in return?

### 5.2 Pooling versus strided convolution

Both halve the spatial size. Max-pooling picks the strongest activation in each 2×2 window and has no parameters; a stride-2 convolution *learns* how to downsample, at the cost of parameters.

Run `downsample ∈ {"max", "avg", "stride"}`, three seeds.

**Task**

1. Report accuracy, parameters and training time for each.
2. Max vs average pooling: which suits MNIST's thin bright strokes on a dark background, and why?
3. Explain why max-pooling gives a little **translation invariance**, and why that is a different thing from the translation **equivariance** of the convolution itself.

### 5.3 BatchNorm — and `model.eval()` for the second time

BatchNorm normalises each channel using the **statistics of the current batch** during training, while keeping a running average to use at evaluation time. That makes it the second layer — after Dropout in Lab 6-2 — whose behaviour depends on `model.train()` vs `model.eval()`.

Run `norm ∈ {"none", "batch"}` with `lr ∈ {1e-3, 1e-2}`.

**Task**

1. Does BatchNorm help accuracy, or mainly **speed of convergence**? Compare the epoch at which each run first reaches 98% validation accuracy, not just the final number.
2. BatchNorm is supposed to make training tolerant of a larger learning rate. Does the `lr=1e-2` run confirm that?
3. **Reproduce the `eval()` bug on BatchNorm.** Evaluate a trained BatchNorm model five times in `train()` mode and five times in `eval()` mode, using a helper that does **not** call `evaluate()` (which would repair it silently, exactly as in Lab 6-2). Then do the same with `batch_size=1`.

   > ⚠️ A BatchNorm model evaluated in `train()` mode with **batch size 1** normalises each sample by its own statistics, which destroys the signal. This is the single nastiest version of the forgotten-`eval()` bug, and it is why inference code must always call `model.eval()`.

### 5.4 Data augmentation and the shifted test set

This is the experiment that shows what convolution is actually for.

MNIST digits are **centred**, so a model can do well by memorising absolute positions. Build a test set where that stops working:

```python
from img_dataset import MNISTImageDataset

train_plain = MNISTImageDataset(train=True, shift_max=0)
train_aug   = MNISTImageDataset(train=True, shift_max=4)                 # augmentation
test_plain  = MNISTImageDataset(train=False)
test_shift  = MNISTImageDataset(train=False, shift_max=4, fixed_shift=True)  # fixed, reproducible
```

Train four models — `{mlp, cnn} × {plain, augmented}` — and evaluate each on **both** test sets. That is a 4×2 table.

**Task**

1. Fill in the table. Which cell drops the most, and which model degrades most gracefully without augmentation?
2. Augmentation costs you something on the *plain* test set in exchange for robustness on the shifted one. Quantify that trade.
3. Convolution is translation **equivariant**, so why does an un-augmented CNN still lose accuracy on shifted digits? (Think about what the `flatten` head does to a shifted feature map — and try `head="gap"` to test your explanation.)
4. Look at `inputs/train` in TensorBoard's IMAGES tab for the augmented run. Do the shifted digits look right, or is there a visible border artifact?

### 5.5 What the filters learn

```python
import matplotlib.pyplot as plt
import torch

from cnn_model import SimpleCNN

ckpt = torch.load("checkpoints/ex_cnn_32_64/best.pt", map_location="cpu")
model = SimpleCNN(channels=(32, 64))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# --- first-layer kernels: 32 filters of 1x3x3 ---
w = model.features[0].weight.detach()            # (32, 1, 3, 3)
fig, axes = plt.subplots(4, 8, figsize=(10, 5))
for ax, kernel in zip(axes.ravel(), w):
    ax.imshow(kernel[0], cmap="gray")
    ax.axis("off")
fig.suptitle("Layer-1 kernels")
plt.tight_layout()
plt.show()
```

**Task**

1. Plot the 32 first-layer kernels. Can you spot edge detectors at different orientations? (At 3×3 they are small and noisy — say honestly what you can and cannot see.)
2. Use `model.feature_maps(x)` on one test digit and plot the activations after each block. Which parts of the digit light up at layer 1 versus layer 2?
3. Feature maps get **smaller and more numerous** with depth. Relate that to the receptive-field arithmetic from Section 1.

### 5.6 Overfitting on a small subset

As in Lab 6-2, a model on full MNIST barely overfits. Subsample to make the effect visible:

```python
small_train = MNISTImageDataset(train=True, limit=1000)
```

**Task** Sweep `dropout ∈ {0, 0.25, 0.5}` and `weight_decay ∈ {0, 1e-4, 1e-2}` on the small subset, and compare against adding `shift_max=4` augmentation instead. Watch `generalization_gap` in TensorBoard. **Which is the stronger regulariser here — dropout, weight decay, or more data through augmentation?**

---

## 6️⃣ Inference: Loading `best.pt`

```python
"""predict_digit.py - classify one image with a saved checkpoint."""

import cv2
import numpy as np
import torch

from cnn_model import SimpleCNN
from img_dataset import MNIST_MEAN, MNIST_STD

CHECKPOINT_PATH = "checkpoints/ex_cnn_32_64/best.pt"   # hardcoded - change
IMAGE_PATH = "my_digit.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
config = ckpt["config"]
print(f"Loaded epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

model = SimpleCNN(
    channels=tuple(config.get("channels", (32, 64))),
    kernel_size=config.get("kernel_size", 3),
    norm=config.get("norm", "none"),
    downsample=config.get("downsample", "max"),
    head=config.get("head", "flatten"),
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()                 # disables dropout AND switches BatchNorm to running stats

gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError(IMAGE_PATH)
gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

# Preprocess EXACTLY as training did: scale to [0,1], then normalize.
x = torch.tensor(gray, dtype=torch.float32)[None, None] / 255.0
x = ((x - MNIST_MEAN) / MNIST_STD).to(device)

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]   # softmax HERE, not in forward()

pred = int(probs.argmax())
print(f"Predicted: {pred}  (confidence {probs[pred]:.3f})")
for score, idx in zip(*[t.tolist() for t in torch.topk(probs, 3)]):
    print(f"  {idx}: {score:.3f}")
```

> ⚠️ **MNIST is white-on-black.** A digit you photograph or draw in most paint programs is black-on-white, and the model will be confidently wrong. Invert it (`gray = 255 - gray`) and check the digit is centred and roughly 20 px tall with a 4 px margin, which is how MNIST was built. This preprocessing mismatch — not the model — is the usual reason "it works on the test set but not on my own digit".

---

## 🧪 Student Exercise (Evaluative) — Architecture and Robustness Study

### Part 1 — The architecture grid

On MNIST, train every combination of:

- **`channels`:** `(16,)`, `(32, 64)`, `(32, 64, 128)`
- **`downsample`:** `max`, `stride`
- **`norm`:** `none`, `batch`

That is 12 configurations × 3 seeds. Report, for each: parameter count, receptive field, best epoch, test accuracy (mean ± std), training time.

**Required baselines in the same table:** Lab 6's `mlp_128` and a larger `mlp_512` — *trained by this lab's `train()` on this lab's data*, so the comparison is controlled.

### Part 2 — Robustness

Build the 4×2 table from Section 5.4 (`{mlp, cnn} × {plain, augmented}` evaluated on plain and shifted test sets), and extend it:

1. Add a **rotated** test set (±15°) as a third column. Convolution is not rotation-equivariant — does rotation augmentation fix it, and does it cost you on the plain test set?
2. Report the **parameter-matched** comparison: pick an MLP and a CNN within 10% of each other on parameter count, and compare both test columns.

### Part 3 — Data efficiency

Train your best CNN and your best MLP on `limit ∈ {500, 2000, 10000, 60000}` training images. Plot test accuracy against training-set size, log x-axis, both models on one figure.

### Requirements

1. All results as tables with mean ± std over three seeds. State the seeds.
2. Overlay all runs in one TensorBoard; include SCALARS and HPARAMS screenshots, plus one IMAGES screenshot showing your augmented inputs.
3. Apply the mechanics consistently (early stopping, whichever regularisation helped) and **say what you applied**.
4. Visualise the first-layer kernels of your best model and comment on what, if anything, is interpretable.
5. Submit the `best.pt` of your best configuration and a `predict_digit.py` run on a digit **you drew or photographed yourself** — including the preprocessing you had to do to make it work.

### Analysis — answer these specifically

- **Where does the CNN's advantage actually come from?** Design a comparison that separates *parameter sharing* from *parameter count*. (One idea among several: an MLP with roughly the CNN's parameter count, and a CNN shrunk to roughly the MLP's.)
- **How much of the plain-MNIST gap survives on shifted digits?** If the gap grows, explain why in terms of equivariance and the classifier head.
- **Is BatchNorm worth it here?** Separate "higher final accuracy" from "gets there in fewer epochs" from "tolerates a larger learning rate" — these are three different claims and your table should answer each.
- **Data efficiency.** At which training-set size do the CNN and MLP curves separate most? Why would the inductive bias matter most when data is scarce?
- **What did augmentation cost you?** Report the plain-test-set accuracy you gave up, and argue whether the trade was worth it.

### Some things to think about

- MNIST is nearly saturated, so a 0.3% difference may be noise. Your seed spread tells you what counts as a real gap — the Lab 7 discipline applies unchanged.
- A convolution is translation equivariant but **not** rotation or scale equivariant. Which of the invariances you actually need should come from the architecture, and which from augmentation?
- The CNN reads the whole image through a stack of **local** windows, growing its receptive field layer by layer. The next lab replaces that with a mechanism where every position can look at every other position **in one step** — and asks what you give up for it.

> 📌 No reference solution is included in this lab sheet.

---
