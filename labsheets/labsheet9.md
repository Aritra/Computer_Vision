# 🖥️ Computer Vision Lab 9 — Vision Transformers vs CNNs on CODaN (Day → Night)

## 🎯 Objectives

In this lab, you will:

1. Implement **scaled dot-product attention** from scratch and verify it against `nn.MultiheadAttention`.
2. Prove self-attention is **permutation equivariant**, and see why that forces **positional encoding** to exist.
3. Turn an image into **patch tokens**, and see that patch embedding *is* a strided convolution.
4. Build a **Vision Transformer** and train it with Lab 8's `train()` — unchanged.
5. Work with **CODaN**, a real 10-class colour dataset of everyday objects, instead of MNIST.
6. Compare **ViT vs CNN** on equal terms: accuracy, parameters, wall-clock, and **data efficiency**.
7. Measure the **day → night domain gap** — a *real* distribution shift, not a synthetic one.
8. **Visualize attention maps** and judge honestly how interpretable they are.

> 📌 **This lab answers the question Lab 7 ended on.** A recurrent model "processes a series one step at a time... slow compared with architectures that look at all positions in parallel." **Attention is that architecture.** Lab 8 then showed what a *spatial prior* buys you; this lab asks what happens when you discard that prior — on images hard enough for the answer to matter.

> 📌 **Why not MNIST?** Because every architecture scores ~98% on it, so the comparison is uninformative. CODaN is small enough to train on a laptop CPU but genuinely difficult, **and it ships separate day and night test sets** — which turns Lab 8's synthetic shifted-digit experiment into a real-world robustness measurement.

> 📌 **Reuses Lab 8's `cnn_model.py` and `img_train.py` unchanged.** Two new files: `codan_dataset.py`, and `vit_model.py`. Lab 8's `train()` already accepts a ready-made `model=`, which is exactly the extension point both a ViT and a 3-channel CNN need.

## 📖 Background: Three Ways to Mix Information

Every architecture in this course answers one question: **how does information at position A reach position B?**

| | How positions interact | Cost per layer | Path A→B | Built-in prior |
|---|---|---|---|---|
| **MLP** (Lab 6) | all-to-all, fixed weights | `O(n²)` params | 1 | none — position is just an index |
| **RNN/LSTM** (Lab 7) | sequentially, through a state | `O(n·d²)`, **not parallel** | `O(n)` | recency / order |
| **CNN** (Lab 8) | local window, shared kernel | `O(n·k·d²)` | `O(n/k)` layers | locality + translation |
| **Self-attention** (this lab) | all-to-all, **weights computed from the data** | `O(n²·d)` | **1** | **none** |

The last row is the crucial one. In an MLP the weight connecting A to B is *learned and fixed*. In self-attention it is **computed from the content of A and B at run time**, so the same layer attends differently to every input. That is what "attention" means, and why the path between any two positions is length 1.

**The price:** `O(n²)` comparisons per layer, and **no built-in prior at all**. A CNN knows neighbouring pixels are related before it sees one example. A transformer must learn that from data — which is why transformers are famously data-hungry, and why 10,000 training images is a hard test for one.

### Scaled dot-product attention

Each token emits a **query** (what am I looking for?), a **key** (what do I offer?) and a **value** (what do I pass on?). Token `i` attends to token `j` in proportion to how well `q_i` matches `k_j`:

```
    Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

> 🔎 **Why divide by `√d_k`?** `q·k` sums `d_k` products. With roughly independent unit-variance components the dot product has variance `d_k`, so it grows with dimension. Large values saturate the softmax — one weight goes to ~1, the rest to ~0 — and almost no gradient reaches the ignored tokens. Dividing by `√d_k` keeps the variance near 1 at any head size. You will measure this in Section 1.

---

## 0️⃣ Prerequisites

- The **`cv-dl`** environment and kernel from Lab 6-1.
- **`cnn_model.py` and `img_train.py` from Lab 8**, in the same folder. This lab does not modify them.
- `pillow` (already present via torchvision).

### Getting CODaN

**Common Objects Day and Night** (CODaN) is a 10-class colour dataset built from COCO, ImageNet and ExDark, by Lengyel, Garg, Milford and van Gemert (TU Delft, MIT licence).

```bash
conda activate cv-dl
git clone https://github.com/Attila94/CODaN.git
cd <your lab9 folder>
```

| Split | Images | Source |
|---|---|---|
| `train` | 10,000 | COCO + ImageNet (daytime) |
| `val` | 500 | COCO + ImageNet (daytime) |
| `test_day` | 2,500 | COCO + ImageNet (daytime) |
| `test_night` | 2,500 | **ExDark** (genuine low-light photographs) |

Ten classes: `Bicycle, Car, Motorbike, Bus, Boat, Cat, Dog, Bottle, Cup, Chair`. Native resolution is 224×224.

> ⚠️ **The download is about 370 MB**, and the training split is stored as **three part-files** (`codan_train.tar.bz2.part0/1/2`) because GitHub caps single files at 100 MB. They must be concatenated before extraction. `codan_dataset.py` does this for you — and, unlike the project's own loader, it **does not delete the parts afterwards**, so a failed extraction does not force you to re-download.

> 💡 **No download, or no network?** `codan_dataset.py` falls back to a **synthetic** 10-class dataset with a built-in day/night shift, so every section below still runs. It uses the same split sizes as real CODaN and is calibrated so a small CNN loses roughly **40% of its daytime accuracy at night** — about what the published ResNet-18 baseline loses on the real thing. The absolute numbers will not match real CODaN and the images are abstract textures rather than photographs, so **say which you used in your report**.

### Published reference point

The dataset's authors report a **ResNet-18** baseline at full 224×224 resolution:

| | Day accuracy | Night accuracy |
|---|---|---|
| ResNet-18 (224×224) | **80.39% ± 0.38** | **48.31% ± 1.33** |

> ⚠️ **Do not expect to match this.** You will train small models from scratch at 64×64 on a CPU; that reference uses 12× the pixels and a much larger network. **The number to pay attention to is not the accuracy — it is the ~32-point day→night gap.** That gap is what this lab is about, and you should be able to reproduce *it* even if your absolute numbers are far lower.

### Check you're ready

```python
import sys, torch
print("python:", sys.executable)         # must contain 'cv-dl'
from cnn_model import SimpleCNN
from img_train import train
print("Lab 8 modules imported OK")
```

```python
%load_ext tensorboard
%tensorboard --logdir runs
```

---

## 1️⃣ Self-Attention from Scratch

```python
import math
import torch


def scaled_dot_product_attention(q, k, v, scale=None):
    """q, k, v: (..., seq, d). Returns (output, attention_weights).

    attention_weights[..., i, j] = how much token i attends to token j.
    Each row sums to 1, because the softmax is over the LAST axis (j).
    """
    d_k = q.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(d_k)
    scores = (q @ k.transpose(-2, -1)) * scale
    weights = scores.softmax(dim=-1)
    return weights @ v, weights
```

### Why `√d_k` matters — measure it

```python
torch.manual_seed(0)
for d_k in (8, 64, 512):
    q, k = torch.randn(1, 50, d_k), torch.randn(1, 50, d_k)
    raw = q @ k.transpose(-2, -1)
    for name, s in [("unscaled", 1.0), ("scaled", 1.0 / math.sqrt(d_k))]:
        w = (raw * s).softmax(dim=-1)
        entropy = -(w * w.clamp_min(1e-9).log()).sum(-1).mean()
        print(f"d_k={d_k:4d} {name:9s} score_std={raw.std() * s:7.3f} "
              f"max_weight={w.max():.3f} entropy={entropy:.3f}")
```

A collapsed softmax (max weight near 1, entropy near 0) passes almost no gradient to the tokens it ignored.

### Task

1. Run the `√d_k` experiment. At `d_k = 512`, what is the max attention weight without scaling, and why does that kill the gradient?
2. **Verify against PyTorch.** Build `nn.MultiheadAttention(dim, heads, batch_first=True)` and your `MultiHeadSelfAttention` from Section 3, copy the weights across (`in_proj_weight` ↔ `qkv.weight`, `out_proj` ↔ `proj`), and assert the outputs match to `1e-5`.
3. **Prove permutation equivariance:** show `attention(x[:, p]) == attention(x)[:, p]` for a permutation `p`. Then explain why this means a transformer cannot tell a cup from an upside-down cup unless something tells it where each patch came from.
4. Count parameters of `MultiHeadSelfAttention(dim=64, heads=h)` for `h ∈ {1, 2, 4, 8}`. Confirm they are identical, and say what *does* change.

---

## 2️⃣ `codan_dataset.py`: Real Images, Cached Small

CODaN ships 224×224 images. Training on those with a CPU is not realistic for a lab, so we **resize once and cache the result** — exactly the pattern from Lab 6-2's LBP cache: pay the cost a single time, then every later run loads a tensor instantly.

```
   224×224 JPEG on disk
        └─> resize to size×size, convert to RGB
             └─> stack into one uint8 tensor  (N, 3, size, size)
                  └─> torch.save to cache/codan_<split>_<size>.pt
```

> ⚠️ **Do not follow the official loader's approach of holding every image in memory as a PIL object.** That costs roughly 2.5 GB across all four splits. A cached `uint8` tensor at 64×64 is about **123 MB** for the 10,000 training images, and 31 MB at 32×32.

> 💡 **Which resolution?** Use **64×64** for the worked example and Section 5's main experiments, and drop to **32×32** for the large sweeps in the student exercise — otherwise the 36-run grid becomes an overnight job. The sheet says which to use where; the day→night gap is visible at both.

```python
"""codan_dataset.py - CODaN (Common Objects Day and Night), resized and cached.

Dataset: https://github.com/Attila94/CODaN  (MIT licence)
Lengyel, Garg, Milford, van Gemert - "Zero-Shot Day-Night Domain Adaptation
with a Physics Prior", ICCV 2021.
"""

import math
import os
import shutil
import tarfile

import numpy as np
import torch
from torch.utils.data import Dataset

CODAN_CLASSES = ["Bicycle", "Car", "Motorbike", "Bus", "Boat",
                 "Cat", "Dog", "Bottle", "Cup", "Chair"]
SPLITS = ("train", "val", "test_day", "test_night")


# ------------------------------------------------------------------ unpacking
def _join_train_parts(data_dir):
    """Concatenate codan_train.tar.bz2.part{0,1,2} into one archive.

    GitHub rejects files over 100 MB, so the training split is stored in three
    pieces. Unlike the project's own loader we KEEP the parts after joining:
    deleting them means a failed extraction costs you the whole download again.
    """
    target = os.path.join(data_dir, "codan_train.tar.bz2")
    if os.path.exists(target):
        return target
    parts = [os.path.join(data_dir, f"codan_train.tar.bz2.part{i}") for i in range(3)]
    missing = [p for p in parts if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"missing train part(s): {missing}")
    print("[codan] joining 3 training parts...")
    with open(target, "wb") as out:
        for part in parts:
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out)
    return target


def _extract(data_dir, split):
    out_dir = os.path.join(data_dir, split)
    if os.path.isdir(out_dir):
        return out_dir
    archive = (_join_train_parts(data_dir) if split == "train"
               else os.path.join(data_dir, f"codan_{split}.tar.bz2"))
    if not os.path.exists(archive):
        raise FileNotFoundError(archive)
    print(f"[codan] extracting {split} (one time)...")
    with tarfile.open(archive) as tar:
        try:
            # Python 3.12+ wants an explicit extraction filter; older versions
            # do not accept the argument at all.
            tar.extractall(path=data_dir, filter="data")
        except TypeError:
            tar.extractall(path=data_dir)
    return out_dir


# -------------------------------------------------------------------- caching
def _build_split(data_dir, split, size):
    """Read every JPEG of one split, resize, and stack into a uint8 tensor."""
    from PIL import Image

    split_dir = _extract(data_dir, split)
    images, targets = [], []
    for class_id, class_name in enumerate(CODAN_CLASSES):
        class_dir = os.path.join(split_dir, class_name)
        files = sorted(f for f in os.listdir(class_dir)
                       if f.lower().endswith((".jpg", ".jpeg")))
        for fname in files:
            with Image.open(os.path.join(class_dir, fname)) as im:
                im = im.convert("RGB").resize((size, size), Image.BILINEAR)
                images.append(torch.from_numpy(np.asarray(im, dtype=np.uint8)))
            targets.append(class_id)
    x = torch.stack(images).permute(0, 3, 1, 2).contiguous()   # (N, 3, size, size)
    return x, torch.tensor(targets, dtype=torch.long)


def make_synthetic_codan(split, size, seed=0, night_scale=0.53, noise=0.12):
    """Fallback: 10 synthetic classes with a built-in day/night shift.

    Each class is a hue on the colour wheel plus an oriented stripe texture,
    placed at a random position under a Gaussian envelope. Hue, orientation,
    frequency, position and brightness are all jittered per sample, so classes
    OVERLAP and the task is genuinely non-trivial rather than a giveaway.

    The 'test_night' split is darkened and noisier. `night_scale` is calibrated
    so a small CNN loses roughly 40% of its daytime accuracy at night, which is
    about what the published ResNet-18 baseline loses on real CODaN
    (80.4% -> 48.3%). Split sizes match the real dataset.

    This is a fallback, not a substitute: use it when the download is
    unavailable, and say so in your report.
    """
    counts = {"train": 10000, "val": 500, "test_day": 2500, "test_night": 2500}
    n = counts[split]
    # A FIXED offset per split: hash() is randomised per process, so using it
    # here would silently give different "synthetic CODaN" data every run.
    g = torch.Generator().manual_seed(seed + SPLITS.index(split) * 1000)
    y = torch.randint(0, len(CODAN_CLASSES), (n,), generator=g)

    coords = torch.linspace(-1, 1, size)
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    x = torch.zeros(n, 3, size, size)
    n_cls = len(CODAN_CLASSES)
    for i in range(n):
        c = int(y[i])
        angle = c * math.pi / n_cls + 0.3 * torch.randn(1, generator=g).item()
        freq = 4 + (c % 3) + torch.randn(1, generator=g).item()
        # Class-dependent position, jittered. This matters: if the object sat
        # at a RANDOM position, class would be carried entirely by hue and
        # texture - both invariant to patch order - and Section 5.3's
        # patch-shuffle ablation would show no effect at all, because the task
        # would be a bag-of-patches problem by construction. Real objects have
        # spatial structure, so the synthetic ones must too.
        ox = 0.45 * math.cos(2 * math.pi * c / n_cls) + 0.12 * torch.randn(1, generator=g).item()
        oy = 0.45 * math.sin(2 * math.pi * c / n_cls) + 0.12 * torch.randn(1, generator=g).item()
        stripes = torch.sin(freq * ((gx - ox) * math.cos(angle)
                                    + (gy - oy) * math.sin(angle)))
        hue_angle = 2 * math.pi * c / n_cls + 0.25 * torch.randn(1, generator=g).item()
        hue = torch.tensor([0.5 + 0.5 * math.cos(hue_angle),
                            0.5 + 0.5 * math.cos(hue_angle - 2.1),
                            0.5 + 0.5 * math.cos(hue_angle + 2.1)])
        blob = torch.exp(-((gx - ox) ** 2 + (gy - oy) ** 2) / 0.6)
        x[i] = ((0.35 + 0.4 * stripes * blob)[None] * hue[:, None, None]
                + 0.15 * torch.rand(1, generator=g).item())
    x = x + noise * torch.randn(x.shape, generator=g)

    if split == "test_night":
        # A low-light shift is NOT just "darker". Fewer photons means a darker
        # image, lower contrast, a colour cast from artificial lighting, and
        # more sensor noise. Modelling only brightness would make the shift
        # trivially undoable by brightness augmentation alone, which would
        # teach the wrong lesson in Section 5.4.
        cast = torch.tensor([0.85, 0.95, 1.20])[None, :, None, None]   # cool cast
        x = x * night_scale * cast
        mu = x.mean(dim=(2, 3), keepdim=True)
        x = mu + (x - mu) * 0.75                                        # less contrast
        x = x + 0.09 * torch.randn(x.shape, generator=g)                # sensor noise

    return (x.clamp(0, 1) * 255).to(torch.uint8), y


def load_split(root="./CODaN", split="train", size=64, cache_dir="./cache"):
    """Returns (images uint8 (N,3,size,size), targets int64), cached on disk."""
    assert split in SPLITS, f"split must be one of {SPLITS}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"codan_{split}_{size}.pt")

    if os.path.exists(cache_path):
        blob = torch.load(cache_path)
        return blob["images"], blob["targets"]

    data_dir = os.path.join(root, "data")
    try:
        images, targets = _build_split(data_dir, split, size)
        print(f"[codan] {split}: {len(targets)} images at {size}x{size}")
    except (FileNotFoundError, OSError) as exc:
        print(f"[codan] WARNING: could not read real CODaN ({type(exc).__name__}: {exc}).")
        print("[codan] Falling back to SYNTHETIC data - say so in your report.")
        images, targets = make_synthetic_codan(split, size)

    torch.save({"images": images, "targets": targets}, cache_path)
    return images, targets


def _channel_stats(images_u8, chunk=1024):
    """Per-channel mean/std over a uint8 tensor, in chunks.

    Computed chunk-wise so we never hold the whole split as float32 at once.
    """
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    count = 0
    for i in range(0, len(images_u8), chunk):
        x = images_u8[i:i + chunk].double() / 255.0
        total += x.sum(dim=(0, 2, 3))
        total_sq += (x ** 2).sum(dim=(0, 2, 3))
        count += x.shape[0] * x.shape[2] * x.shape[3]
    mean = total / count
    var = (total_sq / count - mean ** 2).clamp_min(1e-12)
    return mean.float(), var.sqrt().float().clamp_min(1e-6)


# -------------------------------------------------------------------- dataset
class CODaNDataset(Dataset):
    """CODaN split as normalized float tensors, with optional augmentation.

    stats : (mean, std) per channel. Pass the TRAIN split's stats to every
            other split - fitting normalisation on val/test would leak, the
            same rule as Lab 4's vocabulary and Lab 6-1's StandardScaler.
    """

    def __init__(self, root="./CODaN", split="train", size=64, limit=None,
                 stats=None, augment=False, cache_dir="./cache", random_state=42):
        images, targets = load_split(root, split, size, cache_dir)

        if limit is not None and limit < len(targets):
            g = torch.Generator().manual_seed(random_state)
            keep = torch.randperm(len(targets), generator=g)[:limit]
            images, targets = images[keep], targets[keep]

        # Keep the uint8 tensor and normalise per item. Materialising the whole
        # split as float32 would cost ~500 MB at 64x64 for 10,000 images, and
        # augmentation would copy it again.
        self.images_u8 = images
        self.targets = targets
        self.augment = augment
        self.size = size

        if stats is None:                       # fit on THIS split (train only!)
            self.mean, self.std = _channel_stats(images)
        else:
            self.mean, self.std = stats

    @property
    def stats(self):
        return (self.mean, self.std)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.images_u8[idx].float() / 255.0
        img = (img - self.mean[:, None, None]) / self.std[:, None, None]
        if self.augment:
            if torch.rand(1).item() < 0.5:                 # horizontal flip
                img = img.flip(-1)
            # brightness jitter: the day/night shift is largely an illumination
            # change, so this is the augmentation most likely to matter here.
            img = img * (0.6 + 0.8 * torch.rand(1).item())
        return img, self.targets[idx]


def build_codan(root="./CODaN", size=64, limit=None, augment=False, cache_dir="./cache"):
    """Returns (train, val, test_day, test_night) with TRAIN-fitted normalisation."""
    train = CODaNDataset(root, "train", size, limit, None, augment, cache_dir)
    stats = train.stats
    rest = [CODaNDataset(root, s, size, None, stats, False, cache_dir)
            for s in ("val", "test_day", "test_night")]
    return (train, *rest)
```

### Task

1. Build all four splits at `size=64` and report the shapes and counts. Do they match the table in Section 0?
2. Display a 10×4 grid: four example images per class. Then display four `test_night` images beside four `test_day` images of the same class. **Describe in two sentences what a model trained only on daytime images is up against.**
3. Report the per-channel mean and std of the train split, and of `test_night`. How far apart are they? That distance *is* the domain gap, measured in input space before any model sees it.
4. Why must `val`, `test_day` and `test_night` all be normalised with the **training** statistics rather than their own?

---

## 3️⃣ `vit_model.py`

```python
"""vit_model.py - a Vision Transformer, written out block by block."""

import math

import torch
from torch import nn


def scaled_dot_product_attention(q, k, v, scale=None):
    """q, k, v: (..., seq, d) -> (output, attention weights)."""
    d_k = q.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(d_k)
    scores = (q @ k.transpose(-2, -1)) * scale
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


class MultiHeadSelfAttention(nn.Module):
    """One fused qkv projection, h heads, one output projection.

    Parameter layout matches nn.MultiheadAttention (in_proj_weight is [q; k; v]
    stacked along dim 0), so you can copy weights across and check - the same
    exercise as Lab 7's ManualLSTM and Lab 8's ManualConv2d.
    """

    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)              # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        out, attn = scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, D)    # concatenate the heads
        out = self.drop(self.proj(out))
        return (out, attn) if return_attention else (out, None)


class TransformerBlock(nn.Module):
    """Pre-norm block: x = x + attn(norm(x)); x = x + mlp(norm(x)).

    Pre-norm rather than the original post-norm, because it trains stably
    without a learning-rate warmup - which matters when you have a lab
    session rather than a TPU pod.
    """

    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x, return_attention=False):
        h, attn = self.attn(self.norm1(x), return_attention)
        x = x + h                      # residual: the gradient highway, as in Lab 7's LSTM
        x = x + self.mlp(self.norm2(x))
        return (x, attn) if return_attention else (x, None)


class VisionTransformer(nn.Module):
    """(B, 3, img_size, img_size) -> (B, num_classes).

    pos_embed : set False to ABLATE position information (Section 5.3)
    pool      : "cls" (prepended token) | "mean" (average of patch tokens)
    Returns raw logits - softmax at display time, as in every lab since 6-1.
    """

    def __init__(self, img_size=64, patch_size=8, in_channels=3, num_classes=10,
                 dim=96, depth=4, num_heads=4, mlp_ratio=2.0, dropout=0.0,
                 pos_embed=True, pool="cls"):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"patch_size {patch_size} must divide img_size {img_size}; "
                f"otherwise Conv2d silently drops the leftover strip")
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.pool = pool
        self.use_pos = pos_embed

        # Patch embedding IS a stride-p convolution (Section 4).
        self.patch_embed = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        n_tokens = self.num_patches + (1 if pool == "cls" else 0)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def tokenize(self, x):
        B = x.shape[0]
        t = self.patch_embed(x)                     # (B, dim, H/p, W/p)
        t = t.flatten(2).transpose(1, 2)            # (B, num_patches, dim)
        if self.pool == "cls":
            t = torch.cat([self.cls_token.expand(B, -1, -1), t], dim=1)
        if self.use_pos:
            t = t + self.pos
        return self.drop(t)

    def forward(self, x, return_attention=False):
        t = self.tokenize(x)
        attentions = []
        for block in self.blocks:
            t, attn = block(t, return_attention)
            if return_attention:
                attentions.append(attn)
        t = self.norm(t)
        summary = t[:, 0] if self.pool == "cls" else t.mean(dim=1)
        logits = self.head(summary)
        return (logits, attentions) if return_attention else logits


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

---

## 4️⃣ From Image to Tokens

```
   64×64 image, patch_size = 8
        ├─> 8×8 = 64 patches of 8×8×3
        ├─> linear projection -> dim-vector each   (the patch embedding)
        ├─> prepend a learned [CLS] token          -> 65 tokens
        └─> add a learned positional embedding
```

> 🔗 **Patch embedding is a strided convolution.** "Cut into non-overlapping `p×p` patches and apply the same linear map to each" *is* `nn.Conv2d(3, dim, kernel_size=p, stride=p)` — kernel size equal to stride means the windows tile without overlapping. So a ViT's first layer **is** a Lab 8 convolution; what follows it is the part that differs.

| `patch_size` (at 64×64) | tokens (+CLS) | attention cost | what a token sees |
|---|---|---|---|
| 4 | 256 + 1 | 257² ≈ 66k | fine detail, very long sequence |
| 8 | 64 + 1 | 65² ≈ 4.2k | a small object part |
| 16 | 16 + 1 | 17² = 289 | a quadrant-ish region |

> ⚠️ **`patch_size` must divide `img_size`.** At 64×64 that allows 4, 8, 16, 32. At 32×32 use 4, 8, 16. A size that does not divide leaves a strip that `Conv2d` silently discards.

### The `[CLS]` token

Classification needs **one** vector, but attention returns one per token. The standard trick prepends a single learned token belonging to no patch; after the blocks its output is the summary. Because attention is all-to-all, it can gather from every patch. The alternative is averaging all patch outputs (`pool="mean"`), compared in Section 5.2.

### Task

1. Verify `MultiHeadSelfAttention` against `nn.MultiheadAttention` (Section 1, task 2). **This is the core deliverable** — if it passes, your attention is PyTorch's attention.
2. Print `vit.tokenize(x).shape` for `patch_size ∈ {4, 8, 16}` at `img_size=64`. Confirm the token counts match the table.
3. Count parameters for `depth ∈ {2, 4, 8}` at `dim=96`. Derive the per-block count by hand (attention `4·dim²` plus biases; MLP `2·dim²·mlp_ratio` plus biases; two LayerNorms) and check.
4. What breaks at `patch_size=5`? The model raises — explain what `Conv2d` would otherwise have done.

---

## 5️⃣ Worked Example and Experiments

### 5.0 The worked example: ViT vs CNN on CODaN

Lab 8's `train()` takes a ready-made `model=`, so both models drop straight in — same loop, same data, same early stopping. **That is the only reason this comparison is fair.**

```python
"""run_codan_example.py - CNN vs ViT on CODaN, day and night."""

import torch
from torch.utils.data import DataLoader
from torch import nn

from codan_dataset import build_codan, CODAN_CLASSES
from cnn_model import SimpleCNN, count_params
from img_train import train, evaluate
from vit_model import VisionTransformer

SIZE = 64
train_ds, val_ds, day_ds, night_ds = build_codan(size=SIZE, augment=False)
print(f"train={len(train_ds)} val={len(val_ds)} "
      f"day={len(day_ds)} night={len(night_ds)}")

base = dict(epochs=15, patience=4, batch_size=128, optimizer="adamw",
            lr=1e-3, weight_decay=0.05, scheduler="cosine")

criterion, device = nn.CrossEntropyLoss(), torch.device("cpu")


def both_tests(model):
    day = evaluate(model, DataLoader(day_ds, batch_size=256), criterion, device)[1]
    night = evaluate(model, DataLoader(night_ds, batch_size=256), criterion, device)[1]
    return day, night


models = {
    "cnn":  lambda: SimpleCNN(in_channels=3, num_classes=10, channels=(32, 64, 128),
                              norm="batch", input_size=SIZE),
    "vit":  lambda: VisionTransformer(img_size=SIZE, patch_size=8, in_channels=3,
                                      dim=96, depth=4, num_heads=4, dropout=0.1),
}

print(f"\n{'model':6s} {'params':>9s} {'day':>8s} {'night':>8s} {'gap':>8s} {'time_s':>8s}")
for name, factory in models.items():
    model = factory()
    res = train({**base, "run_name": f"ex_{name}", "model": name},
                train_ds, val_ds, test_ds=None, model=model)
    # train() restores the best checkpoint before returning, so this is best.pt
    day, night = both_tests(model)
    print(f"{name:6s} {res['n_params']:9d} {day:8.4f} {night:8.4f} "
          f"{day - night:8.4f} {res['train_time']:8.1f}")
```

> 💡 **Note `augment=False` here.** The worked example deliberately trains *without* augmentation so the raw day→night gap is visible. Section 5.4 then adds augmentation as the variable under test — turning it on here would partly close the gap before you had measured it.

> ⚠️ **`train()` returns the model with the best checkpoint loaded**, which is why `both_tests` is called on `model` afterwards rather than re-loading from disk. If you refactor, keep that property — evaluating the *last* epoch instead of the *best* one is the mistake Lab 6 onwards has been guarding against.

### What to look for (and what *not* to expect)

- **Both models will score far below the 80% ResNet-18 reference.** You have 12× fewer pixels, a far smaller network, no ImageNet pre-training and a CPU budget. Judge the **day→night gap**, not the absolute number.
- **The CNN should beat the ViT**, probably by a wide margin. That is the correct result, not a bug: 10,000 images is nowhere near enough for a transformer to learn locality from scratch, and CODaN is much harder than MNIST. Section 5.5 measures this properly.
- **Both should lose a lot of accuracy at night.** Reproducing a large gap is the headline finding.
- Use **`adamw` with real weight decay** (0.05) and dropout; ViTs overfit 10k images very fast.

> ⚠️ **Do not tune the ViT until it wins.** The honest finding — *this architecture needs more data than CODaN provides* — is the entire point. A victory manufactured on a 2,500-image test set is exactly the self-deception Lab 7 warned about.

### 5.1 Patch size

Run `patch_size ∈ {4, 8, 16}` at `img_size=64`, fixed `dim=96, depth=4, heads=4`, three seeds.

**Task** Report tokens, parameters, day accuracy, night accuracy and seconds/epoch. Attention cost grows with the **square** of the token count — does accuracy grow fast enough to justify it? Where is the sweet spot, and why would it move on a 224×224 image?

### 5.2 Depth, width, heads, pooling

| # | What varies | Values |
|---|---|---|
| 1 | `depth` | 2, 4, 8 |
| 2 | `dim` | 48, 96, 192 |
| 3 | `num_heads` | 1, 2, 4, 8 |
| 4 | `pool` | `cls`, `mean` |

**Task** Which factor buys the most accuracy per parameter? Confirm `num_heads` changes **no** parameters, then say what it *does* change and whether it helped. Does removing the `[CLS]` token (`pool="mean"`) matter?

### 5.3 Positional encoding — the ablation that proves the point

Self-attention is **permutation equivariant**: shuffle the tokens and the outputs shuffle identically. Positional embeddings are the *only* thing telling a ViT where a patch came from.

Run three configurations: `pos_embed=True`; `pos_embed=False`; and `pos_embed=False` **with patches shuffled by a fixed permutation at test time**.

```python
import torch


def shuffled_logits(model, x, perm):
    """Scramble WHICH patch sits at each position, then run the blocks.

    The order here is the whole experiment. We embed the patches, permute
    them, and only THEN add the positional embedding - so patch `perm[i]`
    arrives carrying position `i`'s embedding. The image is scrambled while
    the position information stays put.

    Doing it the other way round (calling model.tokenize(), which already
    adds positions, and permuting afterwards) permutes complete
    content+position pairs - and attention is equivariant to that, so the
    output would be unchanged and the experiment would silently prove nothing.
    """
    B = x.shape[0]
    t = model.patch_embed(x).flatten(2).transpose(1, 2)   # (B, num_patches, dim)
    t = t[:, perm]                                        # shuffle patch CONTENT
    if model.pool == "cls":
        t = torch.cat([model.cls_token.expand(B, -1, -1), t], dim=1)
    if model.use_pos:
        t = t + model.pos                                 # positions stay in place
    for block in model.blocks:
        t, _ = block(t)
    t = model.norm(t)
    summary = t[:, 0] if model.pool == "cls" else t.mean(dim=1)
    return model.head(summary)
```

> ⚠️ **The order of "permute" and "add positions" is the entire experiment.** Shuffle *after* the positional embedding and you are permuting whole tokens, which attention is equivariant to — you will measure **exactly zero change even on a model that uses positions**, and wrongly conclude position does not matter. The code looks right and the number it prints is stable, which is what makes this trap dangerous.

**Task**

1. Report accuracy for all three. How much does removing positional embeddings cost?
> 💡 **If you are on the synthetic fallback, expect a *smaller* drop than on real CODaN.** The synthetic classes are still partly identifiable from hue and texture alone, which survive shuffling, so the `pos_embed=True` model loses some accuracy rather than collapsing. On real photographs, scrambling the patches of a cat destroys the cat, and the drop is far more dramatic. The qualitative result you must reproduce either way is the **exact** zero-change for `pos_embed=False`.

2. **The decisive check:** for `pos_embed=False`, normal and patch-shuffled accuracy should be **exactly identical** — not close, identical — because the model is mathematically a bag of patches. For `pos_embed=True` it should **collapse**. Report both and explain.
3. A ViT without positional embeddings is a bag of patches, yet it still scores well above the 10% chance level. What can you tell about a *cat* from an unordered multiset of its 8×8 patches — and why might that work better on CODaN than it would on MNIST digits?

### 5.4 The day → night domain gap

**This is the experiment CODaN exists for**, and the real-world counterpart of Lab 8's shifted-MNIST test.

Train on daytime images only (which is all `train` contains), then evaluate on `test_day` and `test_night`. Build this table:

| model | augmentation | day | night | gap |
|---|---|---|---|---|
| CNN | none | | | |
| CNN | flip + brightness | | | |
| ViT | none | | | |
| ViT | flip + brightness | | | |

**Task**

1. Fill the table, three seeds. Which model degrades **more gracefully** at night, in absolute terms and as a *fraction* of its own day accuracy?
2. Does brightness augmentation close the gap? By how much, and at what cost to day accuracy? (The shift is largely illumination, so this is the augmentation most likely to help — but check rather than assume.)
3. Compare your gap with the published ResNet-18 figures (80.39 → 48.31, a 32-point drop). Is your gap larger or smaller in **relative** terms, and what would explain the difference?
4. Per-class analysis: which classes survive the night best, and which collapse? Relate that to what a low-light photograph of each object actually looks like.

### 5.5 Data efficiency — where the inductive bias earns its keep

Train the best ViT and best CNN at `limit ∈ {1000, 2500, 5000, 10000}` training images, three seeds. **Use `size=32` here** to keep the runtime sane.

**Task** Plot day accuracy against training-set size (log x-axis) with error bars. At which size is the gap widest? Is it closing as data grows? Relate this to the Background table — which row explains it?

### 5.6 Attention maps

```python
import matplotlib.pyplot as plt
import torch

from codan_dataset import build_codan, CODAN_CLASSES
from vit_model import VisionTransformer

# Load a TRAINED model and real test images explicitly. Do not rely on `model`
# or `x` left over from an earlier cell - attention needs a (B, 3, H, W) image,
# and the nearest `x` in scope may well be a token tensor instead.
ckpt = torch.load("checkpoints/ex_vit/best.pt", map_location="cpu")
model = VisionTransformer(img_size=64, patch_size=8, in_channels=3,
                          dim=96, depth=4, num_heads=4, dropout=0.1)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

_, _, day_ds, _ = build_codan(size=64)
x = torch.stack([day_ds[i][0] for i in range(3)])

logits, attentions = model(x[:1], return_attention=True)
last = attentions[-1][0]                       # (heads, tokens, tokens)
cls_attn = last[:, 0, 1:]                      # what [CLS] looks at, per head
side = int(cls_attn.shape[-1] ** 0.5)

fig, axes = plt.subplots(1, cls_attn.shape[0] + 1, figsize=(13, 3))
raw = x[0].permute(1, 2, 0)
axes[0].imshow((raw - raw.min()) / (raw.max() - raw.min()))
axes[0].set_title(CODAN_CLASSES[int(day_ds[0][1])])
axes[0].axis("off")
for h, ax in enumerate(axes[1:]):
    ax.imshow(cls_attn[h].reshape(side, side).detach(), cmap="viridis")
    ax.set_title(f"head {h}")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

**Task**

1. Plot `[CLS]` attention for every head in the last block, for three different classes. Do heads specialise, or look alike?
2. Compare the first block's attention with the last. Which is more spatially focused?
3. Repeat on a **night** image of the same class. Does the attention go somewhere sensible, or wander?
4. **Be sceptical.** Attention weights are *not* an explanation of the prediction: a token can be heavily attended and barely influence the output (its value vector may be near zero), and residual connections carry information *around* the attention entirely. State one concrete reason an attention map could mislead you here.

### 5.7 Parameter-matched comparison

Accuracy gaps mean little if one model is simply bigger. Pick a ViT and a CNN **within 10% on parameter count** and compare day accuracy, night accuracy, seconds/epoch, and accuracy at `limit=2500`.

**Task** Which would you ship for CODaN, and which for a problem with 100× the data? Justify with your numbers, not with what you have read about transformers.

---

## 6️⃣ Inference: Loading `best.pt`

```python
"""predict_codan.py - classify one image with a saved checkpoint."""

import cv2
import numpy as np
import torch

from codan_dataset import CODAN_CLASSES, build_codan
from vit_model import VisionTransformer

CHECKPOINT_PATH = "checkpoints/ex_vit/best.pt"   # hardcoded - change
IMAGE_PATH = "my_photo.jpg"
SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
print(f"Loaded epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

model = VisionTransformer(img_size=SIZE, patch_size=8, in_channels=3,
                          dim=96, depth=4, num_heads=4).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# The normalisation MUST be the training split's statistics.
train_ds, *_ = build_codan(size=SIZE)
mean, std = train_ds.stats

bgr = cv2.imread(IMAGE_PATH)
if bgr is None:
    raise FileNotFoundError(IMAGE_PATH)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

x = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
x = ((x - mean[None, :, None, None]) / std[None, :, None, None]).to(device)

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]   # softmax HERE, not in forward()

pred = int(probs.argmax())
print(f"Predicted: {CODAN_CLASSES[pred]}  (confidence {probs[pred]:.3f})")
for score, idx in zip(*[t.tolist() for t in torch.topk(probs, 3)]):
    print(f"  {CODAN_CLASSES[int(idx)]:10s} {score:.3f}")
```

> ⚠️ **OpenCV reads BGR; CODaN was trained on RGB.** Forget `cvtColor` and you feed the network colour-swapped images — which matters far more here than it would on greyscale MNIST, because colour is a real cue for these classes. The other classic mistake is normalising with the *wrong* statistics; they must come from the training split, which is why they are loaded above rather than hardcoded.

---

## 🧪 Student Exercise (Evaluative) — ViT vs CNN Under a Real Domain Shift

### Part 1 — Build and verify

1. Your `MultiHeadSelfAttention` must match `nn.MultiheadAttention` to `1e-5`. **Show the assert passing** — the core deliverable, as `ManualLSTM` was in Lab 7 and `ManualConv2d` in Lab 8.
2. Demonstrate permutation equivariance of raw attention, and the `pos_embed=False` shuffle-invariance of the full model.
3. Report the dataset diagnostics from Section 2 (class grid, day-vs-night examples, channel statistics).

### Part 2 — The grid

**Use `size=32` for this grid**; a 64×64 sweep of this size is an overnight job.

- **`patch_size`:** 4, 8
- **`depth`:** 2, 4
- **`num_heads`:** 2, 4

12 ViT configurations × 3 seeds. Report parameters, tokens, best epoch, day accuracy, night accuracy (mean ± std), seconds/epoch.

**Required baselines in the same table**, trained by the same `train()` on the same data: Lab 8's CNN at two sizes, and a **parameter-matched** CNN/ViT pair.

### Part 3 — Day → night

The Section 5.4 table (model × augmentation × {day, night}), three seeds, plus per-class night accuracy for your best model of each family.

### Requirements

1. Mean ± std over three stated seeds, everywhere.
2. The positional-encoding ablation with all three configurations.
3. The data-efficiency curve (Section 5.5).
4. Overlay all runs in one TensorBoard; include SCALARS and HPARAMS screenshots.
5. Attention-map figures for at least three classes, including one night image, with an honest caption.
6. Submit the `best.pt` of your best model plus a `predict_codan.py` run on a photo **you took yourself** of one of the ten classes — ideally one taken at night.
7. State clearly whether you used **real CODaN or the synthetic fallback**, and at which resolution.

### Analysis — answer these specifically

- **Did the ViT beat the CNN?** If not — the likely outcome — explain *why* in terms of inductive bias and dataset size, and point to the evidence in your data-efficiency curve. A clearly argued negative result earns full marks; an unexplained one does not.
- **Which architecture is more robust to the night shift?** Compare absolute and relative degradation. Does the model with *less* built-in prior transfer better or worse, and did you expect that?
- **What does the positional embedding buy?** Quantify it and explain the shuffle result.
- **Separate size from architecture** using your parameter-matched pair: is any remaining gap about the *mechanism* or merely the *budget*?
- **Cost.** Compare seconds/epoch at matched parameters. Attention is `O(n²)` in tokens while convolution is linear in pixels — does your timing show that, and at which patch size does it bite?
- **Augmentation.** You augmented with flips and brightness. Which helped at night, which helped during the day, and what would you add next given what night images actually look like?

### Some things to think about

- Every model in this course has now mixed information differently: fixed all-to-all (MLP), sequential (RNN), local and shared (CNN), content-dependent all-to-all (attention). The **prior** each encodes decides how much data it needs.
- The ViT's first layer is a strided convolution. Hybrids keeping a few conv layers before the transformer blocks are popular precisely because they re-introduce the locality prior cheaply — a reasonable extension if you want one.
- The published ResNet-18 baseline reaches 80% by day and 48% at night **with ImageNet pre-training and 224×224 inputs**. Think about which of those two advantages you would spend your compute on first.
- Labs 7 and 8 taught that evaluation discipline matters more than architecture. Nothing about transformers changes that.

> 📌 No reference solution is included in this lab sheet.

---
