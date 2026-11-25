Labsheet 10 - All-Pairs Geometry & Efficient Attention: RAFT-3D and SLAB

##Objectives

In this lab, you will:

1. Understand **scene flow** (pixelwise 3D motion) and how it differs from optical flow, using pinhole geometry and **SE3 rigid motion** in code.
2. Build a **synthetic RGB-D benchmark** with exact ground-truth 3D motion, and score methods with **3D end-point error (EPE)** and threshold accuracy.
3. Implement the **RAFT skeleton**: an **all-pairs correlation volume**, a **correlation lookup**, and a **recurrent (ConvGRU) update loop**, as a small scene-flow model (`MiniRAFT3D`).
4. See how far a per-pixel translation field gets you, and what the real **RAFT-3D** adds (rigid-motion embeddings and Dense-SE3).
5. Implement the two ideas of **SLAB**: **Simplified Linear Attention (SLA)** and **Progressive Re-parameterized BatchNorm (PRepBN)**, and **prove numerically** that the re-parameterisation and fusion tricks are exact.
6. Train small vision transformers with and without these ideas, and measure **accuracy, latency and scaling** honestly.
7. Connect the two parts: both pay a **quadratic cost for all-pairs comparisons**, and both papers fight it; compare *how* they do it and why the same trick does not transfer.
8. Load saved checkpoints, run **inference**, and complete a **multi-part exercise**.

---

## Background: RAFT-3D and SLAB in a Nutshell

### RAFT-3D (Recurrent All-Pairs Field Transforms for 3D)

**Optical flow** estimates the 2-D motion of every pixel between two frames. **Scene flow** estimates the **3-D motion** of every pixel, given a pair of stereo or RGB-D frames. It matters for path planning, collision avoidance and AR/VR, because the world moves in 3-D, not just in the image plane.

RAFT-3D builds on **RAFT**, an optical-flow architecture with a simple recipe:

1. Extract **features** from both frames with a CNN.
2. Build a **4D all-pairs correlation volume**: the dot-product similarity between **every pixel of frame 1 and every pixel of frame 2**.
3. Keep a current **estimate** of the motion, and **iterate** with a recurrent unit (a ConvGRU): look up correlations around where the current estimate points, and use them, plus a hidden state, to predict an update.

RAFT-3D keeps this skeleton but changes what is being estimated: instead of a 2-D flow vector per pixel it maintains a **dense field of per-pixel SE3 motion** (a rotation and translation, `p' = R p + t`). Its key ideas are:

- **Rigid-motion embeddings:** a per-pixel vector that **softly groups pixels into rigid objects**. Pixels with similar embeddings should follow the same SE3 motion.
- **Dense-SE3:** a differentiable layer that runs unrolled **Gauss-Newton** iterations to make the per-pixel SE3 field consistent with the current embeddings and pixel correspondences. Because of it, the embeddings can be trained from **ground-truth 3D scene flow alone**, with no object box or mask supervision.

### SLAB (Simplified Linear Attention + Progressive Re-parameterized BatchNorm)

SLAB (Guo, Chen, Tang, Wang, ICML 2024) attacks the two slowest parts of a transformer block at inference time:

**1. LayerNorm** needs its mean and variance **computed on the fly at inference**. **BatchNorm** uses fixed statistics, so it can be **merged into the neighbouring linear layer** and disappears at inference, but training a transformer with plain BatchNorm is often unstable. SLAB's remedy has two ingredients:

```
RepBN(X)   = BN(X) + η·X                      (η is a learnable scalar)
PRepBN(X)  = γ·LN(X) + (1 − γ)·RepBN(X)       γ decays linearly from 1 to 0 during training
```

Training starts as a LayerNorm network (stable) and ends as a pure BatchNorm network (fast). At the end `γ = 0`, and `RepBN` collapses exactly into an ordinary BatchNorm with adjusted scale and shift (`α + ησ`, `β + ημ`), which then merges into the following linear layer.

**2. Softmax attention** costs `O(N²C)` for `N` tokens. **Linear attention** replaces the softmax similarity by a kernel and regroups the computation to compute `KᵀV` first, costing `O(NC²)`. SLAB's **Simplified Linear Attention** uses the simplest possible kernel and adds a local term:

```
Sim(Qᵢ, Kⱼ) = ReLU(Qᵢ) · ReLU(Kⱼ)ᵀ
O  =  Σⱼ [Sim(Qᵢ, Kⱼ) / Σⱼ Sim(Qᵢ, Kⱼ)] · Vⱼ   +   DWC(V)       (DWC = depth-wise convolution)
```

The depth-wise convolution restores **local detail** that a low-rank linear attention tends to lose.

### Why these two papers are in the same lab

Both are about **all-pairs interactions**. RAFT-3D compares every pixel in frame 1 with every pixel in frame 2; attention compares every token with every other token. Both cost **quadratic in the number of positions**. SLAB makes that cheaper by *re-ordering the computation* (possible because attention only needs a weighted *sum*). RAFT cannot use the same trick, because it needs the *individual similarity values* for its lookups. Part C of the lab makes this precise by measurement.

---

### The key idea of this lab

| Part | Question |
|---|---|
| **A — Scene flow** | How far does the RAFT recipe (all-pairs correlation + recurrent refinement) get on a controlled RGB-D benchmark, versus a classical "optical flow + depth" baseline? What is still missing relative to RAFT-3D? |
| **B — SLAB** | Are simplified linear attention and progressive BatchNorm *exact* where claimed, and what do they cost or buy in accuracy and speed on a small transformer? |
| **C — All-pairs cost** | Where does the quadratic cost bite, and why can one all-pairs computation be linearised while the other can only be made memory-efficient? |

**Report what you measure, not what you expected.** The SLAB paper's numbers are for ImageNet-scale models; a tiny model on CIFAR-10 may show smaller differences, or different ones.

---

## 0️⃣ Prerequisites

- The **`cv-dl`** environment from Lab 6-1. Everything below uses only `torch`, `torchvision`, `numpy`, `opencv-python`, `matplotlib` and `pandas`, which you already have from Labs 7 and 8.

```bash
conda activate cv-dl
cd <your lab9 folder>
```

- **CPU is enough** for everything marked as required, but a GPU makes Part B much faster. CIFAR-10 (about 170 MB) downloads automatically through `torchvision`.

- **Optional (official RAFT-3D).** The official code requires the **LieTorch** package (which compiles its own CUDA kernels), so it needs a CUDA-capable Linux setup. See the repository `princeton-vl/RAFT-3D` and follow its README. Nothing in the required lab depends on it.

### Check you're ready

```python
import sys, torch, cv2, torchvision
print("python:", sys.executable)                  # must contain 'cv-dl'
print("torch :", torch.__version__, "| cuda:", torch.cuda.is_available())
print("opencv:", cv2.__version__, "| torchvision:", torchvision.__version__)
```

Start TensorBoard before training anything:

```python
%load_ext tensorboard
%tensorboard --logdir runs
```

---

# PART A — Scene Flow with a RAFT-Style Model

## 1️⃣ Geometry and a Synthetic RGB-D Benchmark

### 1.1 The geometry you need

A pinhole camera with focal length `f` (in pixels) and principal point `(cx, cy)` relates a pixel `(u, v)` with depth `Z` to a 3-D point:

```
X = (u − cx)·Z / f        Y = (v − cy)·Z / f        Z = Z
u = f·X / Z + cx          v = f·Y / Z + cy
```

A **rigid motion** (SE3) moves a point by `p' = R·p + t`. The **scene flow** of a pixel is `p' − p`, a 3-vector in scene units (metres here). Projecting `p'` back into the image gives the pixel's **optical flow**. Scene flow therefore contains the optical flow *and* the depth change, which is why RGB-D (or stereo) input is needed.

### 1.2 The benchmark

We cannot use a real driving dataset in a lab, so `flow3d_geometry.py` renders **scenes of textured fronto-parallel "cards"** at different depths in front of a textured background. Each card translates by its own random 3-D vector between frames, so the **ground-truth scene flow is exactly known**, and is **piecewise constant** (the same property that motivates RAFT-3D's rigid-object grouping). We keep rotations out of the renderer for simplicity; this is the main thing the benchmark leaves out relative to real SE3 motion, and a stretch task asks you to add it.

```python
"""flow3d_geometry.py - pinhole geometry, SE3 motion, synthetic card scenes, metrics."""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================ geometry
def pixel_grid(H, W, device=None, dtype=torch.float32):
    v, u = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                          torch.arange(W, device=device, dtype=dtype), indexing="ij")
    return u, v                                              # each (H, W)


def backproject(depth, f, cx=None, cy=None):
    """depth (B,1,H,W) -> 3-D points (B,3,H,W)."""
    _, _, H, W = depth.shape
    cx = W / 2 if cx is None else cx
    cy = H / 2 if cy is None else cy
    u, v = pixel_grid(H, W, depth.device, depth.dtype)
    return torch.cat([(u - cx) * depth / f, (v - cy) * depth / f, depth], dim=1)


def project(points, f, cx=None, cy=None):
    """3-D points (B,3,H,W) -> pixel coordinates (B,2,H,W) as (u, v)."""
    _, _, H, W = points.shape
    cx = W / 2 if cx is None else cx
    cy = H / 2 if cy is None else cy
    X, Y, Z = points[:, 0:1], points[:, 1:2], points[:, 2:3].clamp(min=1e-6)
    return torch.cat([f * X / Z + cx, f * Y / Z + cy], dim=1)


def rodrigues(w):
    """Axis-angle vectors (..., 3) -> rotation matrices (..., 3, 3)."""
    theta = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = w / theta
    K = torch.zeros(*w.shape[:-1], 3, 3, dtype=w.dtype, device=w.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    th = theta[..., None]
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def apply_se3_field(points, R, t):
    """Per-pixel rigid motion p' = R p + t.  points (B,3,H,W), R (B,H,W,3,3), t (B,3,H,W)."""
    return torch.einsum("bhwij,bjhw->bihw", R, points) + t


def flow3d_from_correspondence(depth1, du, dv, dz, f, cx=None, cy=None):
    """Pixel offsets (du, dv) and depth change dz -> 3-D scene flow (B,3,H,W)."""
    _, _, H, W = depth1.shape
    cx = W / 2 if cx is None else cx
    cy = H / 2 if cy is None else cy
    u, v = pixel_grid(H, W, depth1.device, depth1.dtype)
    z1, z2 = depth1, depth1 + dz
    X1, Y1 = (u - cx) * z1 / f, (v - cy) * z1 / f
    X2, Y2 = (u + du - cx) * z2 / f, (v + dv - cy) * z2 / f
    return torch.cat([X2 - X1, Y2 - Y1, z2 - z1], dim=1)


def correspondence_from_flow3d(depth1, flow3d, f, cx=None, cy=None):
    """Inverse of the above: returns du, dv, dz and a mask of targets that stay in view."""
    _, _, H, W = depth1.shape
    p2 = backproject(depth1, f, cx, cy) + flow3d
    uv2 = project(p2, f, cx, cy)
    u, v = pixel_grid(H, W, depth1.device, depth1.dtype)
    du, dv, dz = uv2[:, 0:1] - u, uv2[:, 1:2] - v, flow3d[:, 2:3]
    inside = ((uv2[:, 0:1] >= 0) & (uv2[:, 0:1] <= W - 1) &
              (uv2[:, 1:2] >= 0) & (uv2[:, 1:2] <= H - 1)).float()
    return du, dv, dz, inside


# ============================================================ synthetic scenes
def _texture(rng, h=48, w=48):
    low = rng.uniform(0, 1, (6, 6, 3)).astype(np.float32)
    tex = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(tex + rng.uniform(-0.15, 0.15, (h, w, 1)).astype(np.float32), 0, 1)


def render(cards, size=64, f=64.0):
    """Painter's algorithm, far to near. Returns image (H,W,3), depth (H,W), owner (H,W)."""
    H = W = size
    cx = cy = size / 2
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)
    depth = np.zeros((H, W), np.float32)
    owner = np.full((H, W), -1, np.int64)
    for k in sorted(range(len(cards)), key=lambda i: -cards[i]["z"]):
        c = cards[k]
        X, Y = (u - cx) * c["z"] / f, (v - cy) * c["z"] / f
        ins = (X >= c["x0"]) & (X < c["x0"] + c["w"]) & (Y >= c["y0"]) & (Y < c["y0"] + c["h"])
        th, tw = c["tex"].shape[:2]
        tu = np.clip(((X - c["x0"]) / c["w"] * (tw - 1)).round().astype(int), 0, tw - 1)
        tv = np.clip(((Y - c["y0"]) / c["h"] * (th - 1)).round().astype(int), 0, th - 1)
        img[ins] = c["tex"][tv[ins], tu[ins]]
        depth[ins] = c["z"]
        owner[ins] = k
    return img, depth, owner


def make_pair(seed, n_cards=3, size=64, f=64.0, max_t=(0.15, 0.15, 0.30)):
    """Two RGB-D frames plus exact scene flow. Card 0 is the static background."""
    rng = np.random.default_rng(seed)
    bg = dict(x0=-3.0, y0=-3.0, w=6.0, h=6.0, z=5.0, tex=_texture(rng, 96, 96))
    cards, motions = [bg], [np.zeros(3, np.float32)]
    for _ in range(n_cards):
        z = rng.uniform(1.5, 4.0)
        half = size / 2 * z / f                              # half-width of the view at depth z
        w, h = rng.uniform(0.4, 0.9) * half, rng.uniform(0.4, 0.9) * half
        cards.append(dict(x0=rng.uniform(-half, half - w), y0=rng.uniform(-half, half - h),
                          w=w, h=h, z=z, tex=_texture(rng)))
        motions.append(rng.uniform(-1, 1, 3).astype(np.float32) * np.array(max_t, np.float32))

    img1, d1, owner = render(cards, size, f)
    moved = [dict(c, x0=c["x0"] + m[0], y0=c["y0"] + m[1], z=c["z"] + m[2])
             for c, m in zip(cards, motions)]
    img2, d2, _ = render(moved, size, f)
    flow3d = np.array(motions)[owner].transpose(2, 0, 1)    # (3,H,W): the owner card's motion

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    out = dict(img1=t(img1.transpose(2, 0, 1)), img2=t(img2.transpose(2, 0, 1)),
               depth1=t(d1)[None], depth2=t(d2)[None], flow3d=t(flow3d),
               owner=torch.from_numpy(owner))

    # valid = target stays in view AND is not occluded in frame 2 (depth agrees)
    du, dv, dz, inside = correspondence_from_flow3d(out["depth1"][None], out["flow3d"][None], f)
    u, v = pixel_grid(size, size)
    tu = (u + du[0, 0]).round().clamp(0, size - 1).long()
    tv = (v + dv[0, 0]).round().clamp(0, size - 1).long()
    consistent = ((out["depth2"][0][tv, tu] - (out["depth1"][0] + dz[0, 0])).abs() < 0.1).float()
    out["inside"] = inside[0]
    out["valid"] = (inside[0, 0] * consistent)[None]
    return out


class CardsDataset(Dataset):
    """Deterministic: sample i is always scene (seed0 + i)."""

    def __init__(self, n, seed0=0, **kw):
        self.n, self.seed0, self.kw = n, seed0, kw

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return make_pair(self.seed0 + i, **self.kw)


# ============================================================ metrics and baselines
def scene_flow_metrics(pred, gt, valid, moving=None, thresholds=(0.05, 0.10)):
    """pred, gt (B,3,H,W); valid (B,1,H,W). Errors are in scene units (metres)."""
    err = (pred - gt).norm(dim=1, keepdim=True)
    m = valid > 0.5
    out = {"epe3d": err[m].mean().item()}
    for th in thresholds:
        out[f"acc@{th}"] = (err[m] < th).float().mean().item()
    if moving is not None:
        mv, st = m & (moving > 0.5), m & (moving <= 0.5)
        out["epe3d_moving"] = err[mv].mean().item() if mv.any() else float("nan")
        out["epe3d_static"] = err[st].mean().item() if st.any() else float("nan")
    return out


def farneback_scene_flow(sample, f=64.0):
    """Classical baseline: Farneback optical flow, then lift to 3-D with the frame-2 depth."""
    g = lambda im: cv2.cvtColor((im.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                                cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g(sample["img1"]), g(sample["img2"]), None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)          # (H,W,2): dx, dy
    d1, d2 = sample["depth1"][0].numpy(), sample["depth2"][0].numpy()
    H, W = d1.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    tu = np.clip(np.round(uu + flow[..., 0]), 0, W - 1).astype(int)
    tv = np.clip(np.round(vv + flow[..., 1]), 0, H - 1).astype(int)
    dz = d2[tv, tu] - d1                                                   # depth at the target pixel
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()[None, None]
    out = flow3d_from_correspondence(t(d1), t(flow[..., 0]), t(flow[..., 1]), t(dz), f)
    return out[0]


def evaluate_baseline(fn, ds):
    preds, gts, valids = [], [], []
    for i in range(len(ds)):
        s = ds[i]
        preds.append(fn(s)); gts.append(s["flow3d"]); valids.append(s["valid"])
    gt = torch.stack(gts)
    moving = (gt.norm(dim=1, keepdim=True) > 1e-6).float()
    return scene_flow_metrics(torch.stack(preds), gt, torch.stack(valids), moving)
```

### Task

1. **Round trip.** On a random depth map, check that `project(backproject(depth))` recovers the pixel grid (max error < 1e-4).
2. **SE3 by hand.** Build a per-pixel SE3 field with a constant rotation (use `rodrigues`, e.g. 5° about the vertical axis) and translation `t = (0, 0, 0.2)`. Compute (a) the 3-D scene flow `apply_se3_field(p, R, t) − p` and (b) the 2-D optical flow from projecting `p'`. For pure forward motion (`R = I`, `t = (0,0,tz)`), derive the 2-D flow analytically from the projection equations. Where is it zero, and how does it vary with distance from the image centre and with depth? Check your formula against the code.
3. **The benchmark.** Display a pair (`img1`, `img2`), `depth1`, and the ground-truth flow (e.g. the 2-D flow from `correspondence_from_flow3d` drawn with `quiver`, plus a plot of the `valid` mask). Why are some pixels invalid?
4. **Is the ground truth consistent?** Warp `img2` back to frame 1 with the ground-truth 2-D flow (`torch.nn.functional.grid_sample`) and compute the photometric error on valid pixels versus all pixels. What does the residual on invalid pixels show?
5. **Classical baselines.** Evaluate (a) **static** (predict zero motion) and (b) **Farneback + depth lifting** on 100 scenes with `evaluate_baseline`. Report EPE, `acc@0.05`, `acc@0.10`, and the EPE separately on moving and static pixels. Why does the static baseline do well on the static pixels but badly on the moving ones, and where does Farneback fail?

---

## 2️⃣ `MiniRAFT3D`: the RAFT Recipe in Miniature

### What it is, and what it is not

`MiniRAFT3D` keeps RAFT's three components and the *spirit* of RAFT-3D's problem (estimate 3-D motion from RGB-D frames), but it is deliberately simpler than the real model:

| | `MiniRAFT3D` (this lab) | RAFT-3D (the paper) |
|---|---|---|
| Features | small CNN on RGB + depth | larger feature and context encoders |
| Matching | one-level **all-pairs correlation volume** | RAFT-style all-pairs correlation volume (see the paper for details) |
| Refinement | **ConvGRU** loop, correlation lookup at the current estimate | ConvGRU loop, lookup *indexed by the current SE3 estimate* |
| State per pixel | **2-D flow offset + depth change** (a translation field) | **dense SE3** (rotation + translation) |
| Grouping into rigid objects | none | **rigid-motion embeddings** |
| Geometric consistency | none (the network just regresses the update) | **Dense-SE3**, a differentiable Gauss-Newton layer |
| Output | 3-D flow lifted from (flow, depth change) | SE3 field, projectable to flow |

That gap is the point of Task 7 below: you will see what a translation-only, non-grouped model gets wrong.

```python
"""mini_raft3d.py - RAFT's skeleton, scaled down for a lab."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow3d_geometry import flow3d_from_correspondence


def coords_grid(B, h, w, device):
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    return torch.stack([xs, ys], dim=0)[None].repeat(B, 1, 1, 1)           # (B,2,h,w) as (x, y)


def make_encoder(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
        nn.Conv2d(64, out_ch, 3, padding=1))                               # output is 1/4 resolution


class CorrBlock:
    """ALL-PAIRS correlation: corr[b, i, j] = <f1_i, f2_j> / sqrt(C) for every pixel pair (i, j).

    Memory is (h*w)^2 per image. Lookup samples a (2r+1)^2 window around where each
    pixel of frame 1 currently THINKS it landed in frame 2.
    """

    def __init__(self, f1, f2, radius=4):
        B, C, h, w = f1.shape
        corr = torch.matmul(f1.flatten(2).transpose(1, 2), f2.flatten(2)) / C ** 0.5   # (B, hw, hw)
        self.corr = corr.reshape(B * h * w, 1, h, w)
        self.r = radius

    def __call__(self, coords):                  # coords (B,2,h,w): current target (x, y), low-res px
        B, _, h, w = coords.shape
        r = self.r
        d = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        dy, dx = torch.meshgrid(d, d, indexing="ij")
        delta = torch.stack([dx, dy], dim=-1)                              # (2r+1, 2r+1, 2)
        centre = coords.permute(0, 2, 3, 1).reshape(B * h * w, 1, 1, 2)
        grid = centre + delta[None]
        grid = torch.stack([2 * grid[..., 0] / (w - 1) - 1, 2 * grid[..., 1] / (h - 1) - 1], dim=-1)
        out = F.grid_sample(self.corr, grid, align_corners=True)           # (B*h*w, 1, 2r+1, 2r+1)
        return out.view(B, h, w, -1).permute(0, 3, 1, 2).contiguous()      # (B, (2r+1)^2, h, w)


class OnTheFlyCorr:
    """Same lookup as CorrBlock, but NEVER builds the (hw x hw) volume (used in Part C)."""

    def __init__(self, f1, f2, radius=4):
        self.f1, self.f2, self.r = f1, f2, radius
        self.scale = f1.shape[1] ** -0.5

    def __call__(self, coords):
        B, C, h, w = self.f1.shape
        d = torch.linspace(-self.r, self.r, 2 * self.r + 1, device=coords.device)
        out = []
        for dy in d:                                                       # same order as CorrBlock
            for dx in d:
                x, y = coords[:, 0] + dx, coords[:, 1] + dy                # (B,h,w)
                grid = torch.stack([2 * x / (w - 1) - 1, 2 * y / (h - 1) - 1], dim=-1)
                f2s = F.grid_sample(self.f2, grid, align_corners=True)     # (B,C,h,w)
                out.append((self.f1 * f2s).sum(1, keepdim=True) * self.scale)
        return torch.cat(out, dim=1)


class ConvGRU(nn.Module):
    """Lab 7's GRU, with convolutions instead of matrix products."""

    def __init__(self, hidden, inp):
        super().__init__()
        self.convz = nn.Conv2d(hidden + inp, hidden, 3, padding=1)
        self.convr = nn.Conv2d(hidden + inp, hidden, 3, padding=1)
        self.convq = nn.Conv2d(hidden + inp, hidden, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z, r = torch.sigmoid(self.convz(hx)), torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class MiniRAFT3D(nn.Module):
    """(img1, img2, depth1, depth2) -> list of 3-D scene-flow estimates, one per iteration."""

    def __init__(self, hidden=64, radius=4, f=64.0, use_depth_input=True, corr_impl="volume"):
        super().__init__()
        in_ch = 4 if use_depth_input else 3
        self.use_depth_input, self.hidden, self.radius, self.f = use_depth_input, hidden, radius, f
        self.corr_impl = corr_impl
        self.fnet = make_encoder(in_ch, 64)                  # matching features
        self.cnet = make_encoder(in_ch, 2 * hidden)          # context: initial state + GRU input
        self.motion = nn.Sequential(nn.Conv2d((2 * radius + 1) ** 2 + 3, 96, 3, padding=1),
                                    nn.ReLU(inplace=True), nn.Conv2d(96, 64, 3, padding=1),
                                    nn.ReLU(inplace=True))
        self.gru = ConvGRU(hidden, hidden + 64)
        self.head = nn.Sequential(nn.Conv2d(hidden, 64, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 3, 3, padding=1))   # update to (du, dv, dz)

    def _stack(self, img, depth):
        return torch.cat([img, depth / 5.0], dim=1) if self.use_depth_input else img

    def to_scene_flow(self, state, depth1):
        H, W = depth1.shape[-2:]
        h, w = state.shape[-2:]
        up = F.interpolate(state, size=(H, W), mode="bilinear", align_corners=False)
        du, dv, dz = up[:, 0:1] * (W / w), up[:, 1:2] * (H / h), up[:, 2:3]   # px at full res, metres
        return flow3d_from_correspondence(depth1, du, dv, dz, self.f)

    def forward(self, img1, img2, depth1, depth2, iters=6):
        x1, x2 = self._stack(img1, depth1), self._stack(img2, depth2)
        f1, f2 = self.fnet(x1), self.fnet(x2)
        net, inp = torch.split(self.cnet(x1), [self.hidden, self.hidden], dim=1)
        net, inp = torch.tanh(net), torch.relu(inp)

        corr = (CorrBlock if self.corr_impl == "volume" else OnTheFlyCorr)(f1, f2, self.radius)
        B, _, h, w = f1.shape
        base = coords_grid(B, h, w, f1.device)
        state = torch.zeros(B, 3, h, w, device=f1.device)    # (du, dv) in low-res px, dz in metres

        preds = []
        for _ in range(iters):
            state = state.detach()                           # as in RAFT: no gradient through the lookup
            c = corr(base + state[:, :2])
            m = self.motion(torch.cat([c, state], dim=1))
            net = self.gru(net, torch.cat([inp, m], dim=1))
            state = state + self.head(net)
            preds.append(self.to_scene_flow(state, depth1))
        return preds


def load_mini_raft(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    c = ckpt["config"]
    model = MiniRAFT3D(hidden=c.get("hidden", 64), radius=c.get("radius", 4),
                       use_depth_input=c.get("use_depth_input", True)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()
```

### `train_scene_flow.py`

The loss follows RAFT's recipe: an **L1 error on every iteration's output**, with **later iterations weighted more heavily**, restricted to valid pixels.

```python
"""train_scene_flow.py"""

import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from flow3d_geometry import CardsDataset, scene_flow_metrics
from mini_raft3d import MiniRAFT3D


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def sequence_loss(preds, gt, valid, gamma=0.8):
    n, loss = len(preds), 0.0
    for i, p in enumerate(preds):
        w = gamma ** (n - i - 1)                         # last iteration has weight 1
        loss = loss + w * ((p - gt).abs() * valid).sum() / (valid.sum() * 3 + 1e-6)
    return loss


@torch.no_grad()
def evaluate_scene_flow(model, ds, iters_list=(1, 2, 4, 6, 12), device=None, batch_size=50):
    """Metrics after k iterations, for each k in iters_list (the loop is the same trajectory)."""
    device = device or get_device()
    model.eval()
    results = {k: [] for k in iters_list}
    for batch in DataLoader(ds, batch_size=batch_size):
        b = to_device(batch, device)
        preds = model(b["img1"], b["img2"], b["depth1"], b["depth2"], iters=max(iters_list))
        moving = (b["flow3d"].norm(dim=1, keepdim=True) > 1e-6).float()
        for k in iters_list:
            results[k].append(scene_flow_metrics(preds[k - 1], b["flow3d"], b["valid"], moving))
    return {k: {m: sum(r[m] for r in rs) / len(rs) for m in rs[0]} for k, rs in results.items()}


def train_scene_flow(config):
    torch.manual_seed(config.get("seed", 0))
    device = get_device()
    model = MiniRAFT3D(hidden=config.get("hidden", 64), radius=config.get("radius", 4),
                       use_depth_input=config.get("use_depth_input", True)).to(device)
    steps, bs, iters = config.get("steps", 2000), config.get("batch_size", 16), config.get("iters", 6)
    lr = config.get("lr", 5e-4)

    train_ds = CardsDataset(config.get("n_train", 8000), seed0=10_000, n_cards=config.get("n_cards", 3))
    val_ds = CardsDataset(100, seed0=0)                  # fixed validation scenes (never trained on)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)

    run = config["run_name"]
    writer = SummaryWriter(os.path.join("runs", run))
    os.makedirs(os.path.join("checkpoints", run), exist_ok=True)
    ckpt_path = os.path.join("checkpoints", run, "best.pt")

    best, step, t0 = float("inf"), 0, time.perf_counter()
    while step < steps:
        for batch in loader:
            model.train()
            b = to_device(batch, device)
            preds = model(b["img1"], b["img2"], b["depth1"], b["depth2"], iters=iters)
            loss = sequence_loss(preds, b["flow3d"], b["valid"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            writer.add_scalar("loss/train", loss.item(), step)

            if step % config.get("eval_every", 250) == 0 or step == steps:
                r = evaluate_scene_flow(model, val_ds, (iters,), device)[iters]
                for k, v in r.items():
                    writer.add_scalar(f"val/{k}", v, step)
                if r["epe3d"] < best:
                    best = r["epe3d"]
                    torch.save({"model_state_dict": model.state_dict(), "config": config,
                                "step": step, "val_epe3d": best}, ckpt_path)
            if step >= steps:
                break
    writer.close()
    return dict(run_name=run, best_val_epe3d=best, train_time=time.perf_counter() - t0,
                n_params=sum(p.numel() for p in model.parameters()), ckpt=ckpt_path)
```

> 💡 **Notice what this reuses from earlier labs:** a GRU-style gated update (Lab 7), convolutional features (Lab 8), per-iteration supervision, gradient clipping and a best-checkpoint rule (Labs 6-1 and 7). Nothing here is a new training technique.

### 2.1 Worked example

```python
"""run_scene_flow_example.py"""

import torch
from flow3d_geometry import CardsDataset, evaluate_baseline, farneback_scene_flow
from train_scene_flow import train_scene_flow, evaluate_scene_flow
from mini_raft3d import load_mini_raft

val_ds = CardsDataset(100, seed0=0)

print("static   :", evaluate_baseline(lambda s: torch.zeros_like(s["flow3d"]), val_ds))
print("farneback:", evaluate_baseline(farneback_scene_flow, val_ds))

res = train_scene_flow(dict(run_name="scene_flow_base", steps=2000, batch_size=16, iters=6))
print({k: v for k, v in res.items()})

model = load_mini_raft(res["ckpt"])
for k, r in evaluate_scene_flow(model, val_ds, iters_list=(1, 2, 4, 6, 12)).items():
    print(f"{k:>2} iterations:", {m: round(v, 4) for m, v in r.items()})
```

### What to look for

- The **static** baseline gives the floor: its error on moving pixels is the motion itself.
- **Farneback + lifting** is a strong classical baseline for textured regions but is unreliable at motion boundaries and for the depth component (it lifts *using the depth at a rounded pixel*).
- For `MiniRAFT3D`, watch how EPE changes with the **number of iterations**. The model is trained with 6; evaluating at 12 tests whether the recurrent loop has learned a *convergent* refinement or something specific to 6 steps. **Report which you see.**
- On a CPU, 2000 steps takes minutes, not seconds. Use `steps=500` for a smoke test.

---

## 3️⃣ Part A — Experiments

Give every run a new `run_name` and overlay them in TensorBoard. Use **three seeds** for any comparison you plan to rank.

### 3.1 Tasks

1. **Baselines vs model.** One table: static, Farneback+lift, and `MiniRAFT3D` (mean ± std over three seeds) with EPE, `acc@0.05`, `acc@0.10`, and EPE on moving vs static pixels.
2. **Iterations.** Plot EPE against the number of test-time iterations (1 to 12 or more) for your trained model. Does EPE keep improving past the number of training iterations, plateau, or get worse? Explain, in terms of what a recurrent refinement is supposed to learn.
3. **Training iterations.** Train with `iters ∈ {2, 6, 10}` and compare final EPE and training time per step. What does each extra iteration buy, and at what cost?
4. **Lookup radius.** Train with `radius ∈ {1, 2, 4}`. The maximum motion (at 1/4 resolution) is a couple of pixels. Is the large radius necessary? What happens to the correlation volume lookup cost?
5. **Does depth help the features?** Train with `use_depth_input=False` (depth is still used for the geometry, but not fed to the encoders). Report the change.
6. **Where does it fail?** Compute the error as a function of **distance to the nearest motion boundary** (use `owner` to find pixels adjacent to a different owner) and on **occlusion-adjacent** pixels. Show the error map for three scenes. Describe the pattern.
7. **What is missing relative to RAFT-3D?** `MiniRAFT3D` predicts every pixel's motion independently (through shared convolutions). Using your failure analysis, explain how a **rigid-motion grouping** (pixels in the same object share one SE3 motion) would help at boundaries and in textureless interiors, and why enforcing it needs something like Dense-SE3 rather than just a smoother network. (Answer from your own measurements; do not just paraphrase the paper.)
8. **Optional: official RAFT-3D.** If you have a CUDA setup, install the official repository and run its demo script as described in its README. Compare its output qualitatively with your model on a pair of your own (see Section 7). Report anything that surprised you.

---

# PART B — SLAB: Simplified Linear Attention and Progressive Re-parameterized BatchNorm

## 4️⃣ Implementing SLAB

### 4.1 The modules

Everything below is implemented from the paper's equations. The paper trains the full-size models on ImageNet with its own recipe; this is a **small re-implementation for study**, not the authors' code.

> 💡 **Design choices the paper does not fix, so they are yours to ablate:** the initial value of `η` (we default to `1.0`), the exact depth-wise kernel size (we use 3), and where the final normalisation sits. Each is exposed as an argument.

```python
"""slab_modules.py - SLA, RepBN, PRepBN and a tiny ViT built from them."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ normalisations
class TokenBN(nn.Module):
    """BatchNorm over the channel dimension of (B, N, C) tokens."""

    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class RepBN(nn.Module):
    """RepBN(X) = BN(X) + eta * X   (paper Eq. 2), eta learnable."""

    def __init__(self, dim, eta_init=1.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))

    def forward(self, x):
        xt = x.transpose(1, 2)
        return (self.bn(xt) + self.eta * xt).transpose(1, 2)


class PRepBN(nn.Module):
    """PRepBN(X) = gamma * LN(X) + (1 - gamma) * RepBN(X)   (paper Eq. 5).

    gamma is a buffer set from the training loop (see set_gamma). In training both
    branches always run, so BatchNorm keeps tracking its statistics while gamma is large.
    """

    def __init__(self, dim, eta_init=1.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.rep = RepBN(dim, eta_init)
        self.register_buffer("gamma", torch.tensor(1.0))

    def forward(self, x):
        g = float(self.gamma)
        if not self.training and g == 0.0:               # inference after the transition: BN only
            return self.rep(x)
        return g * self.ln(x) + (1.0 - g) * self.rep(x)


def set_gamma(model, step, total_steps):
    """Linear decay gamma = (T - t) / T  (paper Eq. 6), clamped at 0 once t >= T."""
    g = max(0.0, (total_steps - step) / total_steps)
    for m in model.modules():
        if isinstance(m, PRepBN):
            m.gamma.fill_(g)
    return g


def make_norm(kind, dim, eta_init=1.0):
    if kind == "ln":
        return nn.LayerNorm(dim)
    if kind == "bn":
        return TokenBN(dim)
    if kind == "repbn":
        return RepBN(dim, eta_init)
    if kind == "prepbn":
        return PRepBN(dim, eta_init)
    raise ValueError(f"unknown norm {kind!r}")


# ------------------------------------------------------------------ re-parameterisation
@torch.no_grad()
def fuse_repbn(rep):
    """Lemma 4.1: RepBN(X; mu, sigma, alpha, beta) = BN(X; mu, sigma, alpha + eta*sigma, beta + eta*mu)."""
    bn = rep.bn
    sigma = torch.sqrt(bn.running_var + bn.eps)
    fused = nn.BatchNorm1d(bn.num_features, eps=bn.eps).to(bn.weight.device)
    fused.running_mean.copy_(bn.running_mean)
    fused.running_var.copy_(bn.running_var)
    fused.weight.copy_(bn.weight + rep.eta * sigma)
    fused.bias.copy_(bn.bias + rep.eta * bn.running_mean)
    return fused.eval()


@torch.no_grad()
def fuse_bn_into_linear(bn, lin):
    """Linear(BN(x)) as ONE Linear.  BN(x) = s*x + b with s = w/sigma, b = beta - s*mu."""
    sigma = torch.sqrt(bn.running_var + bn.eps)
    s = bn.weight / sigma
    b = bn.bias - s * bn.running_mean
    new = nn.Linear(lin.in_features, lin.out_features, bias=True).to(lin.weight.device)
    new.weight.copy_(lin.weight * s[None, :])
    old_bias = lin.bias if lin.bias is not None else torch.zeros_like(new.bias)
    new.bias.copy_(old_bias + lin.weight @ b)
    return new


def to_bn(norm):
    if isinstance(norm, TokenBN):
        return norm.bn
    if isinstance(norm, RepBN):
        return fuse_repbn(norm)
    if isinstance(norm, PRepBN):
        if float(norm.gamma) != 0.0:
            raise RuntimeError("PRepBN can only be merged once gamma == 0")
        return fuse_repbn(norm.rep)
    raise TypeError(f"cannot convert {type(norm).__name__} to a BatchNorm")


# ------------------------------------------------------------------ attention
class SoftmaxAttention(nn.Module):
    """Standard multi-head attention: O(N^2 C)."""

    def __init__(self, dim, num_heads=4, **_):
        super().__init__()
        self.h, self.scale = num_heads, (dim // num_heads) ** -0.5
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)        # (B,h,N,N)
        return self.proj((a @ v).transpose(1, 2).reshape(B, N, C))


class SimplifiedLinearAttention(nn.Module):
    """SLAB Eq. 9:  O = [ReLU(Q) (ReLU(K)^T V)] / [ReLU(Q) sum_j ReLU(K_j)]  +  DWC(V).

    grid=(H, W) is the token layout, needed by the depth-wise convolution.
    use_dwc=False and quadratic=True exist for ablation / verification.
    """

    def __init__(self, dim, num_heads=4, grid=(8, 8), dwc_kernel=3, use_dwc=True,
                 quadratic=False, eps=1e-6):
        super().__init__()
        self.h, self.grid, self.eps = num_heads, grid, eps
        self.use_dwc, self.quadratic = use_dwc, quadratic
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, dwc_kernel, padding=dwc_kernel // 2, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        h, d = self.h, C // self.h
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        local = 0.0
        if self.use_dwc:
            v_map = v.transpose(1, 2).reshape(B, C, *self.grid)
            local = self.dwc(v_map).flatten(2).transpose(1, 2)             # (B,N,C)

        q = F.relu(q).view(B, N, h, d).transpose(1, 2)                     # (B,h,N,d)
        k = F.relu(k).view(B, N, h, d).transpose(1, 2)
        v = v.view(B, N, h, d).transpose(1, 2)

        if self.quadratic:                                                 # reference: builds N x N
            a = q @ k.transpose(-2, -1)
            out = (a / (a.sum(dim=-1, keepdim=True) + self.eps)) @ v
        else:                                                              # linear: K^T V first
            kv = torch.einsum("bhnd,bhne->bhde", k, v)                     # (B,h,d,d)
            z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + self.eps)
            out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)
        return self.proj(out.transpose(1, 2).reshape(B, N, C) + local)


def make_attn(kind, dim, heads, grid, **kw):
    if kind == "softmax":
        return SoftmaxAttention(dim, heads)
    if kind == "sla":
        return SimplifiedLinearAttention(dim, heads, grid, **kw)
    raise ValueError(f"unknown attention {kind!r}")


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, norm, attn, grid, eta_init=1.0, **attn_kw):
        super().__init__()
        self.norm1, self.norm2 = make_norm(norm, dim, eta_init), make_norm(norm, dim, eta_init)
        self.attn = make_attn(attn, dim, heads, grid, **attn_kw)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(),
                                 nn.Linear(mlp_ratio * dim, dim))

    def forward(self, x):                                  # X = X + Attn(Norm(X)); X = X + MLP(Norm(X))
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

    @torch.no_grad()
    def merge_norms(self):
        if isinstance(self.norm1, nn.LayerNorm):
            return                                         # LayerNorm cannot be merged
        self.attn.qkv = fuse_bn_into_linear(to_bn(self.norm1), self.attn.qkv)
        self.norm1 = nn.Identity()
        self.mlp[0] = fuse_bn_into_linear(to_bn(self.norm2), self.mlp[0])
        self.norm2 = nn.Identity()


class TinyViT(nn.Module):
    def __init__(self, img_size=32, patch=4, dim=96, depth=4, heads=4, mlp_ratio=2,
                 num_classes=10, norm="ln", attn="softmax", eta_init=1.0, **attn_kw):
        super().__init__()
        grid = (img_size // patch, img_size // patch)
        self.patch_embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, grid[0] * grid[1], dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, norm, attn, grid, eta_init, **attn_kw)
                                     for _ in range(depth)])
        self.norm = make_norm(norm, dim, eta_init)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x).mean(dim=1))         # mean-pool tokens (no class token)

    @torch.no_grad()
    def merge_norms(self):
        """Fold every BatchNorm-type norm into its following Linear. Call in eval mode."""
        assert not self.training, "call model.eval() first"
        for blk in self.blocks:
            blk.merge_norms()
        if not isinstance(self.norm, nn.LayerNorm):         # affine commutes with mean-pooling
            self.head = fuse_bn_into_linear(to_bn(self.norm), self.head)
            self.norm = nn.Identity()
```

### 4.2 Verify the paper's claims numerically

Before training anything, check that the algebra is exact. These checks cost seconds and catch most implementation bugs.

```python
import torch
from slab_modules import (SimplifiedLinearAttention, RepBN, PRepBN, fuse_repbn,
                          fuse_bn_into_linear, TinyViT)

torch.manual_seed(0)

# (1) linear (K^T V first) == quadratic (explicit N x N) attention
sla = SimplifiedLinearAttention(dim=64, num_heads=4, grid=(8, 8)).eval()
x = torch.randn(2, 64, 64)
y_lin = sla(x)
sla.quadratic = True
y_quad = sla(x)
print("SLA linear vs quadratic, max |diff|:", (y_lin - y_quad).abs().max().item())

# (2) RepBN -> BN re-parameterisation (Lemma 4.1), after giving BN non-trivial statistics
rep = RepBN(64, eta_init=0.7)
for _ in range(20):
    rep(torch.randn(8, 64, 64) * 2 + 1)                      # train-mode passes update running stats
rep.eval()
x = torch.randn(4, 50, 64) * 2 + 1
print("RepBN vs fused BN, max |diff|:",
      (rep(x) - fuse_repbn(rep)(x.transpose(1, 2)).transpose(1, 2)).abs().max().item())

# (3) a whole model: merge all norms into linear layers, outputs must not change
model = TinyViT(norm="prepbn", attn="sla")
model.train()
for _ in range(10):
    model(torch.randn(16, 3, 32, 32))                        # populate BN statistics
for m in model.modules():
    if isinstance(m, PRepBN):
        m.gamma.fill_(0.0)                                   # the end-of-training state
model.eval()
x = torch.randn(8, 3, 32, 32)
before = model(x)
model.merge_norms()
print("whole-model merge, max |diff|:", (model(x) - before).abs().max().item())
print("LayerNorm modules left:", sum(isinstance(m, torch.nn.LayerNorm) for m in model.modules()))
```

### Task

1. Run the checks. Report the three differences (expect values at floating-point rounding level; say what you got). After the whole-model merge, how many normalisation layers remain at inference, and what does the model contain instead?
2. **Break it on purpose.** Run check (3) *without* setting `γ = 0` (leave it at 1, or 0.5). What fails and why? Why is the progressive schedule necessary for the merge, not just for training stability?
3. **Prove Lemma 4.1 yourself.** Starting from `BN(X) = α(X − μ)/σ + β`, derive `RepBN = BN(X; μ, σ, α + ησ, β + ημ)` on paper, and also derive the formula for `fuse_bn_into_linear` (weights `W·diag(s)` and bias `b + W(β − sμ)`). Submit the derivation.
4. **Rank.** The paper argues that its attention keeps a high rank. For one head with `N = 64` tokens, compare `torch.linalg.matrix_rank` of (a) the softmax attention matrix, (b) the normalised ReLU linear-attention matrix, and (c) that matrix **plus** the matrix of the depth-wise convolution acting on a single channel. (Build the conv matrix by applying a random `nn.Conv2d(1, 1, 3, padding=1, bias=False)` to each of the `N` one-hot images on the 8×8 grid and stacking the results as columns.) Explain why (b) is limited by the head dimension, and why (c) is not.

---

## 5️⃣ Part B — Experiments

### 5.1 Training harness

CIFAR-10 is used because MNIST is too easy to expose differences between architectures. As in Lab 8, **augmentation is applied only to the training view**, and the validation split shares the same underlying tensors.

```python
"""slab_train.py"""

import copy
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from slab_modules import TinyViT, set_gamma, PRepBN

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cifar_loaders(batch_size=128, val_size=5000, seed=42, root="./data", num_workers=2, limit=None):
    norm = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    aug = transforms.Compose([transforms.RandomCrop(32, padding=4),
                              transforms.RandomHorizontalFlip()] + norm)
    train_aug = datasets.CIFAR10(root, train=True, download=True, transform=aug)
    train_plain = copy.copy(train_aug)
    train_plain.transform = transforms.Compose(norm)                 # val: no augmentation
    perm = torch.randperm(len(train_aug), generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    if limit is not None:
        train_idx = train_idx[:limit]
    test = datasets.CIFAR10(root, train=False, download=True, transform=transforms.Compose(norm))
    mk = lambda ds, idx, sh: DataLoader(Subset(ds, idx.tolist()), batch_size=batch_size,
                                        shuffle=sh, num_workers=num_workers)
    return (mk(train_aug, train_idx, True), mk(train_plain, val_idx, False),
            DataLoader(test, batch_size=500, num_workers=num_workers))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = n = 0
    for x, y in loader:
        out = model(x.to(device))
        correct += (out.argmax(1).cpu() == y).sum().item()
        n += len(y)
    return correct / n


@torch.no_grad()
def refresh_bn_stats(model, loader, device, max_batches=100):
    """Paper Section 5.1: freeze the weights and only update BatchNorm statistics.
    (They do this at the end of training to repair the variance shift caused by drop-path.)"""
    model.train()
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        model(x.to(device))


def build_model(config):
    arch = {k: config[k] for k in ("img_size", "patch", "dim", "depth", "heads", "mlp_ratio",
                                   "eta_init") if k in config}
    kw = {k: config[k] for k in ("use_dwc", "dwc_kernel") if k in config}
    return TinyViT(norm=config["norm"], attn=config["attn"], **arch, **kw)


def train_vit(config, loaders):
    torch.manual_seed(config.get("seed", 0))
    device = get_device()
    train_loader, val_loader, test_loader = loaders
    model = build_model(config).to(device)
    lr, epochs = config.get("lr", 1e-3), config.get("epochs", 10)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.get("weight_decay", 0.05))
    total = epochs * len(train_loader)
    warm = len(train_loader)                                          # 1 epoch of linear warm-up
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    # gamma reaches 0 after gamma_frac of training, then the rest trains a pure BN network
    T_gamma = int(config.get("gamma_frac", 0.8) * total)
    crit = nn.CrossEntropyLoss()

    ckpt_dir = os.path.join("checkpoints", config["run_name"])
    os.makedirs(ckpt_dir, exist_ok=True)
    best, step, t0 = -1.0, 0, time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            g = set_gamma(model, step, T_gamma) if config["norm"] == "prepbn" else None
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            step += 1
        val_acc = evaluate(model, val_loader, device)
        eligible = config["norm"] != "prepbn" or g == 0.0           # only a gamma=0 model is mergeable
        print(f"epoch {epoch:>2}  val_acc {val_acc:.4f}" + (f"  gamma {g:.2f}" if g is not None else ""))
        if eligible and val_acc > best:
            best = val_acc
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                        "epoch": epoch, "val_acc": val_acc}, os.path.join(ckpt_dir, "best.pt"))

    if config.get("bn_refresh_batches", 0) > 0:
        refresh_bn_stats(model, train_loader, device, config["bn_refresh_batches"])
    train_time = time.perf_counter() - t0
    return model, dict(run_name=config["run_name"], best_val_acc=best,
                       final_val_acc=evaluate(model, val_loader, device),
                       test_acc=evaluate(model, test_loader, device), train_time=train_time,
                       n_params=sum(p.numel() for p in model.parameters()))
```

> ⚠️ **Why the checkpoint rule is different for PRepBN.** While `γ > 0` the model is still partly a LayerNorm network and **cannot be merged**; a "best epoch" from that phase would not be the model the method is claiming. So for `prepbn` only epochs with `γ = 0` are eligible. If you let early epochs win, you would be reporting a LayerNorm model under a BatchNorm model's name.

### 5.2 Latency and attention-scaling benchmark

```python
"""slab_bench.py"""

import time
import torch
from slab_modules import make_attn


@torch.no_grad()
def latency_ms(model, x, warmup=5, iters=30):
    model.eval()
    for _ in range(warmup):
        model(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    return (time.perf_counter() - t0) / iters * 1000.0


def bench_attention(kind, side, dim=64, heads=4, batch=4, **kw):
    """Time ONE attention layer on N = side*side tokens."""
    attn = make_attn(kind, dim, heads, (side, side), **kw).eval()
    x = torch.randn(batch, side * side, dim)
    return latency_ms(attn, x, warmup=2, iters=10)
```

### 5.3 Worked example

```python
"""run_slab_example.py - LN+softmax vs PRepBN+SLA on CIFAR-10 (tiny ViT)."""

import torch
from slab_train import get_cifar_loaders, train_vit
from slab_bench import latency_ms

loaders = get_cifar_loaders(batch_size=128)           # add limit=10000 for a quick CPU run
base = dict(img_size=32, patch=4, dim=96, depth=4, heads=4, epochs=10, lr=1e-3)

rows = {}
for name, norm, attn in [("ln_softmax", "ln", "softmax"), ("prepbn_sla", "prepbn", "sla")]:
    model, res = train_vit({**base, "run_name": f"cifar_{name}", "norm": norm, "attn": attn}, loaders)
    model.eval().cpu()
    x1, x128 = torch.randn(1, 3, 32, 32), torch.randn(128, 3, 32, 32)
    before = latency_ms(model, x128)
    ref = model(x128)
    model.merge_norms()                                # no-op for LayerNorm models
    after = latency_ms(model, x128)
    res.update(lat_b128_before=before, lat_b128_after=after, lat_b1=latency_ms(model, x1),
               merge_max_diff=(model(x128) - ref).abs().max().item())
    rows[name] = res
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
```

### What to look for

- **Accuracy gap:** the paper reports that SLA costs little accuracy at scale; a small model on CIFAR-10 may behave differently. Report the number you get.
- **Merge difference:** must be at rounding level for the PRepBN model, and the LayerNorm model should show no change from `merge_norms()` (nothing to merge).
- **Latency:** a tiny model with 64 tokens is **not** where linear attention shines (the `N²` term is small), and a CPU is not the paper's V100. Do not assume a speed-up; measure it, then use Section 5.5 to see where it should appear.

### 5.4 Experiments

Train each configuration with **three seeds** and the same budget.

| # | What varies | Values |
|---|---|---|
| 1 | norm × attention | `{ln, bn, repbn, prepbn}` × `{softmax, sla}` (8 configs) |
| 2 | `gamma_frac` (PRepBN, SLA) | `0.4`, `0.8`, `1.0` |
| 3 | `eta_init` (RepBN-based) | `0.0`, `1.0` |
| 4 | SLA's depth-wise conv | `use_dwc` `True`, `False` |
| 5 | BatchNorm statistic refresh | `bn_refresh_batches` `0`, `100` |

**Task**

1. **One table** for experiment 1 with: parameters, best val accuracy, **test accuracy (mean ± std over seeds)**, training time, **latency (batch 1 and batch 128, after `merge_norms()`)**, and the merge difference. Mark any run that collapsed or failed to train.
2. **Separate the effects.** The 2×4 grid lets you separate the contribution of the *attention* choice from the *normalisation* choice on both accuracy and latency. For each of the two, state its effect and whether it exceeds the seed-to-seed spread. Which change buys the latency, and which costs accuracy?
3. **Does plain BN fail?** The paper reports that directly training a BatchNorm transformer can be unstable or collapse at larger scale. Does `bn` underperform `ln` here? Does the progressive schedule (`prepbn`) close the gap? Report what you see at this scale, and say whether it supports the claim.
4. **Schedule.** For experiment 2, plot validation accuracy against epoch together with `γ`. What happens to accuracy around the epochs where `γ` approaches 0? Why can `gamma_frac=1.0` be risky (think about how many epochs are left to train the pure-BN network)?
5. **Ablations.** Report experiments 3 to 5, and say which of the paper's design choices (the `η` shortcut, the DWC term, a statistic refresh) mattered in your setting. Be explicit about which of your differences are within noise.
6. Overlay all runs in TensorBoard (log your own scalars, or export the per-epoch accuracies) and include a screenshot.

### 5.5 Scaling: where does linear attention pay off?

```python
import numpy as np
from slab_bench import bench_attention

sides = [4, 8, 16, 32, 64]                       # N = 16, 64, 256, 1024, 4096 tokens
res = {k: [bench_attention(k, s) for s in sides] for k in ("softmax", "sla")}
for k, v in res.items():
    slope = np.polyfit(np.log([s * s for s in sides]), np.log(v), 1)[0]
    print(k, [round(t, 2) for t in v], "log-log slope:", round(slope, 2))
```

**Task**

1. Plot latency against `N` on log-log axes for both attentions. Fit the slopes. Are they consistent with `O(N²C)` for softmax and `O(NC²)` for SLA (+ the `O(N)` depth-wise conv)? Where do the curves cross, if they do?
2. Compute the attention matrix memory for softmax at `N = 4096` (floats per head per image) and compare with SLA's `KᵀV` state (`d × d` per head). Why is memory often the first thing to break?
3. **Does the crossover matter for your classifier?** Your TinyViT has `N = 64`. Train the LN+softmax and PRepBN+SLA models with `patch=2` (`N = 256`; use `limit` or fewer epochs on a CPU) and compare latency again. At what token count would you start to prefer SLA?
4. Why is the paper's claim about *latency* more subtle than the FLOP count suggests? (Hint: the depth-wise convolution and the ReLU are cheap in FLOPs, but memory-bound on some hardware. Check your own measurements for hints.)

---

# PART C — The Cost of All-Pairs, Two Ways

## 6️⃣ Why the Same Trick Doesn't Transfer

Both papers contain an `N × N` object. Linear attention avoids ever building it by **re-associating** `(QKᵀ)V` to `Q(KᵀV)`, which works because attention only needs a *weighted sum over j*. RAFT's correlation volume `C[i, j] = ⟨f1_i, f2_j⟩` is used differently: the lookup needs **individual values** `C[i, j]` for specific `j` near the current estimate, so there is no sum to re-associate. What *does* work is **never storing the volume**: compute only the entries the lookup asks for. `OnTheFlyCorr` (in `mini_raft3d.py`) does exactly that, and it is exactly equal to the stored-volume lookup, because bilinear interpolation is linear: `⟨f1, Σₖ wₖ f2ₖ⟩ = Σₖ wₖ ⟨f1, f2ₖ⟩`.

```python
import time, torch
from mini_raft3d import CorrBlock, OnTheFlyCorr, coords_grid

torch.manual_seed(0)
for h in (16, 32, 64):                                # 64 -> a (4096 x 4096) volume per image
    f1, f2 = torch.randn(1, 64, h, h), torch.randn(1, 64, h, h)
    coords = coords_grid(1, h, h, f1.device) + torch.randn(1, 2, h, h)
    t0 = time.perf_counter(); a = CorrBlock(f1, f2)(coords); t_vol = time.perf_counter() - t0
    t0 = time.perf_counter(); b = OnTheFlyCorr(f1, f2)(coords); t_fly = time.perf_counter() - t0
    print(f"h={h:>3}  max|diff|={(a - b).abs().max().item():.2e}  "
          f"volume entries={(h * h) ** 2:>10,}  time volume {t_vol:.3f}s  on-the-fly {t_fly:.3f}s")
```

### Task

1. **Verify the equivalence.** Report the maximum difference at each `h`. Explain why it is zero up to rounding, including the behaviour near the image border (zero padding).
2. **Memory arithmetic.** For a `640×480` image at feature stride 8 and at stride 4, compute the number of entries in the all-pairs volume and its size in float32. Do the same for a `224×224` ViT with 16×16 patches (`N = 196`) and for `N = 4096` tokens. Which quantity grows as `N²`, and which does the *lookup-only* variant keep growing only as `N`?
3. **Time vs memory.** On your machine, which implementation is faster at each `h`, and which uses less memory (use `tracemalloc` or, on GPU, `torch.cuda.max_memory_allocated`)? Why is "memory-efficient" not the same as "faster"?
4. **Compare the two tricks.** In four sentences, contrast SLAB's re-association (associativity of matrix products) with on-the-fly correlation (linearity of interpolation plus locality of the lookup): what property of the computation does each exploit, and what does each give up? Why could you not apply SLA's trick to RAFT's lookup?
5. **A bridge experiment (optional but recommended).** Add **one global attention block** to `MiniRAFT3D`'s feature maps and compare softmax and SLA versions for accuracy and time:

   ```python
   class FeatureAttention(torch.nn.Module):
       """Global self-attention over the 1/4-resolution feature map (tokens = pixels)."""
       def __init__(self, dim=64, grid=(16, 16), attn="sla"):
           super().__init__()
           from slab_modules import Block
           self.block = Block(dim, heads=4, mlp_ratio=2, norm="ln", attn=attn, grid=grid)
       def forward(self, f):                             # f: (B, C, h, w)
           B, C, h, w = f.shape
           return self.block(f.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, h, w)
   ```

   Wrap `fnet` as `nn.Sequential(make_encoder(...), FeatureAttention(...))` (the grid must equal the feature-map size, `(16, 16)` for 64×64 inputs). Does global context help the scene-flow error? Does SLA match softmax attention here, and is there a speed difference at `N = 256`?

---

## 7️⃣ Inference: Loading Saved Checkpoints

### 7.1 Scene flow on a pair you provide

```python
"""predict_flow.py - estimate scene flow for one RGB-D pair using best.pt"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from flow3d_geometry import correspondence_from_flow3d
from mini_raft3d import load_mini_raft

CHECKPOINT_PATH = "checkpoints/scene_flow_base/best.pt"      # hardcoded - change as needed
PAIR_PATH = "my_pair.npz"      # arrays: img1, img2 (H,W,3 in 0..1), depth1, depth2 (H,W metres), f (pixels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_mini_raft(CHECKPOINT_PATH, device)               # includes model.eval()

d = np.load(PAIR_PATH)
H0, W0 = d["img1"].shape[:2]
S = 64                                                         # the size this model was trained at
f = float(d["f"]) * S / W0                                     # focal length scales with the resize
rs = lambda a, mode: cv2.resize(a.astype(np.float32), (S, S), interpolation=mode)
img1, img2 = [torch.from_numpy(rs(d[k], cv2.INTER_AREA).transpose(2, 0, 1))[None].to(device)
              for k in ("img1", "img2")]
dep1, dep2 = [torch.from_numpy(rs(d[k], cv2.INTER_NEAREST))[None, None].to(device)
              for k in ("depth1", "depth2")]
model.f = f

with torch.no_grad():
    flow3d = model(img1, img2, dep1, dep2, iters=12)[-1]       # (1,3,S,S) metres

du, dv, dz, _ = correspondence_from_flow3d(dep1, flow3d, f)
mag = flow3d.norm(dim=1)[0].cpu().numpy()
print(f"mean 3-D motion: {mag.mean():.3f} m, max: {mag.max():.3f} m")
print("mean (dx, dy, dz) over the 20% fastest pixels:",
      flow3d[0][:, mag > np.quantile(mag, 0.8)].mean(dim=1).cpu().numpy().round(3))

ys, xs = np.mgrid[0:S:3, 0:S:3]
plt.imshow(img1[0].permute(1, 2, 0).cpu().numpy())
plt.quiver(xs, ys, du[0, 0].cpu().numpy()[::3, ::3], dv[0, 0].cpu().numpy()[::3, ::3],
           color="yellow", angles="xy", scale_units="xy", scale=1)
plt.title("predicted 2-D flow (from 3-D scene flow)"); plt.axis("off"); plt.show()
```

To create a test pair, save one from the benchmark:

```python
import numpy as np
from flow3d_geometry import make_pair
s = make_pair(seed=424242)
np.savez("my_pair.npz", img1=s["img1"].permute(1, 2, 0).numpy(), img2=s["img2"].permute(1, 2, 0).numpy(),
         depth1=s["depth1"][0].numpy(), depth2=s["depth2"][0].numpy(), f=64.0)
```

>  **A real RGB-D pair is out of distribution.** Your model has only ever seen flat textured cards. On a real pair (e.g., from a depth camera) expect poor results; that is a limitation of the *training data*, not of the recipe. Say what you observe.

### 7.2 A SLAB classifier on one image

```python
"""predict_cifar.py - classify one image with a saved TinyViT, after merging norms"""

import cv2
import torch

from slab_train import build_model, MEAN, STD

CHECKPOINT_PATH = "checkpoints/cifar_prepbn_sla/best.pt"      # hardcoded - change as needed
IMAGE_PATH = "my_image.jpg"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
model = build_model(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()                                   # required BEFORE merge_norms
model.merge_norms()                            # fuses BatchNorm into Linear (no-op for LayerNorm models)
print(f"Loaded epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f}), norm={ckpt['config']['norm']}")

bgr = cv2.imread(IMAGE_PATH)
if bgr is None:
    raise FileNotFoundError(IMAGE_PATH)
img = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (32, 32), interpolation=cv2.INTER_AREA)
x = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
x = ((x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None])[None]

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]          # softmax at display time, as in Lab 6-1
for p, i in zip(*torch.topk(probs, 3)):
    print(f"  {CLASSES[int(i)]:12s} {float(p):.3f}")
```

---

## Student Exercise  — The Price of All-Pairs

A study that treats RAFT-3D and SLAB as two answers to one problem: *how much accuracy do you trade for how much cost?*

### Part 1 — SLAB study (CIFAR-10)

Run the 2 × 4 grid from Section 5.4 (`{softmax, sla}` × `{ln, bn, repbn, prepbn}`) with **three seeds** each, identical budget and hyperparameters, plus the ablations for `gamma_frac`, `use_dwc` and `eta_init`.

1. **Table:** parameters, test accuracy (mean ± std), training time, merged-model latency at batch 1 and 128, and the merge difference.
2. **Show the Lemma and the merge numerically** for your best PRepBN model, and show your handwritten derivation.
3. **Scaling:** the Section 5.5 sweep, with fitted slopes and the measured crossover. Then repeat the classifier comparison at `patch=2`.
4. **Pareto plot:** accuracy against latency for all configurations, marking which are on the frontier.

### Part 2 — Scene-flow study (synthetic RGB-D)

1. Static, Farneback+lift and `MiniRAFT3D` (three seeds), as in Section 3.
2. Ablate **iterations (train and test)**, **lookup radius**, **depth input** and **number of moving objects** (`n_cards ∈ {1, 3, 6}`), reporting EPE and `acc@0.05`.
3. A **failure analysis**: error against distance to the nearest motion boundary, and on occluded pixels.
4. **Cost:** time per forward pass and the correlation-volume size for your model, and the same with `OnTheFlyCorr` (set `model.corr_impl = "otf"` on a trained `MiniRAFT3D`; any value other than `"volume"` selects it).

### Part 3 — Synthesis

1. **The bridge experiment** (Section 6 task 5): global feature attention with softmax vs SLA inside `MiniRAFT3D`. Report EPE, parameters and time.
2. **Cost model.** Write one formula each for the floating-point cost and the memory of (a) softmax attention over `N` tokens, (b) SLA over `N` tokens, (c) the stored all-pairs volume over `N` pixels, (d) the on-the-fly lookup with window `(2r+1)²`. Validate at least two of them against your measurements.
3. **Hardware reality.** Re-run your latency measurement with a different thread count (`torch.set_num_threads`) or on a GPU if you have one. Did the *ranking* of configurations change?
