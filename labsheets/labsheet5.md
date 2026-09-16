# 🖥️ Computer Vision Lab 5 — Classical ML for Segmentation & Retrieval

## 🎯 Objectives

In this lab, you will:

1. Review the landscape of **classical CV features and classifiers**, and when each combination tends to be used.
2. Download **PASCAL VOC 2012** and its segmentation annotations.
3. Build a **superpixel-based semantic segmentation** pipeline: SLIC superpixels → per-superpixel color histogram features → a Random Forest classifier → VOC-colored output, evaluated with **mean IoU**.
4. Improve that pipeline's features (texture, HSV, or your own combination).
5. Build a **content-based image retrieval (CBIR)** system on Caltech-256, evaluated for both **accuracy (top-5/top-10)** and **speed**.

---

## 0️⃣ Prerequisites

- `cv-env` (or `cvlab`) — plain OpenCV windows, no PySide6 needed this lab.
- Install what this lab adds on top of previous labs:

```bash
conda activate cv-env
pip install scikit-learn scikit-image pillow
```

- `scikit-image` — SLIC superpixels (`skimage.segmentation.slic`) and, later, LBP/texture descriptors.
- `Pillow` (PIL) — needed to correctly read PASCAL VOC's *palette-indexed* ground-truth PNGs (see Section 2 — `cv2.imread` cannot read these the way we need).

---

## 1️⃣ Theory: Matching Features to Classifiers

Over the last four labs you've used a wide range of classical features and classifiers somewhat in isolation. Before building this lab's pipelines, it's worth stepping back and looking at the landscape as a whole — which feature types pair naturally with which classifiers, and why.

### Feature types

| Feature type | What it captures | Where you've seen it | Typical pairing |
|---|---|---|---|
| **Color histograms** (RGB/HSV/Lab) | Overall color distribution, ignores spatial layout | Lab 4's HSV K-Means segmentation | KNN, Random Forest, simple thresholding |
| **Texture** (LBP, GLCM, Gabor filters) | Local micro-patterns — roughness, regularity, orientation of fine detail | Suggested for Lab 4's Caltech-256 task | SVM, Random Forest |
| **Shape / gradient** (HOG, Hu moments) | Edge orientation structure or global contour shape | Lab 1–2 (contours, `approxPolyDP`) | Linear SVM (HOG+SVM is the classic pairing) |
| **Local keypoints + Bag-of-Visual-Words** (SIFT/ORB/AKAZE + K-Means codebook) | Distinctive local structure, aggregated into a fixed-length histogram; robust to viewpoint/scale change | Lab 3 (matching), Lab 4 (BoVW classification) | Linear SVM, KNN |
| **Region/superpixel features** (per-region color/texture, aggregated) | Localized, spatially-coherent statistics per image *region* rather than per pixel or per whole image | This lab (Section 3) | Random Forest |

### Classifiers

| Classifier | Strengths | Typical use case |
|---|---|---|
| **KNN** | No training phase (lazy learner); the prediction step *is* a nearest-neighbor search | Retrieval-style tasks where "most similar" IS the answer you want (Section 5, CBIR) |
| **SVM** (linear/kernel) | Strong margin-based separation, works well on medium-dimensional engineered features; sensitive to feature scaling | HOG+SVM detection, BoVW classification (Lab 3, Lab 4) |
| **Random Forest** | Ensemble of decision trees; handles **heterogeneous, concatenated feature vectors** well, and doesn't require feature scaling (each split only compares one dimension against a threshold) | Tabular-style or concatenated multi-type features — like this lab's per-superpixel histograms |
| **K-Means** (unsupervised) | Groups data without labels; used both as a *feature-construction* step (BoVW vocabulary, Lab 4) and directly for segmentation (Lab 4's HSV clustering) | Vocabulary building, exploratory/unsupervised segmentation |

**Why Random Forest for this lab's segmentation task specifically:** our superpixel features will be several *concatenated* histograms (and, in the student task, possibly different feature *types* concatenated together). Random Forest's per-dimension-threshold splitting handles this kind of feature vector gracefully without needing careful joint normalization the way a distance- or margin-based method (KNN, SVM) would — an important practical reason it (and its relatives, like boosted trees) was a standard choice for region-based classical segmentation before deep learning took over the task.

---

## 2️⃣ Dataset: PASCAL VOC 2012 (Segmentation)

Download the official VOC 2012 train/val bundle (images + all annotations, including segmentation masks):

- Official: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
- The official host is a long-running academic server that occasionally goes down or blocks traffic. If it's unreachable, use the Kaggle mirror instead: https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset

> ⚠️ **Important:** VOC's real **test set has no publicly released ground truth** — it was only ever scored by uploading predictions to the official competition server, which is not something we can use for a lab exercise. The convention (used throughout the segmentation literature) is to treat the provided **`val`** split as your local "test set" for measuring accuracy, and that's what we'll do in this lab too.

Extract the tar file. You should end up with this structure (paths below are relative to the extracted `VOCdevkit/VOC2012/` folder):

```
VOCdevkit/VOC2012/
├── JPEGImages/                         <- all images, e.g. 2007_000032.jpg
├── SegmentationClass/                  <- ground-truth masks, e.g. 2007_000032.png
└── ImageSets/Segmentation/
    ├── train.txt                       <- image IDs (no extension) with segmentation GT, ~1464 images
    └── val.txt                         <- ~1449 images - we'll use this as our local test set
```

### The 21 classes and the official color code

VOC segmentation has 21 classes (20 objects + background), plus a special "void/ignore" pixel value (255) marking annotation boundaries that should be excluded from both training and evaluation.

```python
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
```

Each class has a fixed official RGB color, generated by VOC's own bit-interleaving scheme (rather than hand-typing 21 RGB triples and risking a typo, we generate them the same way the dataset's own devkit does, guaranteeing they match):

```python
import numpy as np

def voc_colormap(n=256):
    def bitget(val, idx):
        return (val >> idx) & 1

    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= bitget(c, 0) << (7 - j)
            g |= bitget(c, 1) << (7 - j)
            b |= bitget(c, 2) << (7 - j)
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap

VOC_COLORMAP = voc_colormap()
VOC_COLORMAP[255] = [224, 224, 192]  # VOC's official "void/ignore" border color
```

### A crucial gotcha: reading the ground-truth masks

VOC's `SegmentationClass/*.png` files are **palette-indexed** images — each pixel's stored *value* (0–20, or 255 for void) directly **is** the class index, and the palette is only used by an image viewer to *display* it in color. If you open these with `cv2.imread()`, OpenCV expands the palette and hands you back BGR colors, not the class indices you actually want. Use **PIL** instead, which preserves the raw indices:

```python
from PIL import Image
import numpy as np

label_mask = np.array(Image.open("VOCdevkit/VOC2012/SegmentationClass/2007_000032.png"))
# label_mask is now a single-channel array of class indices: 0-20, or 255 for void
```

---

## 3️⃣ Sample Code: SLIC Superpixels + Histogram Features + Random Forest

### Pipeline

1. **Train:** for every training image, compute SLIC superpixels, extract a channel-wise binned color histogram per superpixel, and label each superpixel by the **majority ground-truth class** among its (non-void) pixels. Collect all (feature, label) pairs across the whole training set and fit a Random Forest.
2. **Inference/evaluation:** for every val ("test") image, compute SLIC superpixels with the **same parameters**, extract the same feature per superpixel, predict its class with the trained Random Forest, and paint the result using the official VOC color code.
3. **Metric:** accumulate a confusion matrix over every (non-void) pixel across the whole val set, then compute **per-class IoU** and **mean IoU (mIoU)** from it.

> ⚠️ **Runtime note:** processing the full ~1464 train + ~1449 val images will take a while (superpixel computation + feature extraction per region, across ~1500 images each). `MAX_TRAIN_IMAGES`/`MAX_VAL_IMAGES` below let you subsample while developing — set them back to `None` for your final, reported run.

```python
"""
Lab 5 - Semantic Segmentation on PASCAL VOC 2012 using SLIC superpixels +
channel-wise color histograms + a Random Forest classifier.

Pipeline:
  1. Load VOC train/val image-ID lists and the official VOC colormap
  2. TRAIN: SLIC superpixels -> per-superpixel histogram feature ->
     majority-vote ground-truth label (from SegmentationClass)
  3. Train a Random Forest on all (feature, label) pairs collected above
  4. VAL ("test"): SLIC with the SAME parameters -> predict each
     superpixel's label with the trained Random Forest -> paint the
     result with the official VOC color code
  5. Accumulate a confusion matrix across the whole val set and report
     per-class IoU and mean IoU (mIoU)
  6. Show one qualitative example: original / ground truth / prediction
"""

import os
import numpy as np
import cv2
from PIL import Image
from skimage.segmentation import slic
from sklearn.ensemble import RandomForestClassifier

# ---------------- Configuration ----------------
VOC_ROOT = "VOCdevkit/VOC2012"
N_SEGMENTS = 200          # target number of SLIC superpixels per image
COMPACTNESS = 10          # SLIC compactness: higher = more square/regular superpixels
HIST_BINS = 8             # bins per color channel
VOID_LABEL = 255          # VOC's "ignore / boundary" pixel value
MAX_TRAIN_IMAGES = None   # set an int to subsample for faster iteration; None = use all
MAX_VAL_IMAGES = None     # same, for the evaluation loop
RANDOM_STATE = 42

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
NUM_CLASSES = len(CLASSES)  # 21


def voc_colormap(n=256):
    """The official PASCAL VOC color palette, generated the same way the VOC
    devkit does it, so our colors match the dataset's own ground-truth PNGs."""
    def bitget(val, idx):
        return (val >> idx) & 1

    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= bitget(c, 0) << (7 - j)
            g |= bitget(c, 1) << (7 - j)
            b |= bitget(c, 2) << (7 - j)
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap


VOC_COLORMAP = voc_colormap()
VOC_COLORMAP[255] = [224, 224, 192]  # official "void/ignore" border color


def read_image_ids(split):
    """split: 'train' or 'val'."""
    list_path = os.path.join(VOC_ROOT, "ImageSets", "Segmentation", f"{split}.txt")
    with open(list_path) as f:
        return [line.strip() for line in f if line.strip()]


def load_image_rgb(image_id):
    path = os.path.join(VOC_ROOT, "JPEGImages", f"{image_id}.jpg")
    bgr = cv2.imread(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_label_mask(image_id):
    """VOC's SegmentationClass PNGs are PALETTE ('P' mode) images - each pixel
    VALUE (not color!) is the class index. We use PIL, not cv2, to read the
    raw indices directly instead of the colors the palette maps them to."""
    path = os.path.join(VOC_ROOT, "SegmentationClass", f"{image_id}.png")
    return np.array(Image.open(path))  # values: 0-20 = class id, 255 = void/ignore


def channelwise_histogram_feature(image, mask, bins=HIST_BINS):
    """Concatenate a separate binned histogram per color channel, computed
    only over the pixels where mask is True."""
    pixels = image[mask]  # shape (n_pixels, 3)
    features = []
    for c in range(3):
        hist, _ = np.histogram(pixels[:, c], bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total  # normalize so superpixel SIZE doesn't dominate the feature
        features.append(hist)
    return np.concatenate(features)  # length = 3 * bins


def majority_label(mask, label_mask, void_fraction_threshold=0.5):
    """Ground-truth label for a superpixel = the most common non-void pixel
    label inside it. Returns None if there aren't enough non-void pixels
    to trust a majority vote."""
    region_labels = label_mask[mask]
    void_fraction = np.mean(region_labels == VOID_LABEL)
    if void_fraction > void_fraction_threshold:
        return None

    valid_labels = region_labels[region_labels != VOID_LABEL]
    if len(valid_labels) == 0:
        return None

    counts = np.bincount(valid_labels, minlength=NUM_CLASSES)
    return int(np.argmax(counts))


# ---------------- 2) Build the TRAINING feature set ----------------
train_ids = read_image_ids("train")
if MAX_TRAIN_IMAGES is not None:
    train_ids = train_ids[:MAX_TRAIN_IMAGES]

X_train, y_train = [], []

print(f"Extracting superpixel features from {len(train_ids)} training images...")
for i, image_id in enumerate(train_ids):
    image = load_image_rgb(image_id)
    label_mask = load_label_mask(image_id)

    segments = slic(image, n_segments=N_SEGMENTS, compactness=COMPACTNESS,
                     start_label=0, channel_axis=2)

    for seg_id in np.unique(segments):
        seg_mask = segments == seg_id
        label = majority_label(seg_mask, label_mask)
        if label is None:
            continue  # skip superpixels that are mostly void/ambiguous
        X_train.append(channelwise_histogram_feature(image, seg_mask))
        y_train.append(label)

    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(train_ids)} images processed")

X_train = np.array(X_train)
y_train = np.array(y_train)
print(f"Collected {len(X_train)} labeled superpixels for training.")


# ---------------- 3) Train the Random Forest ----------------
clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
clf.fit(X_train, y_train)
print("Random Forest trained.")


# ---------------- 4-5) Inference + evaluation on the VAL split (our local "test set") ----------------
val_ids = read_image_ids("val")
if MAX_VAL_IMAGES is not None:
    val_ids = val_ids[:MAX_VAL_IMAGES]

# Accumulate a confusion matrix across the WHOLE val set, then derive IoU from
# it - the standard way to get a corpus-level mIoU, rather than averaging
# separately-computed per-image mIoU values (which weights small images
# and large images equally, which corpus-level IoU does not).
confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
qualitative_example = None  # we'll grab the first image's visualization to show at the end

print(f"\nEvaluating on {len(val_ids)} validation images...")
for i, image_id in enumerate(val_ids):
    image = load_image_rgb(image_id)
    label_mask = load_label_mask(image_id)

    segments = slic(image, n_segments=N_SEGMENTS, compactness=COMPACTNESS,
                     start_label=0, channel_axis=2)

    seg_ids = np.unique(segments)
    features = np.array([channelwise_histogram_feature(image, segments == s) for s in seg_ids])
    predictions = clf.predict(features)

    predicted_mask = np.zeros(label_mask.shape, dtype=np.uint8)
    for seg_id, pred_label in zip(seg_ids, predictions):
        predicted_mask[segments == seg_id] = pred_label

    valid = label_mask != VOID_LABEL  # exclude void/ignore pixels from evaluation
    gt_flat = label_mask[valid].astype(np.int64)
    pred_flat = predicted_mask[valid].astype(np.int64)

    # Vectorized confusion-matrix update (a python-level per-pixel loop here
    # would be far too slow across a whole dataset's worth of pixels)
    idx = gt_flat * NUM_CLASSES + pred_flat
    confusion += np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

    if qualitative_example is None:
        colored_prediction = VOC_COLORMAP[predicted_mask]
        colored_gt = VOC_COLORMAP[label_mask]
        qualitative_example = (image, colored_gt, colored_prediction)

    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(val_ids)} images evaluated")

# ---------------- Per-class IoU and mean IoU from the accumulated confusion matrix ----------------
print("\nPer-class IoU:")
ious = []
for c in range(NUM_CLASSES):
    tp = confusion[c, c]
    fp = confusion[:, c].sum() - tp
    fn = confusion[c, :].sum() - tp
    denom = tp + fp + fn
    if denom == 0:
        continue  # this class never appeared in ground truth OR predictions - skip it
    iou = tp / denom
    ious.append(iou)
    print(f"  {CLASSES[c]:15s}: {iou:.3f}")

mean_iou = float(np.mean(ious))
print(f"\nMean IoU (mIoU) over {len(ious)} present classes: {mean_iou:.3f}")


# ---------------- 6) Show one qualitative example ----------------
original_rgb, colored_gt_rgb, colored_pred_rgb = qualitative_example
original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
colored_gt_bgr = cv2.cvtColor(colored_gt_rgb, cv2.COLOR_RGB2BGR)
colored_pred_bgr = cv2.cvtColor(colored_pred_rgb, cv2.COLOR_RGB2BGR)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Ground Truth (VOC colors)", cv2.WINDOW_NORMAL)
cv2.namedWindow("Predicted (VOC colors)", cv2.WINDOW_NORMAL)
cv2.imshow("Original", original_bgr)
cv2.imshow("Ground Truth (VOC colors)", colored_gt_bgr)
cv2.imshow("Predicted (VOC colors)", colored_pred_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Task

1. Run the pipeline (on a subsample first via `MAX_TRAIN_IMAGES`/`MAX_VAL_IMAGES` while developing) and report the per-class IoU table and overall mIoU.
2. Try different `N_SEGMENTS` (e.g. 100, 400) and `COMPACTNESS` values — how does superpixel granularity trade off against mIoU and runtime?
3. Look at which classes have the lowest IoU. Do they tend to be classes with few training examples, classes that are visually similar to others (e.g. cat/dog), or classes with a lot of internal color variation?

---

## 🧪 Student Task 1 — Better Superpixel Features

The baseline above uses only a **plain RGB channel histogram** per superpixel — it captures color, and nothing else. Improve it using **texture, HSV, or any combination you can think of**. Some directions:

- Swap or add an **HSV histogram** instead of (or alongside) RGB — as in Lab 4, this can separate color identity from lighting/shading more cleanly.
- Add a **texture feature** per superpixel — e.g. a Local Binary Pattern histogram (`skimage.feature.local_binary_pattern`) computed over the region, or a simpler statistic like local intensity variance.
- Combine multiple feature blocks by **concatenating** them — as in Lab 4's combined-feature task, normalize each block *before* concatenation so one feature type doesn't dominate purely due to differing raw scales.

**Requirements:**

1. Keep everything else identical to Section 3 (same SLIC parameters, same Random Forest, same train/val split, same evaluation methodology) so your result is directly comparable to the baseline.
2. Report the new per-class IoU and mIoU, side by side with the Section 3 baseline numbers.
3. Discuss which classes improved the most, and hypothesize why your added feature(s) would help specifically for those classes (e.g. does a texture feature help distinguish `cat` from `sofa` when both are similarly-colored?).

> 📌 No reference solution is included in this lab sheet.

---

## 4️⃣ Content-Based Image Retrieval (CBIR): Theory Recap

Retrieval is a natural fit for **KNN thinking**: encode every image (query and pool) into a feature vector, and the "answer" to a query is simply its nearest neighbor(s) in that feature space — there's no separate classifier to train at all. The real design decisions are:

- **What encoding to use** — any of Lab 3/4's feature types (SIFT-BoVW, color histograms, HOG, LBP, or a combination) could work here; richer/more discriminative encodings generally retrieve better but are slower to compute and compare.
- **Whether to index ahead of time** — encoding every pool image *once* and storing the vectors (rather than re-encoding the whole pool on every query) is the difference between a usable system and an unusably slow one.
- **Which search structure to use** — brute-force distance comparison against every pool vector (`sklearn.neighbors.NearestNeighbors(algorithm="brute")`) is simple and exact; tree-based structures (`algorithm="kd_tree"` or `"ball_tree"`) can be faster in lower dimensions but degrade in high-dimensional feature spaces — this is exactly the kind of design trade-off worth experiencing hands-on rather than being told the answer to.

---

## 🧪 Student Task 2 — Image-Based Search on Caltech-256

### Task

Build a command-line image retrieval tool:

1. Takes a **query image path from the command line** when the program starts.
2. Encodes the query image and searches a **universal pool** of Caltech-256 images (spanning **many/all classes**, not just the classes you'll test with — the pool needs to contain plenty of "wrong answer" distractors for the accuracy numbers to mean anything).
3. Displays the **query image** in its own window, and the **top-10 retrieved images**, each in its **own window**.
4. Since you know each pool image's true class from its Caltech-256 folder path, label each retrieved image's window title with whether it's correct — e.g. `"Rank 1 - correct"`, `"Rank 2 - wrong"`, using the query image's own folder-derived class as ground truth.

**Design choices left to you** (all reasonable, as long as you can justify your choice): which feature encoding to use, whether to precompute/index the pool ahead of time (strongly recommended — see the timing requirement below) or encode on-the-fly, and which search structure (brute-force vs. a tree-based method, or something else entirely) to use.

### Taking the query image path from the command line

```python
import argparse

parser = argparse.ArgumentParser(description="Query-based image retrieval on Caltech-256")
parser.add_argument("query_image_path", help="Path to the query image")
args = parser.parse_args()

query_path = args.query_image_path
```

(A simpler alternative is `sys.argv[1]`, but `argparse` gives you a usable `--help` and clearer error messages for free.)

### Evaluation

1. **Accuracy:** report both **top-5** and **top-10** retrieval accuracy. Precisely define what "accuracy" means for your report before computing it — two common, both-valid choices are (a) **hit@k**: did *at least one* of the top-`k` results share the query's class (1 or 0 per query), or (b) **precision@k**: *what fraction* of the top-`k` results shared the query's class. State clearly which one you used.
2. **Speed:** measure the wall-clock time of the **retrieval process itself** — from just before your search call starts to when the ranked results are ready — not including window/image display time. Use `time.perf_counter()`:

```python
import time

start = time.perf_counter()
# ... encode query + search the index ...
elapsed = time.perf_counter() - start
print(f"Retrieval took {elapsed * 1000:.1f} ms")
```

3. Repeat the experiment for **3 sample query images from each of 10 classes** (30 queries total), and compile a report of top-5 accuracy, top-10 accuracy, and retrieval speed across all 30. You may automate this part (a batch script reusing your encode/search functions, without popping up 30 sets of windows) — but the single-query CLI tool with the labeled display windows described above should still exist and work on its own, since that's what you'll use to sanity-check correctness labeling and produce screenshots for your report.

> 📌 No reference solution is included in this lab sheet.
