# 🖥️ Computer Vision Lab 4 — Classical Machine Learning for Vision

## 🎯 Objectives

In this lab, you will:

1. Segment an image by color using **K-Means clustering on HSV pixels**, with an interactive slider for the number of clusters.
2. Build a full classical image-classification pipeline on **CIFAR-100**: SIFT features → a Bag-of-Visual-Words vocabulary (K-Means) → a multiclass SVM, evaluated with precision/recall/F1.
3. Extend that pipeline on **Caltech-256** using richer, combined feature engineering (color, texture, and gradient-based features together).

---

## 0️⃣ Prerequisites

- `cv-env` (or `cvlab`) — everything in this lab uses plain OpenCV windows (Qt/GTK-backed `cv2.imshow()`, trackbars), **not** PySide6.
- Install the additional libraries this lab needs:

```bash
conda activate cv-env
pip install scikit-learn scikit-image matplotlib joblib
```

- `scikit-learn` — K-Means, train/val/test splitting, the SGD-based SVM, and evaluation metrics.
- `scikit-image` — used in the Caltech-256 task for texture (LBP) and gradient (HOG) descriptors.
- `joblib` — saving/loading trained models and the visual vocabulary to disk.

---

## 1️⃣ Color Segmentation with K-Means on HSV

### Why HSV?

In RGB, a color and a darker/lighter version of the same color can be numerically far apart, even though they're perceptually "the same color." **HSV** separates **Hue** (the color itself) from **Saturation** and **Value** (intensity/brightness), so clustering in HSV space groups pixels by color more the way we'd expect, rather than being thrown off by shading and lighting variation.

### Pipeline

1. Convert the image to HSV.
2. Cluster all HSV pixels into `k` groups using **K-Means** (`sklearn.cluster.KMeans`).
3. For each cluster, build a binary mask of "pixels belonging to this cluster," and apply **morphological closing** with a small (3-pixel) circular structuring element — this fills tiny holes/gaps within a color region before we summarize it.
4. Replace every pixel in each (cleaned) cluster with that cluster's **mean RGB color** (computed from the *original* color image, not HSV) — this is the final segmented output.

```python
import cv2
import numpy as np
from sklearn.cluster import KMeans

IMAGE_PATH = "sipi-dataset/misc/4.1.05.tiff"  # hardcoded path - change as needed

img = cv2.imread(IMAGE_PATH)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, w = img.shape[:2]
hsv_pixels = hsv.reshape(-1, 3).astype(np.float32)

# A roughly 3-pixel-diameter circular structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def segment_image(k):
    kmeans = KMeans(n_clusters=k, n_init=4, random_state=42)
    labels = kmeans.fit_predict(hsv_pixels)
    label_map = labels.reshape(h, w)

    output = np.zeros_like(img)
    for cluster_id in range(k):
        mask = (label_map == cluster_id).astype(np.uint8) * 255
        # Close small holes/gaps within this color's mask before averaging
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        bool_mask = closed_mask > 0
        if np.any(bool_mask):
            mean_color = img[bool_mask].mean(axis=0)  # mean RGB (BGR order), from the ORIGINAL image
            output[bool_mask] = mean_color

    return output


def on_trackbar(k):
    k = max(k, 2)  # guard in case a backend allows the slider to hit 0/1 before setTrackbarMin kicks in
    output = segment_image(k)
    cv2.imshow("Final Segmented Output", output)


cv2.namedWindow("HSV Output", cv2.WINDOW_NORMAL)
cv2.imshow("HSV Output", hsv)  # NOTE: displayed as if it were BGR, so this looks like "false colors" -
                                # that's expected, it's just a raw look at the data K-Means is clustering on

cv2.namedWindow("Final Segmented Output", cv2.WINDOW_NORMAL)
cv2.createTrackbar("K", "Final Segmented Output", 2, 20, on_trackbar)
cv2.setTrackbarMin("K", "Final Segmented Output", 2)  # keeps the slider from going below 2

on_trackbar(2)  # initial render at the default K=2

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Task

1. Drag the K slider from 2 up to 20. At what point does increasing K stop revealing genuinely new colors and start needlessly splitting a single visually-uniform region into multiple similar clusters (over-segmentation)?
2. Try this on a texture-heavy image and a flat-color/object image (e.g. from the SIPI Textures vs. Miscellaneous volumes). Does the same K value feel "right" for both?
3. Try removing the morphological closing step (skip straight from cluster mask to mean-color fill) — does the output look noticeably noisier/speckled without it?

---

## 2️⃣ Classical Image Classification: CIFAR-100 with SIFT + Bag-of-Visual-Words + SVM

### Dataset

Download **CIFAR-100 (Python version)** from Kaggle: https://www.kaggle.com/datasets/fedesoriano/cifar100

This is the original CIFAR-100 python-pickle distribution — you'll get three files: `train`, `test`, and `meta`, each a pickled dictionary (byte-string keys):

- `train` / `test`: `b"data"` (uint8 array, shape `(N, 3072)` — each row is 32×32 Red pixels, then 32×32 Green, then 32×32 Blue, all flattened), `b"fine_labels"` (100-class labels, 0–99), `b"coarse_labels"` (20 superclass labels).
- `meta`: `b"fine_label_names"` — the 100 human-readable class names, indexed by label ID.

### The pipeline

We'll follow this exact sequence: **DataLoader → train/val/test split → SIFT on grayscale images → K-Means (k=64) vocabulary on all training SIFT descriptors → per-image Bag-of-Visual-Words (BoVW) histograms → multiclass SVM with a plotted loss curve → test-set evaluation (precision/recall/F1) → save the trained pipeline to disk.**

> 💡 A quick note on the SVM + loss curve: `sklearn.svm.SVC` solves an exact optimization problem in one shot, so there's no per-epoch loss to plot. To get an actual training/validation loss curve, we instead use `SGDClassifier(loss="hinge")` — hinge loss trained via stochastic gradient descent **is** a (linear) SVM, just fit iteratively instead of solved exactly, which is exactly what lets us record a loss value after each epoch.

> ⚠️ **Runtime note:** running SIFT + a 64-word K-Means over the *full* 50,000-image training set is a lot of computation for a lab session. The code below subsamples a fixed number of images per class (`MAX_IMAGES_PER_CLASS`) to keep runtime reasonable — raise this constant if you have the time/compute to spare, for a stronger final model.

```python
"""
Lab 4 - CIFAR-100 classification with SIFT + Bag-of-Visual-Words + SVM

Pipeline:
  1. DataLoader           -> unpickle the CIFAR-100 python archive
  2. Train / val / test split
  3. SIFT features on grayscale images
  4. K-Means (k=64) on ALL training SIFT descriptors -> visual vocabulary
  5. Bag-of-Visual-Words (BoVW) histogram per image
  6. Multiclass SVM (linear, trained via SGD so we can plot a loss curve)
  7. Test-set evaluation: precision / recall / F1
  8. Save the trained SVM + vocabulary + label names for later reuse
"""

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# ---------------- Configuration ----------------
CIFAR_DIR = "cifar-100-python"   # folder containing 'train', 'test', 'meta'
VOCAB_SIZE = 64                  # k for the visual-vocabulary K-Means
RESIZE_DIM = 128                 # upscale before SIFT - CIFAR's native 32x32 is too small for a good SIFT response
MAX_IMAGES_PER_CLASS = 50        # subsample for a lab-friendly runtime; raise this for a stronger model
NUM_EPOCHS = 30                  # SGD-SVM training epochs
RANDOM_STATE = 42


# ---------------- 1) DataLoader ----------------
def unpickle(file_path):
    """CIFAR-100's python archive is a pickled dict with byte-string keys."""
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_cifar100_split(split_file):
    raw = unpickle(split_file)
    flat = raw[b"data"]                      # shape (N, 3072): R(1024) + G(1024) + B(1024), row-major
    labels = np.array(raw[b"fine_labels"])   # 0-99

    n = flat.shape[0]
    images = flat.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)  # -> (N, 32, 32, 3), RGB
    return images, labels


def subsample_per_class(images, labels, max_per_class):
    """Cap the number of images kept per class, for a manageable lab runtime."""
    rng = np.random.RandomState(RANDOM_STATE)
    keep_idx = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        chosen = rng.choice(cls_idx, size=min(max_per_class, len(cls_idx)), replace=False)
        keep_idx.extend(chosen)
    keep_idx = np.array(keep_idx)
    return images[keep_idx], labels[keep_idx]


meta = unpickle(f"{CIFAR_DIR}/meta")
fine_label_names = [name.decode("utf-8") for name in meta[b"fine_label_names"]]

train_images_full, train_labels_full = load_cifar100_split(f"{CIFAR_DIR}/train")
test_images, test_labels = load_cifar100_split(f"{CIFAR_DIR}/test")

train_images_full, train_labels_full = subsample_per_class(
    train_images_full, train_labels_full, MAX_IMAGES_PER_CLASS
)

print(f"Loaded {len(train_images_full)} training images, {len(test_images)} test images, "
      f"{len(fine_label_names)} classes.")


# ---------------- 2) Train / val / test split ----------------
# CIFAR-100 already gives us a held-out test set; carve a validation set out of the
# (subsampled) training set, stratified so every class is represented in both splits.
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images_full, train_labels_full,
    test_size=0.15, stratify=train_labels_full, random_state=RANDOM_STATE
)

print(f"Train: {len(train_images)}  Val: {len(val_images)}  Test: {len(test_images)}")


# ---------------- 3) SIFT features on grayscale images ----------------
sift = cv2.SIFT_create()


def extract_sift_descriptors(images):
    """Returns a list of per-image descriptor arrays (each shape [n_keypoints, 128], or None)."""
    all_descriptors = []
    for img in images:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Upscale first - at native 32x32, SIFT's scale-space finds almost nothing
        resized = cv2.resize(bgr, (RESIZE_DIM, RESIZE_DIM), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, descriptors = sift.detectAndCompute(gray, None)
        all_descriptors.append(descriptors)  # may be None if literally no keypoints were found
    return all_descriptors


print("Extracting SIFT descriptors (train)...")
train_descriptors = extract_sift_descriptors(train_images)
print("Extracting SIFT descriptors (val)...")
val_descriptors = extract_sift_descriptors(val_images)
print("Extracting SIFT descriptors (test)...")
test_descriptors = extract_sift_descriptors(test_images)


# ---------------- 4) K-Means visual vocabulary (k=64) on ALL training descriptors ----------------
# Fit the vocabulary on the TRAINING set's descriptors ONLY - fitting it on val/test
# descriptors too would leak information about those images into the vocabulary.
all_train_descriptors = np.vstack([d for d in train_descriptors if d is not None])
print(f"Total training SIFT descriptors: {all_train_descriptors.shape[0]}")

print(f"Clustering into a {VOCAB_SIZE}-word visual vocabulary...")
vocabulary = KMeans(n_clusters=VOCAB_SIZE, n_init=4, random_state=RANDOM_STATE)
vocabulary.fit(all_train_descriptors)


# ---------------- 5) Bag-of-Visual-Words histogram per image ----------------
def to_bovw_histogram(descriptors, vocabulary, vocab_size):
    """Assign each descriptor to its nearest visual word, and build a
    normalized histogram of word occurrences for the image."""
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(vocab_size, dtype=np.float32)  # no features found -> empty histogram

    word_ids = vocabulary.predict(descriptors)
    histogram, _ = np.histogram(word_ids, bins=np.arange(vocab_size + 1))
    histogram = histogram.astype(np.float32)

    norm = np.linalg.norm(histogram)
    if norm > 0:
        histogram /= norm  # L2-normalize so keypoint COUNT doesn't dominate over word DISTRIBUTION

    return histogram


def build_feature_matrix(descriptor_list, vocabulary, vocab_size):
    return np.array([to_bovw_histogram(d, vocabulary, vocab_size) for d in descriptor_list])


X_train = build_feature_matrix(train_descriptors, vocabulary, VOCAB_SIZE)
X_val = build_feature_matrix(val_descriptors, vocabulary, VOCAB_SIZE)
X_test = build_feature_matrix(test_descriptors, vocabulary, VOCAB_SIZE)

# Standardize features (zero mean, unit variance) - helps SGD converge faster and more stably
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# ---------------- 6) Multiclass SVM via SGD (so we can track a loss curve) ----------------
def multiclass_hinge_loss(decision_values, true_labels, classes):
    """Approximate one-vs-rest multiclass hinge loss from decision_function output."""
    y_binarized = np.array([[1 if c == label else -1 for c in classes] for label in true_labels])
    margins = np.clip(1 - y_binarized * decision_values, 0, None)
    return float(np.mean(margins))


svm = SGDClassifier(loss="hinge", random_state=RANDOM_STATE, learning_rate="optimal")
classes = np.unique(train_labels)

train_losses, val_losses = [], []

for epoch in range(NUM_EPOCHS):
    svm.partial_fit(X_train, train_labels, classes=classes)

    train_loss = multiclass_hinge_loss(svm.decision_function(X_train), train_labels, classes)
    val_loss = multiclass_hinge_loss(svm.decision_function(X_val), val_labels, classes)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Hinge loss")
plt.title("SVM (SGD) Training/Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_curve.png")
plt.show()


# ---------------- 7) Test-set evaluation ----------------
test_predictions = svm.predict(X_test)
report = classification_report(
    test_labels, test_predictions, target_names=fine_label_names, zero_division=0
)
print("\nTest set performance:\n")
print(report)


# ---------------- 8) Save the trained pipeline for later reuse ----------------
joblib.dump(svm, "cifar100_svm.joblib")
joblib.dump(vocabulary, "cifar100_vocabulary.joblib")
joblib.dump(scaler, "cifar100_scaler.joblib")
joblib.dump(fine_label_names, "cifar100_label_names.joblib")
print("Saved model, vocabulary, scaler, and label names to disk.")
```

### Classifying a Single Hardcoded Image

The training/testing script above is deliberately silent (no image windows) — it's meant to run end-to-end over the dataset. This separate script loads the saved model + vocabulary and classifies **one hardcoded image**, displaying it with its predicted label.

```python
"""
Lab 4 - Classify a single hardcoded image using the SVM + vocabulary saved
by the training script above. Unlike that script, this one DOES display
the image, since it's meant for one-off, visual inspection.
"""

import cv2
import joblib
import numpy as np

IMAGE_PATH = "my_test_image.jpg"  # hardcoded path - change to whatever you want to classify
RESIZE_DIM = 128
VOCAB_SIZE = 64

svm = joblib.load("cifar100_svm.joblib")
vocabulary = joblib.load("cifar100_vocabulary.joblib")
scaler = joblib.load("cifar100_scaler.joblib")
label_names = joblib.load("cifar100_label_names.joblib")

sift = cv2.SIFT_create()

img = cv2.imread(IMAGE_PATH)
resized = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
_, descriptors = sift.detectAndCompute(gray, None)

if descriptors is None:
    histogram = np.zeros(VOCAB_SIZE, dtype=np.float32)
else:
    word_ids = vocabulary.predict(descriptors)
    histogram, _ = np.histogram(word_ids, bins=np.arange(VOCAB_SIZE + 1))
    histogram = histogram.astype(np.float32)
    norm = np.linalg.norm(histogram)
    if norm > 0:
        histogram /= norm

feature_vector = scaler.transform(histogram.reshape(1, -1))
predicted_class_id = svm.predict(feature_vector)[0]
predicted_label = label_names[predicted_class_id]

print(f"Predicted class: {predicted_label}")

display_img = img.copy()
cv2.putText(display_img, predicted_label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
cv2.imshow("Prediction", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Task

1. Run the full pipeline and report the test-set precision, recall, and F1 (from `classification_report`).
2. Try `VOCAB_SIZE` values other than 64 (e.g. 32, 128) — how does test performance change? (Remember: changing this means re-running vocabulary fitting *and* re-building every BoVW histogram, since the histogram length depends on it.)
3. Try `MAX_IMAGES_PER_CLASS` at a higher value if your machine can handle it, and see how much test performance improves with more training data.

---

## 🧪 Student Assignment — Caltech-256 with Combined Feature Engineering

### Dataset

Download **Caltech-256** from Kaggle: https://www.kaggle.com/datasets/jessicali9530/caltech256

Unlike CIFAR-100, this dataset is organized as **one folder per class** (e.g. `256_ObjectCategories/001.ak47/`, `.../002.american-flag/`, ...) containing full-resolution JPEG images — 256 object categories plus a "clutter" background class.

### Task

Follow the **same overall pipeline** as Section 2 (DataLoader → train/val/test split → feature extraction → vocabulary/features → SVM with a loss curve → test evaluation with precision/recall/F1 → a separate single-image inference script) — but this time, replace or augment the SIFT-only feature with a **combination of at least two different feature types**. Some options (not exhaustive — mix and match, or use your own ideas):

- **SIFT + Bag-of-Visual-Words** (as in Section 2) for local shape/texture structure.
- **Color histogram** — a histogram over HSV or RGB channels, capturing the overall color distribution of the image (something SIFT ignores almost entirely, since it works on grayscale).
- **Local Binary Patterns (LBP)** — a texture descriptor (`skimage.feature.local_binary_pattern`); build a histogram of LBP codes over the image, similar in spirit to how you built a BoVW histogram over SIFT words.
- **HOG (Histogram of Oriented Gradients)** — captures edge/gradient orientation structure across the image (`skimage.feature.hog`), a good complement to color and texture, since it encodes coarse shape.

**Requirements:**

1. Extract **at least two** different feature types per image, and **concatenate** them into one combined feature vector per image. Normalize each feature type *before* concatenating (e.g. with its own `StandardScaler`, or L2-normalizing each block separately) — otherwise, a feature type with naturally larger raw magnitudes will dominate the combined vector regardless of how useful it actually is.
2. Reuse the same SGD-based multiclass SVM approach with a plotted training/validation loss curve.
3. Report test-set precision, recall, and F1, the same way as Section 2.
4. **Compare against a SIFT-only baseline** — re-run (or reuse) the Section 2-style pipeline on Caltech-256 with SIFT-BoVW alone, and report whether your combined-feature approach improves on it, and by how much.
5. Provide a separate hardcoded-single-image inference script, mirroring the one in Section 2, that works with your saved combined-feature pipeline.

**Some things to think about:**

- Caltech-256 images vary a lot in size and aspect ratio, unlike CIFAR-100's fixed 32×32 — you'll likely want to resize every image to a consistent size before extracting any feature, so your feature vectors stay comparable across images.
- 256 classes is a lot more than 100 — think about whether your `MAX_IMAGES_PER_CLASS`-style subsampling (needed here too, for the same runtime reasons as Section 2) needs to be larger to give the SVM enough signal per class.
- If your combined feature vector doesn't clearly outperform the SIFT-only baseline, that's a legitimate and useful finding to report — not every feature combination helps on every dataset, and explaining *why* it didn't (e.g. background clutter dominating the color histogram) is valuable analysis in itself.

> 📌 As with previous assignments, no reference solution is included in this lab sheet.
