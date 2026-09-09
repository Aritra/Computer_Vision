# 🖥️ Computer Vision Lab 4 — Classical Machine Learning for Vision

## 🎯 Objectives

In this lab, you will:

1. Segment an image by color using **K-Means clustering on HSV pixels**, with an interactive slider for the number of clusters.
2. Build a full classical image-classification pipeline on **Caltech-256**: SIFT features → a Bag-of-Visual-Words vocabulary (K-Means) → a multiclass SVM, evaluated with precision/recall/F1.
3. Improve on that baseline with **richer feature engineering** and **hierarchical classification**.
4. Extend the K-Means segmentation tool to **autonomously choose** the number of clusters for a given image.

---

## 0️⃣ Prerequisites

- `cv-env` (or `cvlab`) — everything in this lab uses plain OpenCV windows (Qt/GTK-backed `cv2.imshow()`, trackbars), **not** PySide6.
- Install the additional libraries this lab needs:

```bash
conda activate cv-env
pip install scikit-learn scikit-image matplotlib joblib
```

- `scikit-learn` — K-Means, train/val/test splitting, the SGD-based SVM, and evaluation metrics.
- `scikit-image` — used later for texture (LBP) and gradient (HOG) descriptors.
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

## 2️⃣ Classical Image Classification: Caltech-256 with SIFT + Bag-of-Visual-Words + SVM

### Dataset

Download **Caltech-256** from Kaggle: https://www.kaggle.com/datasets/jessicali9530/caltech256

It's organized as **one folder per class** (e.g. `256_ObjectCategories/001.ak47/`, `.../002.american-flag/`, ...) containing full-resolution JPEG images — 256 object categories plus a "clutter" background class. Unlike a dataset that ships pre-packaged arrays, you'll be reading images directly off disk here.

### The pipeline

We'll follow this exact sequence: **DataLoader → train/val/test split → SIFT on grayscale images → K-Means (k=64) vocabulary on all training SIFT descriptors → per-image Bag-of-Visual-Words (BoVW) histograms → multiclass SVM with a plotted loss curve → test-set evaluation (precision/recall/F1) → save the trained pipeline to disk.**

> 💡 A quick note on the SVM + loss curve: `sklearn.svm.SVC` solves an exact optimization problem in one shot, so there's no per-epoch loss to plot. To get an actual training/validation loss curve, we instead use `SGDClassifier(loss="hinge")` — hinge loss trained via stochastic gradient descent **is** a (linear) SVM, just fit iteratively instead of solved exactly, which is exactly what lets us record a loss value after each epoch.

> ⚠️ **Runtime note:** Caltech-256 has no fixed train/test split and its images are much larger than a toy dataset's, so we (a) read images from disk one at a time instead of preloading everything into memory, and (b) subsample a fixed number of images per class (`MAX_IMAGES_PER_CLASS`) to keep runtime reasonable for a lab session — raise this constant if you have the time/compute to spare, for a stronger final model.

```python
"""
Lab 4 - Caltech-256 classification with SIFT + Bag-of-Visual-Words + SVM

Pipeline:
  1. DataLoader           -> walk the Caltech-256 folder-per-class structure
  2. Train / val / test split
  3. SIFT features on grayscale images
  4. K-Means (k=64) on ALL training SIFT descriptors -> visual vocabulary
  5. Bag-of-Visual-Words (BoVW) histogram per image
  6. Multiclass SVM (linear, trained via SGD so we can plot a loss curve)
  7. Test-set evaluation: precision / recall / F1
  8. Save the trained SVM + vocabulary + label names for later reuse
"""

import os
import random
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
CALTECH_DIR = "256_ObjectCategories"   # folder containing one subfolder per class, e.g. "001.ak47"
VOCAB_SIZE = 64                        # k for the visual-vocabulary K-Means
RESIZE_DIM = 128                       # resize every image to a consistent size before feature extraction
MAX_IMAGES_PER_CLASS = 50              # subsample for a lab-friendly runtime; raise this for a stronger model
NUM_EPOCHS = 30                        # SGD-SVM training epochs
RANDOM_STATE = 42


# ---------------- 1) DataLoader ----------------
def load_caltech256_paths(root_dir, max_per_class):
    """Walk the one-folder-per-class structure and return (filepath, label_id) pairs,
    subsampled to at most `max_per_class` images per class. We keep file PATHS, not
    loaded images, since Caltech-256 is too large to hold entirely in memory."""
    class_folders = sorted(
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    )
    label_names = [name.split(".", 1)[1] if "." in name else name for name in class_folders]

    rng = random.Random(RANDOM_STATE)
    filepaths, labels = [], []

    for label_id, folder in enumerate(class_folders):
        folder_path = os.path.join(root_dir, folder)
        images_in_class = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        rng.shuffle(images_in_class)
        for fname in images_in_class[:max_per_class]:
            filepaths.append(os.path.join(folder_path, fname))
            labels.append(label_id)

    return filepaths, np.array(labels), label_names


filepaths, labels, label_names = load_caltech256_paths(CALTECH_DIR, MAX_IMAGES_PER_CLASS)
print(f"Loaded {len(filepaths)} images across {len(label_names)} classes.")


# ---------------- 2) Train / val / test split ----------------
# Caltech-256 doesn't ship a fixed split, so we carve out all three ourselves (stratified,
# so every class is represented proportionally in each split).
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    filepaths, labels, test_size=0.30, stratify=labels, random_state=RANDOM_STATE
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=RANDOM_STATE
)

print(f"Train: {len(train_paths)}  Val: {len(val_paths)}  Test: {len(test_paths)}")


# ---------------- 3) SIFT features on grayscale images ----------------
sift = cv2.SIFT_create()


def extract_sift_descriptors(paths):
    """Returns a list of per-image descriptor arrays (each shape [n_keypoints, 128], or None).
    Images are read from disk one at a time here, rather than preloaded, since Caltech-256's
    full-resolution JPEGs are much larger than a toy dataset's packed arrays."""
    all_descriptors = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            all_descriptors.append(None)
            continue
        resized = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, descriptors = sift.detectAndCompute(gray, None)
        all_descriptors.append(descriptors)  # may be None if literally no keypoints were found
    return all_descriptors


print("Extracting SIFT descriptors (train)...")
train_descriptors = extract_sift_descriptors(train_paths)
print("Extracting SIFT descriptors (val)...")
val_descriptors = extract_sift_descriptors(val_paths)
print("Extracting SIFT descriptors (test)...")
test_descriptors = extract_sift_descriptors(test_paths)


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
    test_labels, test_predictions, target_names=label_names, zero_division=0
)
print("\nTest set performance:\n")
print(report)


# ---------------- 8) Save the trained pipeline for later reuse ----------------
joblib.dump(svm, "caltech256_svm.joblib")
joblib.dump(vocabulary, "caltech256_vocabulary.joblib")
joblib.dump(scaler, "caltech256_scaler.joblib")
joblib.dump(label_names, "caltech256_label_names.joblib")
print("Saved model, vocabulary, scaler, and label names to disk.")
```

### Classifying a Single Hardcoded Image

The training/testing script above is deliberately silent (no image windows) — it's meant to run end-to-end over the dataset. This separate script loads the saved model + vocabulary and classifies **one hardcoded image**, displaying it with its predicted label.

```python
"""
Lab 4 - Classify a single hardcoded image using the SVM + vocabulary saved
by the Caltech-256 training script above. Unlike that script, this one
DOES display the image, since it's meant for one-off, visual inspection.
"""

import cv2
import joblib
import numpy as np

IMAGE_PATH = "my_test_image.jpg"  # hardcoded path - change to whatever you want to classify
RESIZE_DIM = 128
VOCAB_SIZE = 64

svm = joblib.load("caltech256_svm.joblib")
vocabulary = joblib.load("caltech256_vocabulary.joblib")
scaler = joblib.load("caltech256_scaler.joblib")
label_names = joblib.load("caltech256_label_names.joblib")

sift = cv2.SIFT_create()

img = cv2.imread(IMAGE_PATH)
resized = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM), interpolation=cv2.INTER_AREA)
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

## 🧪 Student Task 1 — Better Feature Engineering + Hierarchical Classification

The SIFT-BoVW baseline from Section 2 is a reasonable starting point, but 256 visually diverse classes is a hard problem for a single flat classifier working from one feature type. Improve on it along **two independent axes**:

### A. Richer feature engineering

Replace or augment the SIFT-BoVW feature with a **combination of at least two different feature types**. Some options (not exhaustive — mix and match, or use your own ideas):

- **SIFT + Bag-of-Visual-Words** (as in Section 2) for local shape/texture structure.
- **Color histogram** — a histogram over HSV or RGB channels, capturing the overall color distribution of the image (something SIFT ignores almost entirely, since it works on grayscale).
- **Local Binary Patterns (LBP)** — a texture descriptor (`skimage.feature.local_binary_pattern`); build a histogram of LBP codes over the image, similar in spirit to how you built a BoVW histogram over SIFT words.
- **HOG (Histogram of Oriented Gradients)** — captures edge/gradient orientation structure across the image (`skimage.feature.hog`), a good complement to color and texture, since it encodes coarse shape.

When combining feature types, **normalize each type separately before concatenating** (e.g. each with its own `StandardScaler`, or each L2-normalized independently) — otherwise a feature type with naturally larger raw magnitudes will dominate the combined vector regardless of how useful it actually is.

### B. Hierarchical classification

Caltech-256 doesn't ship a built-in class hierarchy (unlike some datasets that provide both fine and coarse labels), so this is about **building your own** and using it to structure the classification problem:

1. Using your baseline model's training features, compute a **mean feature vector per class** (the centroid of all training samples belonging to that class).
2. Run a **hierarchical/agglomerative clustering** (e.g. `sklearn.cluster.AgglomerativeClustering`) over these 256 class-centroids to group visually/feature-similar classes into a smaller number of **coarse "super-groups."**
3. Train a **two-stage classifier**: first, a coarse classifier that predicts which super-group an image belongs to; then, a fine classifier — trained only on that super-group's classes — that predicts the exact class, conditioned on the coarse prediction.
4. Compare this hierarchical approach's overall accuracy/F1 against the flat 256-class baseline from Section 2. Report whether it helps, and reason about *why* — e.g. does it help most on classes that were being confused with visually similar ones under the flat classifier?

**Requirements:**

1. Report test-set precision, recall, and F1 for: (a) the Section 2 SIFT-only flat baseline, (b) your improved-features flat classifier, and (c) your hierarchical classifier — so all three are directly comparable.
2. Reuse the same SGD-based multiclass SVM approach with a plotted training/validation loss curve for each classifier you train.
3. Provide a separate hardcoded-single-image inference script (mirroring the one in Section 2) for your **best-performing** final pipeline.

**Some things to think about:**

- If your combined-feature or hierarchical approach doesn't clearly outperform the flat SIFT-only baseline, that's a legitimate and useful finding to report — not every enhancement helps on every dataset, and explaining *why* it didn't is valuable analysis in itself.
- Think about what happens to an image if the coarse classifier gets the super-group wrong — the fine classifier never even gets a chance at the right answer. How does this failure mode show up in your reported metrics compared to the flat baseline?

> 📌 As with previous assignments, no reference solution is included in this lab sheet.

---

## 🧪 Student Task 2 — Autonomous K Selection for K-Means Segmentation

In Section 1, you picked `K` by hand with a slider. Extend that tool so it can **suggest (or directly choose) a good value of `K` for a given image on its own**, without a person dragging a slider and eyeballing the result.

**This is intentionally open-ended.** You're free to explore standard clustering-evaluation heuristics as a starting point (e.g. the elbow method on K-Means inertia, silhouette score, gap statistic, or an information-criterion-style approach), but:

- **Don't just bolt on a generic textbook/AI-suggested heuristic without thinking about whether it actually fits this problem.** Most standard "best K" heuristics are designed for generic clustering of abstract data points — they don't know anything about what makes a *color segmentation* good or bad for a human looking at the result. Think about what "the right number of segments" should even mean here: is it about statistical cluster separation, about visual/perceptual color distinctness, about the number of contiguous regions after your morphological cleanup, or something else entirely?
- You're encouraged to combine ideas, adapt a standard metric to be more image-aware (e.g. weighting it by segment spatial coherence, or by how large/small the resulting connected regions are after morphological closing), or come up with your own criterion from scratch.

**Requirements:**

1. Your method should take only the image as input and output a suggested `K` (or a small ranked shortlist), without a human manually dragging a slider to find it.
2. Test it on at least 4 visually different images (e.g. a simple few-color object, a busy natural scene, a texture-heavy image, an image with a smooth gradient background) and report the `K` your method picked for each, alongside the resulting segmented output.
3. Critically evaluate your own method: does the suggested `K` actually look right to you on each image? Where does it clearly succeed, and where does it clearly fail or feel arbitrary? An honest account of your method's limitations is expected and valued — this task is about the quality of your reasoning and experimentation, not about arriving at a perfect automatic answer.
4. Briefly document (a couple of paragraphs) the approach(es) you tried, including any that *didn't* work, and why you settled on your final one.

> 📌 As with previous tasks, no reference solution is included in this lab sheet.
