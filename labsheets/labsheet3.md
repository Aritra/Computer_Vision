# 🖥️ Computer Vision Lab 3 — Feature Detectors, Descriptors & Matching

## 🎯 Objectives

In this lab, you will:

1. Understand the **structure tensor** and visualize it as an eigenvector-based ellipse at any point you click on an image.
2. Compare classical feature **detectors and descriptors**: Harris, SIFT, FAST, BRIEF, ORB, AKAZE, and FREAK — including their orientation/scale where applicable.
3. Learn the working principles of **BFMatcher** and **FlannMatcher**.
4. Build a **feature matching pipeline** between two images with a configurable detector + matcher.
5. Apply **RANSAC** to separate inlier matches from outlier matches, and visualize both.
6. Build a **folder-based image stitching application** using everything above.

---

## 0️⃣ Prerequisites & Dataset

- `cv-env` (or `cvlab`) — used for Sections 2–6 (plain OpenCV scripts, `cv2.imshow()`, resizable windows). Make sure `opencv-contrib-python` is installed (BRIEF and FREAK live in the `cv2.xfeatures2d` module, which ships only with the contrib package).
- `cv_gui` — used for Section 1 (PySide6 GUI). Recall this uses `opencv-python-headless`, `opencv-contrib-python-headless`, `numpy<2`, and `PySide6`.

### Dataset: HPatches

Download the **HPatches** dataset: https://github.com/hpatches

We want the **full image sequences** (not just the pre-extracted patches), since we need whole images to run detectors, matchers, and homography estimation on:

- Full image sequences (~1.3 GB): https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip

Download and extract it. Each sequence is a folder named `i_X` (illumination changes) or `v_X` (viewpoint changes), containing:

- `1.ppm` through `6.ppm` — `1.ppm` is the reference image, `2.ppm`–`6.ppm` are the same scene under increasing amounts of illumination or viewpoint change.
- `H_1_2`, `H_1_3`, ... `H_1_6` — text files with the ground-truth 3×3 homography mapping image `1` to each other image.

For the exercises in this lab, pick a **viewpoint** sequence (`v_*`) — these have genuine perspective change between images, which is what feature matching and homography estimation are actually for. `i_*` sequences (illumination-only) are more useful later if you want to specifically test how robust a descriptor is to lighting changes rather than geometry.

---

## 1️⃣ Structure Tensor: Eigenvector-Based Ellipse Visualization

### Theory

At every pixel, we can measure how the image gradient behaves in a small neighborhood using the **structure tensor** (also called the second-moment matrix):

```
M = Σ (over a local window, optionally weighted) [ Ix²   IxIy ]
                                                   [ IxIy   Iy² ]
```

where `Ix` and `Iy` are the image gradients (from a Sobel operator, for example). `M` is a 2×2 symmetric matrix, so it has two real eigenvalues `λ1 ≥ λ2 ≥ 0` and two orthogonal eigenvectors `e1, e2`.

- **`e1`** (the eigenvector for the larger eigenvalue `λ1`) points in the direction where the gradient changes **most**.
- **`e2`** points in the direction where the gradient changes **least**.
- The **magnitudes** `λ1, λ2` tell you how strong that variation is in each direction.

This is exactly the information Harris corner detection is built on:

- `λ1` and `λ2` both large → **corner** (strong gradient variation in every direction).
- `λ1` large, `λ2` small → **edge** (variation only across the edge, none along it).
- `λ1` and `λ2` both small → **flat region** (no meaningful structure).

Drawing an **ellipse** at a point, with its major axis along `e1` (length `∝ √λ1`) and minor axis along `e2` (length `∝ √λ2`), gives a direct visual of this: round ellipses sit on corners, thin elongated ellipses sit on edges (stretched along the edge direction), and tiny/near-invisible ellipses sit on flat regions.

### PySide6 App: Click-to-Visualize Structure Tensor Ellipse

This app loads an image, and draws the structure-tensor ellipse at every point you click, using the true-size-in-`QScrollArea` pattern from earlier labs.

```python
import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QSpinBox, QMessageBox
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


class ClickableImageLabel(QLabel):
    """A QLabel that reports click coordinates (in image-pixel space) to its parent."""

    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window

    def mousePressEvent(self, event):
        if self.pixmap() is None:
            return
        pos = event.position().toPoint()
        self.parent_window.handle_click(pos.x(), pos.y())


class StructureTensorViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lab 3 - Structure Tensor Ellipse Viewer")
        self.resize(900, 750)

        self.original_bgr = None
        self.gray = None
        self.Ix2 = None   # precomputed Ix^2, once per loaded image
        self.Iy2 = None   # precomputed Iy^2
        self.Ixy = None   # precomputed Ix*Iy
        self.click_points = []
        self.display_buffer = None  # keeps QImage's backing memory alive

        # --- Image display (true size, inside a scroll area) ---
        self.image_label = ClickableImageLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setMinimumSize(700, 500)

        # --- Controls ---
        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)

        self.clear_button = QPushButton("Clear Points")
        self.clear_button.clicked.connect(self.clear_points)

        self.window_size_box = QSpinBox()
        self.window_size_box.setRange(5, 101)
        self.window_size_box.setSingleStep(2)
        self.window_size_box.setValue(21)

        self.scale_box = QSpinBox()
        self.scale_box.setRange(1, 50)
        self.scale_box.setValue(5)

        controls_row = QHBoxLayout()
        controls_row.addWidget(self.open_button)
        controls_row.addWidget(QLabel("Window size (px):"))
        controls_row.addWidget(self.window_size_box)
        controls_row.addWidget(QLabel("Ellipse scale:"))
        controls_row.addWidget(self.scale_box)
        controls_row.addWidget(self.clear_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.scroll_area)
        main_layout.addLayout(controls_row)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "All Files (*);;Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.ppm)"
        )
        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            QMessageBox.warning(self, "Error", "Failed to load image.")
            return

        self.original_bgr = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Precompute gradients and gradient-product images ONCE per image,
        # so each click only has to sum a small window, not recompute Sobel.
        Ix = cv2.Sobel(self.gray, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(self.gray, cv2.CV_32F, 0, 1, ksize=3)
        self.Ix2 = Ix * Ix
        self.Iy2 = Iy * Iy
        self.Ixy = Ix * Iy

        self.click_points = []
        self.render_image(self.original_bgr)

    def clear_points(self):
        self.click_points = []
        if self.original_bgr is not None:
            self.render_image(self.original_bgr)

    def handle_click(self, x, y):
        if self.original_bgr is None:
            return
        self.click_points.append((x, y))
        self.redraw_with_ellipses()

    def structure_tensor_at(self, x, y, half_win):
        """Sum the precomputed gradient-product images over a window
        centered at (x, y), then eigen-decompose the resulting 2x2 matrix."""
        h, w = self.gray.shape
        x0, x1 = max(0, x - half_win), min(w, x + half_win + 1)
        y0, y1 = max(0, y - half_win), min(h, y + half_win + 1)

        Sxx = float(np.sum(self.Ix2[y0:y1, x0:x1]))
        Syy = float(np.sum(self.Iy2[y0:y1, x0:x1]))
        Sxy = float(np.sum(self.Ixy[y0:y1, x0:x1]))

        M = np.array([[Sxx, Sxy], [Sxy, Syy]], dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(M)  # ascending order

        lam2, lam1 = eigenvalues[0], eigenvalues[1]     # lam1 = larger
        v2, v1 = eigenvectors[:, 0], eigenvectors[:, 1]  # matching eigenvectors

        return lam1, lam2, v1, v2

    def redraw_with_ellipses(self):
        preview = self.original_bgr.copy()
        half_win = self.window_size_box.value() // 2
        scale = self.scale_box.value()

        for (x, y) in self.click_points:
            lam1, lam2, v1, v2 = self.structure_tensor_at(x, y, half_win)

            lam1 = max(lam1, 0.0)  # guard against tiny negative floating-point noise
            lam2 = max(lam2, 0.0)

            axis1 = max(int(scale * np.sqrt(lam1)), 2)
            axis2 = max(int(scale * np.sqrt(lam2)), 2)
            angle_deg = float(np.degrees(np.arctan2(v1[1], v1[0])))

            cv2.ellipse(preview, (x, y), (axis1, axis2), angle_deg, 0, 360, (0, 255, 255), 2)
            cv2.circle(preview, (x, y), 3, (0, 0, 255), -1)

        self.render_image(preview, keep_points=True)

    def render_image(self, cv_image, keep_points=False):
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.display_buffer = np.ascontiguousarray(rgb)
        h, w, ch = self.display_buffer.shape
        qimg = QImage(self.display_buffer.data, w, h, ch * w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())

        if not keep_points:
            self.click_points = []


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StructureTensorViewer()
    window.show()
    sys.exit(app.exec())
```

### Try it

1. Load an image and click on a clear **corner** (e.g. a window frame, table edge intersection) — the ellipse should look roughly round.
2. Click along a clean straight **edge** — the ellipse should be visibly elongated, with its long axis running *along* the edge.
3. Click on a **flat/textureless** patch — the ellipse should nearly disappear.
4. Try different **window sizes**: a very small window is noisy and local; a very large window blurs together nearby structures. What size seems to give the most stable, interpretable ellipses on your image?

---

## 2️⃣ Classical Feature Detectors & Descriptors

We'll now compare **Harris**, **SIFT**, **FAST**, **BRIEF**, **ORB**, **AKAZE**, and **FREAK** on the same image, each in its own resizable window, drawing orientation and scale wherever the algorithm provides them.

A quick map of what each one actually is:

| Name | Detects keypoints? | Computes a descriptor? | Has scale? | Has orientation? |
|---|:---:|:---:|:---:|:---:|
| Harris | ✅ (corners only) | ❌ | ❌ | ❌ |
| SIFT | ✅ | ✅ (float, 128-d) | ✅ | ✅ |
| FAST | ✅ | ❌ | ❌ | ❌ |
| BRIEF | ❌ (needs an external detector) | ✅ (binary) | inherited from detector | inherited from detector |
| ORB | ✅ | ✅ (binary) | ✅ | ✅ |
| AKAZE | ✅ | ✅ (binary by default) | ✅ | ✅ |
| FREAK | ❌ (needs an external detector) | ✅ (binary) | inherited from detector | inherited from detector |

Since **BRIEF** and **FREAK** are descriptors only, we pair them with a detector that *does* provide scale/orientation (so their circles in the visualization aren't all identical and orientation-less).

```python
import cv2
import numpy as np

img = cv2.imread("hpatches-sequences-release/v_wall/1.ppm")  # hardcoded path — change as needed
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

RICH = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # draws circle sized to keypoint.size + orientation tick

# ---- Harris corners (no descriptor, no scale/orientation) ----
gray_f = np.float32(gray)
harris_response = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
harris_response = cv2.dilate(harris_response, None)  # helps mark local maxima more visibly
harris_img = img.copy()
threshold = 0.01 * harris_response.max()
ys, xs = np.where(harris_response > threshold)
for x, y in zip(xs, ys):
    cv2.circle(harris_img, (int(x), int(y)), 3, (0, 0, 255), 1)

# ---- SIFT ----
sift = cv2.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(gray, None)
sift_img = cv2.drawKeypoints(img, kp_sift, None, color=(0, 255, 0), flags=RICH)

# ---- FAST (no scale/orientation of its own) ----
fast = cv2.FastFeatureDetector_create(threshold=25)
kp_fast = fast.detect(gray, None)
fast_img = cv2.drawKeypoints(img, kp_fast, None, color=(255, 0, 0), flags=RICH)

# ---- BRIEF (descriptor only — paired here with a STAR detector for keypoints) ----
star = cv2.xfeatures2d.StarDetector_create()
kp_star = star.detect(gray, None)
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
kp_brief, des_brief = brief.compute(gray, kp_star)
brief_img = cv2.drawKeypoints(img, kp_brief, None, color=(0, 255, 255), flags=RICH)

# ---- ORB ----
orb = cv2.ORB_create(nfeatures=500)
kp_orb, des_orb = orb.detectAndCompute(gray, None)
orb_img = cv2.drawKeypoints(img, kp_orb, None, color=(255, 0, 255), flags=RICH)

# ---- AKAZE ----
akaze = cv2.AKAZE_create()
kp_akaze, des_akaze = akaze.detectAndCompute(gray, None)
akaze_img = cv2.drawKeypoints(img, kp_akaze, None, color=(0, 128, 255), flags=RICH)

# ---- FREAK (descriptor only — paired here with AKAZE for scale+orientation) ----
kp_for_freak = akaze.detect(gray, None)
freak = cv2.xfeatures2d.FREAK_create()
kp_freak, des_freak = freak.compute(gray, kp_for_freak)
freak_img = cv2.drawKeypoints(img, kp_freak, None, color=(128, 0, 255), flags=RICH)

windows = {
    "Harris Corners": harris_img,
    "SIFT": sift_img,
    "FAST": fast_img,
    "BRIEF (on STAR keypoints)": brief_img,
    "ORB": orb_img,
    "AKAZE": akaze_img,
    "FREAK (on AKAZE keypoints)": freak_img,
}

for name, im in windows.items():
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, im)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

> 🔎 Notice: FAST's circles will all be the same size with no orientation tick — FAST reports neither scale nor orientation. BRIEF's circles inherit whatever the STAR detector reported (STAR does have scale, but no strong orientation estimate), while FREAK's circles inherit AKAZE's full scale+orientation, since we deliberately paired it with a richer detector.

### Hyperparameters to experiment with

| Detector | Key parameters | What they do |
|---|---|---|
| **Harris** | `blockSize`, `ksize`, `k` | `blockSize`: neighborhood size for summing the structure tensor (same idea as Section 1's window). `ksize`: Sobel aperture size for computing gradients. `k`: Harris's empirical sensitivity constant (typically 0.04–0.06) — higher `k` means fewer, stricter corners. |
| **SIFT** | `nfeatures`, `nOctaveLayers`, `contrastThreshold`, `edgeThreshold`, `sigma` | `nfeatures`: cap on keypoints retained (0 = no cap). `nOctaveLayers`: scale-space sampling density per octave. `contrastThreshold`: filters out low-contrast (weak) keypoints — higher means fewer, stronger keypoints. `edgeThreshold`: filters out edge-like keypoints — higher means *more* kept along edges (counter-intuitive, check the docs). `sigma`: initial Gaussian blur applied before building the scale-space. |
| **FAST** | `threshold`, `nonmaxSuppression`, `type` | `threshold`: intensity difference required for a pixel to count as brighter/darker than the center — lower gives more (noisier) keypoints. `nonmaxSuppression`: suppresses weaker nearby detections. `type`: which circular pixel pattern to test (e.g. `TYPE_9_16`). |
| **BRIEF** | `bytes`, `use_orientation` | `bytes`: descriptor length (16/32/64) — longer is more discriminative but slower to match. `use_orientation`: whether to rotate the sampling pattern using the keypoint's orientation (needs a detector that provides one) for rotation invariance. |
| **ORB** | `nfeatures`, `scaleFactor`, `nlevels`, `edgeThreshold`, `fastThreshold` | `nfeatures`: max keypoints retained. `scaleFactor`: pyramid decimation ratio between levels. `nlevels`: number of pyramid levels (controls the scale range covered). `edgeThreshold`: border size where features aren't detected. `fastThreshold`: threshold for the internal FAST stage used to find candidate keypoints. |
| **AKAZE** | `descriptor_type`, `descriptor_size`, `threshold`, `nOctaves`, `nOctaveLayers` | `descriptor_type`: e.g. MLDB (binary) vs KAZE (float). `descriptor_size`: 0 = full size. `threshold`: detector response threshold — lower gives more keypoints. `nOctaves`/`nOctaveLayers`: scale-space resolution, similar to SIFT. |
| **FREAK** | `orientationNormalized`, `scaleNormalized`, `patternScale`, `nOctaves` | `orientationNormalized`/`scaleNormalized`: toggle rotation/scale invariance on or off. `patternScale`: overall size of FREAK's retinal sampling pattern. `nOctaves`: precomputed scale levels for the pattern lookup. |

**Task:** for at least 3 of the detectors above, change 2–3 parameters each and observe how the *number* and *quality* (well-spread vs. clustered, on real structure vs. noise) of detected keypoints changes. Run this on both an `i_*` (illumination) and a `v_*` (viewpoint) HPatches sequence image and note any differences.

---

## 3️⃣ Theory: BFMatcher vs. FlannMatcher

Once you have descriptors for two images, you need to **match** them — for every descriptor in image A, find its most similar descriptor(s) in image B.

### BFMatcher (Brute-Force Matcher)

`cv2.BFMatcher` compares **every** descriptor in image A against **every** descriptor in image B, exhaustively, using a distance metric:

- `cv2.NORM_HAMMING` for **binary** descriptors (ORB, BRIEF, AKAZE-MLDB, FREAK) — counts differing bits.
- `cv2.NORM_L2` (or `NORM_L1`) for **float** descriptors (SIFT) — Euclidean distance.

It's simple and exact — it's guaranteed to find the true nearest neighbor(s) — but its cost grows as `O(N × M)` for `N` and `M` descriptors in each image, which gets slow once you have thousands of keypoints per image.

### FlannMatcher (Fast Library for Approximate Nearest Neighbors)

`cv2.FlannBasedMatcher` uses approximate nearest-neighbor search structures instead of brute force:

- **KD-trees** for float descriptors (SIFT-like).
- **LSH (Locality-Sensitive Hashing)** for binary descriptors (ORB/BRIEF/AKAZE/FREAK-like) — KD-trees don't work well on Hamming-distance binary data, so FLANN needs a different index type here.

This trades a small amount of accuracy (it may occasionally miss the *true* nearest neighbor) for a large speed advantage on bigger descriptor sets.

**Rule of thumb:** BFMatcher for smaller keypoint counts or when you need exact results; FlannMatcher when speed matters more and you have a lot of descriptors to match.

Both support `.match()` (single best match per descriptor) and `.knnMatch()` (`k` best matches per descriptor, which is what enables Lowe's ratio test below).

---

## 4️⃣ Feature Matching Between Two Images

This script matches two images using a **hardcoded choice** of feature type and matcher — change the constants at the top to experiment.

```python
import cv2
import numpy as np

# ---- Hardcoded configuration: change these to experiment ----
FEATURE = "SIFT"     # options: "SIFT", "ORB", "AKAZE"
MATCHER = "FLANN"    # options: "BF", "FLANN"
RATIO_TEST_THRESHOLD = 0.75

img1 = cv2.imread("hpatches-sequences-release/v_wall/1.ppm")
img2 = cv2.imread("hpatches-sequences-release/v_wall/2.ppm")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# --- Build the chosen feature detector/descriptor ---
if FEATURE == "SIFT":
    detector = cv2.SIFT_create()
    is_binary_descriptor = False
elif FEATURE == "ORB":
    detector = cv2.ORB_create(nfeatures=1000)
    is_binary_descriptor = True
elif FEATURE == "AKAZE":
    detector = cv2.AKAZE_create()
    is_binary_descriptor = True
else:
    raise ValueError(f"Unknown FEATURE: {FEATURE}")

kp1, des1 = detector.detectAndCompute(gray1, None)
kp2, des2 = detector.detectAndCompute(gray2, None)

# --- Build the chosen matcher ---
if MATCHER == "BF":
    norm_type = cv2.NORM_HAMMING if is_binary_descriptor else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm_type)
elif MATCHER == "FLANN":
    if is_binary_descriptor:
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)  # FLANN_INDEX_LSH
    else:
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
else:
    raise ValueError(f"Unknown MATCHER: {MATCHER}")

# --- Match with Lowe's ratio test: keep a match only if the best candidate
#     is meaningfully closer than the second-best (filters ambiguous matches) ---
knn_matches = matcher.knnMatch(des1, des2, k=2)
good_matches = []
for pair in knn_matches:
    if len(pair) == 2:
        m, n = pair
        if m.distance < RATIO_TEST_THRESHOLD * n.distance:
            good_matches.append(m)

print(f"{FEATURE} + {MATCHER}: {len(good_matches)} good matches out of {len(knn_matches)} candidates")

match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2.namedWindow("Feature Matches", cv2.WINDOW_NORMAL)
cv2.imshow("Feature Matches", match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Task:** Run every combination of `FEATURE ∈ {SIFT, ORB, AKAZE}` × `MATCHER ∈ {BF, FLANN}` (6 runs total) on the same HPatches viewpoint pair, and note: match count, visual match quality (matches that actually look geometrically consistent vs. scattered), and roughly how long each run takes. Also try `RATIO_TEST_THRESHOLD` values of `0.6`, `0.75`, and `0.9` — what happens to the match count and quality at each?

---

## 5️⃣ RANSAC: Separating Inliers from Outliers

Even after the ratio test, some matches will be wrong (ambiguous local patches that happen to look similar). **RANSAC** (Random Sample Consensus) fits a homography using random minimal subsets of matches, repeatedly, and keeps the fit that the largest number of matches agree with — those agreeing matches are the **inliers**, everything else is an **outlier**.

We'll visualize this directly: the two images placed **side by side** in one resizable window, with **green** lines for inlier matches and **red** lines for outlier matches.

```python
import cv2
import numpy as np

img1 = cv2.imread("hpatches-sequences-release/v_wall/1.ppm")
img2 = cv2.imread("hpatches-sequences-release/v_wall/2.ppm")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
knn_matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]

src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
mask = mask.ravel().tolist()

num_inliers = sum(mask)
print(f"{num_inliers} inliers out of {len(good_matches)} matches "
      f"({num_inliers / len(good_matches) * 100:.1f}%)")


def draw_matches_colored(img_a, kp_a, img_b, kp_b, matches, inlier_mask):
    """Place img_a and img_b side by side and draw each match:
    green if it's an inlier, red if it's an outlier."""
    h1, w1 = img_a.shape[:2]
    h2, w2 = img_b.shape[:2]
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img_a
    canvas[:h2, w1:w1 + w2] = img_b

    for m, is_inlier in zip(matches, inlier_mask):
        pt1 = tuple(np.round(kp_a[m.queryIdx].pt).astype(int))
        pt2 = tuple((np.round(kp_b[m.trainIdx].pt).astype(int) + np.array([w1, 0])))
        color = (0, 255, 0) if is_inlier else (0, 0, 255)  # green = inlier, red = outlier
        cv2.line(canvas, pt1, pt2, color, 1)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)

    return canvas


result = draw_matches_colored(img1, kp1, img2, kp2, good_matches, mask)

cv2.namedWindow("RANSAC: Inliers (green) vs Outliers (red)", cv2.WINDOW_NORMAL)
cv2.imshow("RANSAC: Inliers (green) vs Outliers (red)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Task:**

1. Vary `ransacReprojThreshold` (try `1.0`, `5.0`, `15.0`) — how does the number of inliers change as you loosen or tighten it?
2. Compare the HPatches ground-truth homography (`H_1_2`, etc., provided with the dataset) against the `H` estimated here — are they close? (You can apply both to the corner points of image 1 and compare where they land in image 2.)

---

## 🧪 Student Assignment — Folder-Based Image Stitching

### Datasets

Download from **visionxiang/Image-Stitching-Dataset**: https://github.com/visionxiang/Image-Stitching-Dataset

You need two datasets from that README's table, for evaluating your stitching application:

- **VPG Dataset (2020)** — 36 sets of images (12 synthetic, 24 real), 5–72 images per set: https://drive.google.com/drive/folders/1n9Pf2vqNpT1r7QAjjYnVnwZu_OfuMRhW
- **GES-50 (2022)** — 50 image groups, 2–35 images per group: https://github.com/flowerDuo/GES-GSP-Stitching/tree/master/Dataset

Each "set"/"group" in these datasets is a folder of images meant to be stitched into one panorama — exactly the input shape your application needs to handle.

### Task

Build an application that:

1. Takes a **folder path** as input, containing an arbitrary number of images intended to form one panorama.
2. Uses feature detection + matching + RANSAC (everything from Sections 2–5) to figure out **which images actually overlap** with which.
3. **Rejects** any image that doesn't have a strong enough match to any other image in the set (log which images were rejected and why — e.g. "fewer than N inliers with every other image").
4. Chains the pairwise homographies together and **warps every accepted image onto one common canvas**, producing a single stitched panorama — with the necessary perspective warping applied so the result looks like one continuous, uniform view (as much as is achievable — visible seams or minor ghosting is fine, you aren't expected to build production-grade blending).
5. Displays or saves the final stitched result.

**Requirements:**

- Your feature detector, matcher, and RANSAC parameters should be easy to change in one place (constants at the top of the script, same pattern as Section 4) — you'll need to tune these per-dataset.
- Document, for each folder you test, which images (if any) got rejected and why.
- Test on **at least 3 sets** from VPG and **at least 3 groups** from GES-50. Note where your application does well (clean, mostly-planar scenes) and where it struggles (wide parallax, few textured regions, extreme viewpoint change) — both datasets are explicitly designed to include some hard cases, so struggling on the harder ones is an expected and useful observation, not necessarily a bug.

**Some things to think about (no full solution given — this is the actual assignment):**

- With more than 2 images, you don't have a single pairwise match — you effectively have a small graph, where nodes are images and edges are "these two images have enough inlier matches to be considered overlapping." How will you decide which images to warp relative to which reference frame? (Hint: one common approach is picking one image as the anchor/reference and chaining homographies through the graph to express every other image's transform relative to that anchor.)
- Warping every image directly onto a canvas sized to the reference image alone will clip anything that extends beyond it. How will you size the output canvas so the whole stitched result fits? (Hint: project each image's corner points through its homography *before* warping, and use the bounding box of all projected corners to size the canvas — similar in spirit to the "unclipped rotation" trick from Lab 1, but for a general homography instead of a pure rotation.)
- `cv2.warpPerspective()` needs the destination canvas coordinates, not the original image's — you'll likely need to compose an extra translation into each homography to shift everything into positive canvas coordinates once you know the bounding box.
- For "reject if no strong match," reuse the inlier count from Section 5's RANSAC step as your criterion — decide on a sensible minimum inlier count (or ratio of inliers to total matches) yourself, and justify your choice in your report.

> 📌 As with previous assignments, no reference solution is included in this lab sheet. This is a substantial task — start early, and build it up incrementally: get 2-image stitching solid first, then extend to N images.
