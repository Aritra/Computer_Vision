# 🖥️ Computer Vision Lab 2 — Edge Detection, Morphology, Hough Transforms & Homography

## 🎯 Objectives

In this lab, you will:

1. Explore classical **edge detection** methods and study their differences.
2. Understand the impact of **threshold selection in Canny edge detection**.
3. Learn the **theory and practice of morphological operations** — erosion, dilation, opening, and closing.
4. Apply morphology to improve **contour detection** and **polygon approximation**.
5. Detect straight lines with the **Hough Line Transform** — both the full-line and line-segment variants.
6. Detect circles with the **Hough Circle Transform**.
7. Build a **non-ML surface crack detector** using classical thresholding + morphology.
8. Learn the theory of **homography / perspective transform**, and build a manual four-point "scan to paper size" tool with PySide6.
9. Extend that tool to detect the four corners **automatically**, using a threshold → morphology → edge → line → intersection pipeline.

> ⚠️ **Note on this year's shift (continued from Lab 1):** As with Lab 1, none of this lab uses a live webcam feed. Every experiment below runs on a **static image loaded from disk** (from the SIPI dataset, the crack dataset, or any image you provide). This keeps results reproducible and lets you tune parameters at your own pace instead of chasing a moving feed.

---

## 0️⃣ Prerequisites

- Complete **Lab 1** and have your environments ready:
  - `cv-env` (or `cvlab`) — used for Sections 1–7 below (plain OpenCV scripts, `cv2.imshow()`, trackbars).
  - `cv_gui` — used for Section 8 onward (PySide6 GUI work). Recall this environment uses `opencv-python-headless`, `opencv-contrib-python-headless`, `numpy<2`, and `PySide6`.
- Have a working copy of the **USC-SIPI dataset** from Lab 1 (Textures / Aerials / Miscellaneous / Sequences) — several exercises below use images from it. Any of your own images work too; the code just needs a valid path.
- For Section 7 (crack detector), download the **Surface Crack Detection** dataset from Kaggle: https://www.kaggle.com/datasets/arunrk7/surface-crack-detection — see that section for details.

---

## 1️⃣ Exploring Edge Detection

We'll compare three classical edge detectors on the same image: **Sobel**, **Laplacian of Gaussian (LoG)**, and **Canny**.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sipi-dataset/misc/4.1.05.tiff")  # pick any image with clear edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)  # mild denoising before edge detection

# --- Sobel ---
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)

# --- Laplacian of Gaussian (LoG): Gaussian blur, then Laplacian ---
blurred_for_log = cv2.GaussianBlur(gray, (5, 5), 0)
log = cv2.Laplacian(blurred_for_log, cv2.CV_64F, ksize=3)
log = cv2.convertScaleAbs(log)

# --- Canny ---
canny = cv2.Canny(gray, 100, 200)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
titles = ["Original (Gray)", "Sobel X", "Sobel Y", "Sobel Magnitude", "Laplacian of Gaussian", "Canny"]
images = [gray, cv2.convertScaleAbs(sobel_x), cv2.convertScaleAbs(sobel_y), sobel_magnitude, log, canny]

for ax, title, im in zip(axes.ravel(), titles, images):
    ax.imshow(im, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
```

### Task

1. Run this on at least two different images (e.g. one texture-heavy from the Textures volume, one natural/object image from Miscellaneous).
2. Note down:
   - How the output differs visually between Sobel, LoG, and Canny for the same image.
   - Situations where one detector performs better than the others (e.g. noisy images, fine texture, sharp object boundaries).

---

## 2️⃣ Studying Canny Thresholds

`cv2.Canny()` takes two thresholds — edges with gradient above the high threshold are always kept, edges below the low threshold are always discarded, and edges in between are kept only if connected to a strong edge. Use trackbars to explore this interactively:

```python
import cv2

img = cv2.imread("sipi-dataset/misc/4.1.05.tiff")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)

def nothing(x):
    pass

cv2.namedWindow("Canny Threshold Explorer", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Low Threshold", "Canny Threshold Explorer", 50, 500, nothing)
cv2.createTrackbar("High Threshold", "Canny Threshold Explorer", 150, 500, nothing)

while True:
    low = cv2.getTrackbarPos("Low Threshold", "Canny Threshold Explorer")
    high = cv2.getTrackbarPos("High Threshold", "Canny Threshold Explorer")

    edges = cv2.Canny(gray, low, high)
    cv2.imshow("Canny Threshold Explorer", edges)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
```

### Task

1. Drag both trackbars across their full range and observe:
   - How **low thresholds** make edges too noisy (lots of weak/spurious edges kept).
   - How **high thresholds** cause loss of important edges (real boundaries get discarded).
2. Identify a **balanced threshold range** for a clean, meaningful edge map on your chosen image, and note the values down.

---

## 3️⃣ Theory: Morphological Operations

**Morphological operations** process an image based on shape. They work by sliding a small matrix called a **structuring element** (or **kernel**) across the image and combining it with the local neighborhood of each pixel using a set operation (not a weighted sum, unlike convolution filters). They are most commonly applied to **binary** (black/white) images — typically the output of a threshold or edge-detection step — to clean up noise, close gaps, or reshape regions before further analysis like contour detection.

### The structuring element

```python
import cv2

kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
```

The **shape and size** of this kernel control how aggressively and in what pattern the operation affects the image. Larger kernels produce stronger effects; elliptical/cross kernels avoid the "blocky" artifacts a rectangular kernel can introduce on curved shapes.

### Erosion

**Erosion** "shrinks" white (foreground) regions: a pixel stays white only if **every** pixel under the kernel is white; otherwise it's set to black.

- **Purpose:** remove small white noise specks, detach weakly-connected objects, thin out object boundaries.
- **Trade-off:** it also shrinks and can break apart legitimate thin structures if overused.

```python
eroded = cv2.erode(binary_img, kernel, iterations=1)
```

### Dilation

**Dilation** does the opposite — it "grows" white regions: a pixel becomes white if **any** pixel under the kernel is white.

- **Purpose:** fill small holes/gaps inside objects, join nearby broken fragments (e.g. a dashed line into a solid one), thicken thin structures.
- **Trade-off:** it can merge distinct objects that are close together, and enlarges noise if applied before cleaning it up.

```python
dilated = cv2.dilate(binary_img, kernel, iterations=1)
```

### Opening (erosion → dilation)

**Opening** applies erosion first, then dilation, with the same kernel.

- **Purpose:** removes small noise blobs / thin protrusions from the foreground **without significantly shrinking** larger objects (the follow-up dilation restores their size, but the noise — having been fully eroded away — never comes back).
- **When to use:** cleaning up "salt" noise (small isolated white specks) after thresholding, before contour detection.

```python
opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
```

### Closing (dilation → erosion)

**Closing** applies dilation first, then erosion — the mirror image of opening.

- **Purpose:** fills small holes and gaps **inside** foreground objects and bridges small breaks in an object's boundary, without significantly growing the object overall.
- **When to use:** cleaning up "pepper" noise (small holes inside a solid region), or reconnecting a contour that Canny/thresholding broke into fragments.

```python
closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
```

### Quick reference

| Operation | Formula | Effect on foreground | Typical use |
|---|---|---|---|
| Erosion | shrink | removes small white noise, thins objects | isolating/separating touching objects |
| Dilation | grow | fills small gaps, thickens objects | joining broken fragments, thickening lines |
| Opening | erode → dilate | removes small noise blobs, preserves overall size | cleaning "salt" noise before contour detection |
| Closing | dilate → erode | fills small holes, preserves overall size | cleaning "pepper" noise, sealing broken boundaries |

### Demo: comparing all four on a binary mask

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("sipi-dataset/misc/4.1.05.tiff")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

eroded = cv2.erode(binary, kernel, iterations=1)
dilated = cv2.dilate(binary, kernel, iterations=1)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
titles = ["Binary (Otsu)", "Erosion", "Dilation", "Opening", "Closing"]
images = [binary, eroded, dilated, opened, closed]

for ax, title, im in zip(axes, titles, images):
    ax.imshow(im, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
```

### Task

1. Re-run the demo with kernel sizes `(3,3)`, `(7,7)`, and `(11,11)`, and with `iterations=1,2,3`. Note how the effect scales.
2. Try `cv2.MORPH_RECT` vs `cv2.MORPH_ELLIPSE` on an image with rounded objects — is the difference visible?

---

## 4️⃣ Contour Detection and Polygon Approximation

We'll segment a foreground object from a static image, clean the mask with morphology, then extract and simplify its contour.

```python
import cv2
import numpy as np

img = cv2.imread("your_object_image.jpg")   # any image with one clear foreground object
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Segment foreground from background
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Clean up the mask: opening removes small noise, closing seals small gaps
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours on the cleaned mask
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

result = img.copy()
cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)

# Polygon approximation: simplify the contour to fewer vertices
epsilon = 0.01 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)
cv2.drawContours(result, [approx], -1, (0, 0, 255), 2)

cv2.imshow("Contour (green) vs Approximated Polygon (red)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Task

1. Compare the contour found **with** and **without** the opening/closing cleanup step (comment it out and re-run). Is the contour smoother and more stable with cleanup?
2. Vary the `epsilon` factor (try `0.001`, `0.01`, `0.05` times the arc length) in `cv2.approxPolyDP()` and observe how the polygon simplifies — at what point does it stop resembling the original shape?

---

## 5️⃣ Hough Circle Transform

```python
import cv2
import numpy as np

img = cv2.imread("your_circles_image.jpg")  # e.g. coins, balls, wheels — any image with round objects
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)  # reduces false circle detections from noise

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,             # inverse ratio of accumulator resolution to image resolution
    minDist=30,        # minimum distance between detected circle centers
    param1=100,        # higher Canny threshold used internally
    param2=40,         # accumulator threshold — lower means more (possibly false) circles
    minRadius=10,
    maxRadius=100,
)

result = img.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0, :]:
        cv2.circle(result, (x, y), r, (0, 255, 0), 2)   # circle outline
        cv2.circle(result, (x, y), 2, (0, 0, 255), 3)   # circle center

cv2.imshow("Detected Circles", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Task

1. Vary `minDist`, `param2`, `minRadius`, and `maxRadius` one at a time and note how sensitive circle detection is to each.
2. Find a parameter set that produces false detections (circles that aren't really there), and one that misses real circles. What does a "good" middle ground look like for your image?

---

## 6️⃣ Hough Line Transform

Hough line detection comes in two flavors in OpenCV:

- **`cv2.HoughLines()`** — the **standard** transform. Each detected line is returned as `(rho, theta)`, the parameters of an *infinite* line in the image plane. You have to compute two far-apart points along that line yourself to draw it.
- **`cv2.HoughLinesP()`** — the **probabilistic** transform. It directly returns line **segments** as `(x1, y1, x2, y2)` — actual endpoints, not an infinite line — and is generally faster and more practical since real edges are finite.

```python
import cv2
import numpy as np

img = cv2.imread("your_lines_image.jpg")  # e.g. a building, a page/document, a road — anything with straight edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# ---- Standard Hough Transform: full (infinite) lines ----
lines_img = img.copy()
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)

if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Extend far beyond the image bounds in both directions along the line
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# ---- Probabilistic Hough Transform: line segments ----
segments_img = img.copy()
segments = cv2.HoughLinesP(
    edges, 1, np.pi / 180, threshold=80,
    minLineLength=50, maxLineGap=10
)

if segments is not None:
    for x1, y1, x2, y2 in segments[:, 0]:
        cv2.line(segments_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Edges", edges)
cv2.imshow("Standard Hough Lines (HoughLines)", lines_img)
cv2.imshow("Probabilistic Hough Lines (HoughLinesP)", segments_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Task

1. Compare the two outputs on the same image. Where does `HoughLines` draw a red line across parts of the image with no real edge (because it always draws the *full* infinite line)? Where does `HoughLinesP` correctly stop at the actual segment boundary?
2. Vary the `threshold` parameter for both functions, and `minLineLength` / `maxLineGap` for the probabilistic version. What happens to short/broken edges as you increase `maxLineGap`?

---

## 7️⃣ Student Task 1 (Evaluative) — Non-ML Surface Crack Detector

### Dataset

**Surface Crack Detection** (Kaggle): https://www.kaggle.com/datasets/arunrk7/surface-crack-detection

- Two folders: `Positive/` (images **with** a crack) and `Negative/` (images **without** a crack).
- 20,000 images per class, each 227×227 RGB, showing patches of concrete surface.

Download it (`kaggle datasets download -d arunrk7/surface-crack-detection`, or manually from the Kaggle page), extract it, and — for speed — work with a **random sample** (e.g. 200–500 images per class) rather than the full 40,000 while you're developing and tuning.

### Task

Build a crack detector using **only classical image processing — no machine learning, no trained classifiers.** Your pipeline should combine tools from this lab and Lab 1:

- Thresholding (fixed, Otsu, or adaptive — try more than one)
- Morphological operations (erosion / dilation / opening / closing) to clean up the thresholded mask
- Some rule you design yourself to turn the cleaned mask into a **binary decision**: "crack" or "no crack" for a given image. Some ideas to consider (you don't have to use these exact ones):
  - The proportion of foreground pixels in the cleaned mask
  - Whether any contour is long and thin (high aspect ratio / low area-to-perimeter ratio), which is characteristic of a crack shape, vs. blob-like noise
  - Edge density (fraction of pixels marked as edges by Canny after cleanup)

**Requirements:**

1. Your final decision function must be a simple thresholded rule on some measurable quantity you compute from the image — not a trained model.
2. Run your detector across your sample set and compare its prediction against the true label (the folder the image came from).
3. Report your results as a **confusion matrix**: True Positives, False Positives, True Negatives, False Negatives. Also report accuracy, precision, and recall computed from these four numbers.
4. **Try at least three different threshold combinations** (e.g. different Otsu vs. fixed threshold values, different morphology kernel sizes, different decision-rule cutoffs) and report the confusion matrix for each. Discuss which combination performed best and why you think that is.

### Starter scaffolding (dataset loading + confusion matrix bookkeeping only)

This gives you the boilerplate for looping through the dataset and tallying results — **the actual detection logic (the `is_crack()` function) is for you to design and is intentionally left unimplemented.**

```python
import os
import random
import cv2

POSITIVE_DIR = "surface-crack-detection/Positive"
NEGATIVE_DIR = "surface-crack-detection/Negative"
SAMPLE_SIZE_PER_CLASS = 300  # start small while tuning, increase later


def is_crack(image_path, **params) -> bool:
    """
    TODO: implement your non-ML detection pipeline here.
    Load the image, threshold it, clean it up with morphology,
    compute some measurable quantity, and return True/False
    based on a threshold you choose.
    """
    raise NotImplementedError


def evaluate(positive_dir, negative_dir, sample_size, **params):
    positive_files = random.sample(os.listdir(positive_dir), sample_size)
    negative_files = random.sample(os.listdir(negative_dir), sample_size)

    tp = fp = tn = fn = 0

    for fname in positive_files:
        predicted = is_crack(os.path.join(positive_dir, fname), **params)
        if predicted:
            tp += 1
        else:
            fn += 1

    for fname in negative_files:
        predicted = is_crack(os.path.join(negative_dir, fname), **params)
        if predicted:
            fp += 1
        else:
            tn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Accuracy={accuracy:.3f}  Precision={precision:.3f}  Recall={recall:.3f}")
    return tp, fp, tn, fn


if __name__ == "__main__":
    evaluate(POSITIVE_DIR, NEGATIVE_DIR, SAMPLE_SIZE_PER_CLASS)
```

---

## 8️⃣ Homography / Perspective Transform

### Theory

A **homography** is a 3×3 matrix that maps points on one plane to points on another plane, capturing the effect of viewing a flat surface from a different angle. In OpenCV terms: given **4 corresponding point pairs** (source points in the input image, destination points in the output image), `cv2.getPerspectiveTransform()` computes the 3×3 transform matrix, and `cv2.warpPerspective()` applies it to remap the whole image.

Typical uses: document/whiteboard scanning ("flatten" a photo taken at an angle), bird's-eye-view transforms for road/sports footage, and image stitching/panoramas.

```python
M = cv2.getPerspectiveTransform(src_points, dst_points)  # src, dst: 4x2 float32 arrays
warped = cv2.warpPerspective(img, M, (output_width, output_height))
```

The four points **must be given in a consistent order** on both sides (e.g. top-left, top-right, bottom-right, bottom-left) — mismatched ordering will produce a flipped or twisted result.

### Sample Problem: Manual "Scan to Paper Size" Tool

This PySide6 app lets you:
1. Load an image.
2. Click **4 corners** on the image, in order (top-left → top-right → bottom-right → bottom-left).
3. Pick a target paper size from a dropdown.
4. Warp the selected quadrilateral so its corners land exactly on the corners of that paper size — producing a flattened, "scanned" look.

The image is shown at its **true pixel size inside a `QScrollArea`** (same pattern from the Lab 1 assignment) so that mouse-click coordinates map 1:1 to image pixel coordinates — no scale-factor correction needed.

```python
import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QComboBox, QMessageBox, QDialog
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

# Paper sizes in pixels, using the standard 72-dpi "points" convention (portrait orientation)
PAPER_SIZES = {
    "A4 (595 x 842)": (595, 842),
    "US Letter (612 x 792)": (612, 792),
    "US Legal (612 x 1008)": (612, 1008),
    "A3 (842 x 1191)": (842, 1191),
    "Square (700 x 700)": (700, 700),
}


class ClickableImageLabel(QLabel):
    """A QLabel that records up to 4 click points, in image-pixel coordinates."""

    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.points = []  # list of (x, y) tuples, up to 4

    def mousePressEvent(self, event):
        if self.pixmap() is None or len(self.points) >= 4:
            return
        pos = event.position().toPoint()
        self.points.append((pos.x(), pos.y()))
        self.parent_window.redraw_with_points()

    def reset_points(self):
        self.points = []


class WarpedResultDialog(QDialog):
    """Shows the warped output inside a PySide6 window with a Save button.

    We deliberately do NOT use cv2.imshow()/cv2.waitKey() here: the cv_gui
    environment uses opencv-python-headless (no bundled GUI/Qt), so calling
    cv2.imshow() there always raises:
        cv2.error: ... The function is not implemented.
        Rebuild the library with Windows, GTK+ 2.x or Cocoa support ...
    Since PySide6 is already our display layer in this environment, the
    result should be shown as a normal PySide6 widget, same as everything
    else in this app.
    """

    def __init__(self, warped_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Warped Result")
        self.warped_bgr = warped_bgr
        self.resize(700, 700)

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.result_label)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setMinimumSize(600, 600)

        rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
        self.display_buffer = np.ascontiguousarray(rgb)  # keep buffer alive
        h, w, ch = self.display_buffer.shape
        qimg = QImage(self.display_buffer.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.result_label.setPixmap(pixmap)
        self.result_label.resize(pixmap.size())

        save_button = QPushButton("Save As...")
        save_button.clicked.connect(self.save_image)

        layout = QVBoxLayout()
        layout.addWidget(self.scroll_area)
        layout.addWidget(save_button)
        self.setLayout(layout)

    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Warped Image", "",
            "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        if file_path:
            cv2.imwrite(file_path, self.warped_bgr)


class HomographyTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lab 2 - Manual Scan-to-Paper-Size Tool")
        self.resize(900, 750)

        self.original_bgr = None
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

        self.reset_button = QPushButton("Reset Points")
        self.reset_button.clicked.connect(self.reset_points)

        self.paper_dropdown = QComboBox()
        self.paper_dropdown.addItems(PAPER_SIZES.keys())

        self.warp_button = QPushButton("Compute & Warp")
        self.warp_button.clicked.connect(self.compute_and_warp)

        self.status_label = QLabel("Open an image, then click 4 corners: TL, TR, BR, BL.")

        controls_row = QHBoxLayout()
        controls_row.addWidget(self.open_button)
        controls_row.addWidget(self.reset_button)
        controls_row.addWidget(self.paper_dropdown)
        controls_row.addWidget(self.warp_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.scroll_area)
        main_layout.addLayout(controls_row)
        main_layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "All Files (*);;Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.gif)"
        )
        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            QMessageBox.warning(self, "Error", "Failed to load image.")
            return

        self.original_bgr = image
        self.image_label.reset_points()
        self.render_image(self.original_bgr)
        self.status_label.setText("Click 4 corners, in order: top-left, top-right, bottom-right, bottom-left.")

    def reset_points(self):
        self.image_label.reset_points()
        if self.original_bgr is not None:
            self.render_image(self.original_bgr)
        self.status_label.setText("Points cleared. Click 4 corners again.")

    def redraw_with_points(self):
        """Draw markers for each clicked point on top of the original image."""
        if self.original_bgr is None:
            return

        preview = self.original_bgr.copy()
        for i, (x, y) in enumerate(self.image_label.points):
            cv2.circle(preview, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(preview, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self.render_image(preview, keep_points=True)
        self.status_label.setText(f"{len(self.image_label.points)}/4 points selected.")

    def compute_and_warp(self):
        if self.original_bgr is None:
            QMessageBox.warning(self, "Error", "Load an image first.")
            return
        if len(self.image_label.points) != 4:
            QMessageBox.warning(self, "Error", "Select exactly 4 points first.")
            return

        target_w, target_h = PAPER_SIZES[self.paper_dropdown.currentText()]

        src_points = np.float32(self.image_label.points)
        dst_points = np.float32([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1],
        ])

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(self.original_bgr, M, (target_w, target_h))

        dialog = WarpedResultDialog(warped, self)
        dialog.exec()

    def render_image(self, cv_image, keep_points=False):
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.display_buffer = np.ascontiguousarray(rgb)
        h, w, ch = self.display_buffer.shape
        bytes_per_line = ch * w
        qimg = QImage(self.display_buffer.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())

        if not keep_points:
            self.image_label.reset_points()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HomographyTool()
    window.show()
    sys.exit(app.exec())
```

> ⚠️ Note the small subtlety in `render_image()`: it resets the click points whenever a **fresh** image is loaded (`keep_points=False`, the default), but preserves them when we're just redrawing the same image with marker overlays after a click (`keep_points=True`). Read through `open_image()`, `reset_points()`, and `redraw_with_points()` to see how each one calls it.

### Try it

1. Load a photo of a document, whiteboard, or any rectangular object taken at an angle.
2. Click its 4 corners in order.
3. Pick a paper size and warp — the result should look like a flat, front-on scan.

---

## 🧪 Student Task 2 (Evaluative) — Automatic Corner Detection

Extend the tool from Section 8 so that, instead of the user manually clicking 4 corners, the **4 corners are detected automatically** from the image using the classical CV pipeline you've built up across this lab:

1. **Threshold** the image to separate the document/object from the background.
2. **Morphology** (opening/closing) to clean up the resulting mask.
3. **Edge detection** (Canny) on the cleaned mask or original image.
4. **Line detection** (Hough — either variant from Section 6) to find the straight edges of the document.
5. **Line intersection → corner detection**: compute the intersection points between the detected lines, and pick out the 4 that best represent the document's corners.

Everything downstream (the paper-size dropdown, `cv2.getPerspectiveTransform`, `cv2.warpPerspective`) should stay the same — you're only replacing **how the 4 source points are obtained**, not what happens after.

**Requirements:**

- Reuse the GUI from Section 8, but replace manual point clicking with an **"Auto-Detect Corners"** button that runs your pipeline and populates the 4 points itself (still show them as markers, same as the manual version, so the user can visually verify before warping).
- **Add a "Custom" option to the paper-size dropdown**, alongside the existing presets (A4, Letter, etc.). When "Custom" is selected, show two input fields (e.g. `QSpinBox`) for the target **width and height in pixels**, and use those values — instead of a `PAPER_SIZES` lookup — as `target_w`/`target_h` when computing the destination points and calling `cv2.warpPerspective()`.
- Your line-intersection step needs a way to go from two lines to a point. If your lines are in `(rho, theta)` form (from `cv2.HoughLines`), you can set up two linear equations (one per line) and solve them as a small system — look into `np.linalg.solve()` for a 2×2 system once you've expressed each line as `x*cos(theta) + y*sin(theta) = rho`.
- Not every pair of detected lines is useful — you'll likely detect far more than 4 lines (duplicates at similar angles, spurious short ones, etc.). Think about how to group or filter lines (e.g. by angle, roughly separating "mostly horizontal" from "mostly vertical" lines) before computing intersections, and how to pick which 4 intersection points are the actual corners (vs. intersections that fall outside the image, or where near-parallel lines produce wildly distant points).
- Test on at least 3 different document/object photos with varying backgrounds and lighting, and note where your automatic detection succeeds and where it fails compared to manual selection.

> 📌 As with the Lab 1 assignment, the reference solution for this task is **not included** in this lab sheet — build it yourself from the pieces above. Solutions will be discussed in the next lab session.

---
