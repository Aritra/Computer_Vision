# 🖥️ Computer Vision Lab 1 — OpenCV Fundamentals & GUI-based Image Viewer

## 🎯 Objective

In this lab, you will:

1. Download and organize the **USC-SIPI standard image database** (our primary dataset for the semester).
2. Learn core **OpenCV** operations: reading, writing, and displaying images in different window modes.
3. Inspect image channels and dimensions, and create images from scratch.
4. Scale and rotate images — including both *clipped* and *unclipped* rotation about the image center.
5. Visualize RGB and HSV channels using a Matplotlib image gallery.
6. Access individual pixels via loops and use this to mirror an image horizontally and vertically.
7. Draw shapes and text on a blank canvas.
8. Get introduced to **PySide6** and build a minimal GUI image viewer.

> ⚠️ **Note on this year's shift:** Unlike previous years, we will **not** be using live webcam feeds for general experiments. All lab work this semester will be done on **static images and videos from standard datasets**, starting with USC-SIPI. This makes results reproducible and comparable across the whole class.

---

## 0️⃣ Dataset: USC-SIPI Image Database

We will use the **USC-SIPI Image Database** (https://sipi.usc.edu/database/) as our primary source of test images for this course. It is organized into **four volumes**, and you need to download **all four**:

| Volume | Description | Link |
|---|---|---|
| **Textures** | Brodatz textures, texture mosaics, etc. | https://sipi.usc.edu/database/database.php?volume=textures |
| **Aerials** | High-altitude aerial images | https://sipi.usc.edu/database/database.php?volume=aerials |
| **Miscellaneous** | Classic test images (mandrill, peppers, boats, etc.) | https://sipi.usc.edu/database/database.php?volume=misc |
| **Sequences** | Moving-head, fly-over, and moving-vehicle frame sequences | https://sipi.usc.edu/database/database.php?volume=sequences |

### 📥 Downloading

1. Visit each volume link above.
2. Use the **"Download the full volume in compressed Gnu tar or Zip format"** link at the top of each page to get the entire volume in one archive (recommended), or download individual images if you only need specific ones.
3. Extract all four volumes into a single folder, e.g.:

   ```
   sipi-dataset/
   ├── textures/
   ├── aerials/
   ├── misc/
   └── sequences/
   ```

4. Keep this folder **outside** your Git repository (add it to `.gitignore`) — do **not** commit the dataset itself, only your code.

### 📌 Important notes about the dataset

- All images are stored in **TIFF (`.tiff`)** format. OpenCV can read TIFFs natively via `cv2.imread()`, no extra setup needed.
- Images are either **8 bits/pixel** (grayscale) or **24 bits/pixel** (color).
- Many filenames are numeric (e.g. `4.2.03.tiff`) — these come from the database's original 1981 cataloguing scheme, not a bug.
- USC-SIPI does not hold copyright on most images; they are provided for **research/educational use only**. Do not redistribute the dataset publicly.

---

## 1️⃣ OpenCV Basics: Reading, Writing, and Displaying Images

### Reading an image

```python
import cv2

img = cv2.imread("sipi-dataset/misc/4.2.07.tiff")  # returns a NumPy array (BGR order)

if img is None:
    raise FileNotFoundError("Could not read the image — check the path.")
```

> 🔎 OpenCV loads images in **BGR** channel order, not RGB. Keep this in mind whenever you use another library (like Matplotlib, which expects RGB).

### Displaying images in different window modes

`cv2.namedWindow()` lets you control how the display window behaves *before* you show anything in it:

```python
# 1. Auto-size window: window shrinks/grows to exactly fit the image, cannot be resized by the user
cv2.namedWindow("AutoSize Window", cv2.WINDOW_AUTOSIZE)
cv2.imshow("AutoSize Window", img)

# 2. Resizable window: user can drag to resize; image is scaled to fit
cv2.namedWindow("Resizable Window", cv2.WINDOW_NORMAL)
cv2.imshow("Resizable Window", img)

# 3. Fixed-size window: force a specific window size regardless of image size
cv2.namedWindow("Fixed Size Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Fixed Size Window", 400, 300)
cv2.imshow("Fixed Size Window", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Writing images to disk

```python
cv2.imwrite("output.png", img)     # OpenCV infers format from the extension
cv2.imwrite("output.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
```

---

## 2️⃣ Grayscale Conversion, Channels, and Dimensions

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Grayscale", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Inspecting image properties

```python
print("Image dtype:", img.dtype)                 # usually uint8
print("Shape (H, W, C):", img.shape)              # color image -> 3 dims
print("Shape (H, W):", gray.shape)                # grayscale image -> 2 dims
print("Height:", img.shape[0], "Width:", img.shape[1])

if img.ndim == 3:
    print("Number of channels:", img.shape[2])
else:
    print("Number of channels: 1 (grayscale)")

print("Total pixels:", img.shape[0] * img.shape[1])
print("Total size in memory (bytes):", img.size)
```

> ⚠️ don't rely on `len(img.shape)` to test "is this a 2-channel image" — a 2D array (`len(img.shape) == 2`) means **grayscale**, not "2 channels". OpenCV images only ever have 1 (grayscale), 3 (BGR), or 4 (BGRA) channels. Use `img.ndim` to check dimensionality and `img.shape[2]` (only when `img.ndim == 3`) to check channel count.

---

## 3️⃣ Creating an Empty Image

You can create a blank canvas directly with NumPy — this is useful whenever you need a scratch image to draw on:

```python
import numpy as np

# Black canvas: 512 rows, 512 cols, 3 color channels, 8-bit per channel
blank_color = np.zeros((512, 512, 3), dtype=np.uint8)

# Black grayscale canvas
blank_gray = np.zeros((512, 512), dtype=np.uint8)

# White canvas
blank_white = np.full((512, 512, 3), 255, dtype=np.uint8)
```

---

## 4️⃣ Scaling and Rotating Images

### Scaling

```python
h, w = img.shape[:2]

# Scale by a fixed factor
scaled_up = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
scaled_down = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# Scale to an exact target size
scaled_fixed = cv2.resize(img, (300, 200), interpolation=cv2.INTER_LINEAR)
```

### Rotation about the image center (hardcoded angle)

Rotation in OpenCV is done with an affine transform built from `cv2.getRotationMatrix2D()`. There are two common ways to handle the output size:

- **Clipped rotation** — output canvas stays the same size as the input, so corners of the rotated image get cut off.
- **Unclipped rotation** — output canvas is enlarged to fit the entire rotated image, so nothing is lost.

```python
angle = 45  # hardcoded rotation angle, in degrees (counter-clockwise)

(h, w) = img.shape[:2]
(cX, cY) = (w // 2, h // 2)   # rotate about the image center

M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)  # 1.0 = no additional scaling

# ---- Clipped rotation: same output dimensions as input ----
rotated_clipped = cv2.warpAffine(img, M, (w, h))

# ---- Unclipped rotation: expand canvas to fit the whole rotated image ----
cos = abs(M[0, 0])
sin = abs(M[0, 1])

new_w = int((h * sin) + (w * cos))
new_h = int((h * cos) + (w * sin))

# Shift the rotation matrix so the image is centered in the new, larger canvas
M[0, 2] += (new_w / 2) - cX
M[1, 2] += (new_h / 2) - cY

rotated_unclipped = cv2.warpAffine(img, M, (new_w, new_h))

cv2.imshow("Clipped Rotation", rotated_clipped)
cv2.imshow("Unclipped Rotation", rotated_unclipped)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 5️⃣ Visualizing RGB and HSV Channels (Matplotlib Gallery)

```python
import cv2
import matplotlib.pyplot as plt

img_bgr = cv2.imread("sipi-dataset/misc/4.2.07.tiff")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

R, G, B = cv2.split(img_rgb)
H, S, V = cv2.split(img_hsv)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Original (RGB)")

axes[0, 1].imshow(R, cmap="Reds")
axes[0, 1].set_title("Red Channel")

axes[0, 2].imshow(G, cmap="Greens")
axes[0, 2].set_title("Green Channel")

axes[0, 3].imshow(B, cmap="Blues")
axes[0, 3].set_title("Blue Channel")

axes[1, 0].imshow(img_hsv)
axes[1, 0].set_title("Original (as HSV array)")

axes[1, 1].imshow(H, cmap="hsv")
axes[1, 1].set_title("Hue")

axes[1, 2].imshow(S, cmap="gray")
axes[1, 2].set_title("Saturation")

axes[1, 3].imshow(V, cmap="gray")
axes[1, 3].set_title("Value")

for ax in axes.ravel():
    ax.axis("off")

plt.tight_layout()
plt.show()
```

---

## 6️⃣ Manual Pixel Access: Mirroring an Image

This section is about understanding how pixel indexing works — **not** about performance. In practice you would just use `cv2.flip()`, but here we do it manually with loops.

```python
import cv2
import numpy as np

img = cv2.imread("sipi-dataset/misc/4.2.07.tiff")
h, w = img.shape[:2]

# ---- Horizontal mirror (flip left-right) ----
mirrored_horizontal = np.zeros_like(img)
for i in range(h):
    for j in range(w):
        mirrored_horizontal[i, j] = img[i, w - 1 - j]

# ---- Vertical mirror (flip top-bottom) ----
mirrored_vertical = np.zeros_like(img)
for i in range(h):
    for j in range(w):
        mirrored_vertical[i, j] = img[h - 1 - i, j]

cv2.imshow("Original", img)
cv2.imshow("Mirrored Horizontal", mirrored_horizontal)
cv2.imshow("Mirrored Vertical", mirrored_vertical)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sanity check against OpenCV's built-in flip:
assert np.array_equal(mirrored_horizontal, cv2.flip(img, 1))
assert np.array_equal(mirrored_vertical, cv2.flip(img, 0))
```

> ⏱️ **Note:** nested Python loops over every pixel are slow (this is the whole point of the exercise — to *see* how pixel access works). For any real application, use `cv2.flip(img, 1)` (horizontal) or `cv2.flip(img, 0)` (vertical) instead.

---

## 7️⃣ Drawing Shapes and Text

```python
import cv2
import numpy as np

canvas = np.zeros((512, 512, 3), dtype=np.uint8)

# Line: start point, end point, color (BGR), thickness
cv2.line(canvas, (50, 50), (450, 50), (0, 255, 0), 3)

# Circle: center, radius, color, thickness (-1 = filled)
cv2.circle(canvas, (256, 256), 100, (255, 0, 0), 2)
cv2.circle(canvas, (256, 256), 20, (0, 0, 255), -1)

# Text: text, bottom-left origin, font, scale, color, thickness
cv2.putText(
    canvas,
    "OpenCV Lab 1",
    (100, 400),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2,
    cv2.LINE_AA,
)

cv2.imshow("Canvas", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 8️⃣ Introducing GUIs: PySide6

So far we've displayed images using OpenCV's own `cv2.imshow()` windows, which are basic and not interactive beyond keypresses. From this lab onward, we'll start building small **GUI applications** using **PySide6** (the official Python bindings for the Qt framework) to make our tools more usable — file pickers, buttons, sliders, etc.

### 🧭 Recommended workflow: prototype logic first, wrap in GUI second

**Do not write and debug new image-processing logic directly inside a PySide6 app.** Work in two stages:

1. **Prototype in your existing environment** (`cv-env` / `cvlab`) using plain scripts with `cv2.imshow()` as usual. Get the actual image-processing logic (scaling, rotation, flipping, filtering, etc.) working and verified correctly first.
2. **Only once that logic is confirmed correct**, move it into a GUI wrapper in a **separate, dedicated environment** (created below) built for GUI work. At that point you're just wiring already-working functions to sliders and buttons, not debugging two things at once.

This keeps your GUI environment stable and avoids conflicts, and keeps your debugging simple — you're never trying to figure out whether a bug is in your image logic or in the GUI plumbing at the same time.

### 📥 Setting up a dedicated `cv_gui` environment

GUI work uses a **separate conda environment** from the rest of the course, kept deliberately minimal:

```bash
conda create -n cv_gui python=3.10
conda activate cv_gui

# Headless OpenCV: no bundled GUI/Qt components, since PySide6 handles all display here
# NumPy pinned below 2.0: current opencv-python-headless wheels are compiled
# against NumPy 1.x, and mixing with NumPy 2.x can crash at import time
pip install "numpy<2" opencv-python-headless opencv-contrib-python-headless

# PySide6 for the GUI itself
pip install PySide6
```

> ⚠️ **Why headless, and why a separate environment?** The regular `opencv-python` package bundles its **own** copy of Qt (used internally for `cv2.imshow()`). If it's installed alongside PySide6 in the same environment, the two Qt installs conflict, and PySide6 fails to start with an error like:
> ```
> Could not find the Qt platform plugin "xcb" in "...cv2/qt/plugins"
> ```
> Keeping GUI work in its own environment with **only** the headless OpenCV build avoids this conflict entirely, and keeps your main `cv-env`/`cvlab` environment (used for `cv2.imshow()`-based experiments) untouched.

> ⚠️ **Why `numpy<2`?** If `pip` pulls in NumPy 2.x on its own, you may see a warning/crash like:
> ```
> A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 as it may crash.
> ```
> This happens when the installed OpenCV wheel was compiled against NumPy 1.x but a newer NumPy 2.x got installed alongside it. Pinning `numpy<2` when you first create the environment avoids this entirely. (If you'd rather use NumPy 2.x, the alternative is to upgrade to a recent OpenCV release — 4.9+ — built against NumPy 2.0; don't mix an old OpenCV wheel with NumPy 2.x.)

Verify the install:

```python
import cv2
import PySide6
print("OpenCV version:", cv2.__version__)
print("PySide6 version:", PySide6.__version__)
```

> 💡 PySide6 windows use their **own event loop** (`app.exec()`), separate from OpenCV's `cv2.waitKey()`. Since `opencv-python-headless` doesn't provide `cv2.imshow()` at all, this isn't a choice you need to make in the `cv_gui` environment — display is handled entirely through PySide6 there.

### 🖼️ A Minimal GUI Image Viewer

This app lets you:
- Pick an image file using a native file-open dialog
- Display it in the window
- Toggle between color and grayscale with a button

```python
import sys
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QWidget
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV Lab 1 - Image Viewer")
        self.resize(720, 640)

        self.image_bgr = None        # original image loaded via OpenCV (BGR)
        self.display_buffer = None   # keeps a reference alive for QImage
        self.is_gray = False

        # --- Widgets ---
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        self.open_button = QPushButton("Open Image")
        self.toggle_button = QPushButton("Toggle Grayscale / Color")
        self.toggle_button.setEnabled(False)

        self.open_button.clicked.connect(self.open_image)
        self.toggle_button.clicked.connect(self.toggle_grayscale)

        # --- Layout ---
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.open_button)
        layout.addWidget(self.toggle_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            self.image_label.setText("Failed to load image.")
            return

        self.image_bgr = image
        self.is_gray = False
        self.toggle_button.setEnabled(True)
        self.render_image(self.image_bgr)

    def toggle_grayscale(self):
        if self.image_bgr is None:
            return

        self.is_gray = not self.is_gray
        if self.is_gray:
            gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)
            self.render_image(gray)
        else:
            self.render_image(self.image_bgr)

    def render_image(self, cv_image):
        """Convert an OpenCV (NumPy) image to QPixmap and show it in the label."""
        if cv_image.ndim == 2:
            # Grayscale image
            self.display_buffer = cv_image.copy()  # keep a live reference
            h, w = self.display_buffer.shape
            bytes_per_line = w
            qimg = QImage(
                self.display_buffer.data, w, h, bytes_per_line,
                QImage.Format_Grayscale8
            )
        else:
            # Color image: convert BGR -> RGB for correct display
            self.display_buffer = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = self.display_buffer.shape
            bytes_per_line = ch * w
            qimg = QImage(
                self.display_buffer.data, w, h, bytes_per_line,
                QImage.Format_RGB888
            )

        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageViewer()
    window.show()
    sys.exit(app.exec())
```

> 🧠 **Why `self.display_buffer`?** `QImage` does not copy the pixel buffer you hand it by default — it just points at the NumPy array's memory. If that array gets garbage-collected while the `QImage`/`QPixmap` is still in use, you'll get corrupted or crashing display output. Keeping a reference on `self` avoids this.

---

## 9️⃣ First Task (Non-Evaluative)

🎯 **Goal:** Pick any two images from **different** SIPI volumes (e.g. one from Textures, one from Miscellaneous) and reproduce the full pipeline from this lab on both: load → inspect shape/channels → grayscale → scaled + rotated (clipped and unclipped) → RGB/HSV gallery → manual mirror → drawing demo. Compare how the operations behave differently on a texture image vs. a natural/miscellaneous image.

This task will not be graded, but you're expected to be able to walk through it in the next lab session.

---

## 🧪 Student Assignment — GUI Image Viewer with Scale, Rotate, and Flip

> 🧭 Follow the workflow from Section 8: first verify your scaling/rotation/flip logic works correctly as a plain script in `cv-env`/`cvlab` (reuse what you built in Section 4). Only once that's confirmed, switch to the `cv_gui` environment (`conda activate cv_gui`) to build the actual GUI below.

Extend the PySide6 image viewer from Section 8 into a small interactive tool with the following features:

1. **Open Image** button — same as before.
2. **Scale slider** — a `QSlider` that scales the displayed image in real time (e.g. from 10% to 200% of its original size). 
3. **Rotate slider** — a `QSlider` that rotates the image about its center in real time, from 0° to 360°. Show the current angle as a label. Use **unclipped rotation** so the full image is always visible.
4. **Flip Horizontal** and **Flip Vertical** buttons — toggle-able flips that combine correctly with the current scale and rotation
5. The image should re-render smoothly as you drag either slider (connect to the slider's `valueChanged` signal).

**Requirements / constraints:**
- Do **not** modify `self.image_bgr` (the originally loaded image) in place — always transform a fresh copy of it each time a control changes, so the transformations don't compound or degrade the image over repeated adjustments.
- Structure your code so there is a single method (e.g. `apply_transforms()`) that reads the current slider values and flip states, and re-renders the image — call this method from every control's signal handler.
- Handle the case where no image has been loaded yet (disable the controls, as in Section 8).
- **Display the image at its true, actual size inside a `QScrollArea`** (with scrollbars appearing automatically when the image is bigger than the viewport), rather than re-fitting the pixmap to a fixed-size label. If you rescale the final pixmap to fit a fixed box (e.g. with `pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio)`), that re-fit silently cancels out your scale slider — a shrunk image just gets stretched back up to fill the box, and an enlarged one gets clipped back down to fit it, so scaling will look broken in one direction. Put your `QLabel` inside a `QScrollArea` (`setWidgetResizable(False)`), set the pixmap on the label at its real computed size, and call `label.resize(pixmap.size())` so the scroll area knows when to show scrollbars.

> 📌 The solution code for this assignment is **not included in this lab sheet**. Attempt it yourself using the building blocks from Sections 4 (scaling/rotation) and 8 (PySide6 viewer) above. Solutions/reference implementations will be discussed in the next lab session.

---
