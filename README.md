# 📸 Computer Vision Lab Repository – BITS F459

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()

This repository contains **demo codes, lab sheets, and experiments** used during lectures and labs for the **Computer Vision** course *(Subject Code: BITS F459)* at BITS Pilani Hyderabad campus.

> 📌 A new **labsheet** will be added here every week. Refer to this repository regularly for in-class coding demos, lab assignments, and updates.

---

## 📅 Weekly Labsheets

Labsheets are added weekly as Markdown files under the [`labsheets/`](./labsheets) folder, following the naming convention `labsheetN.md`.

| Week | Labsheet | Topic | Status |
|:----:|----------|-------|:------:|
| 1    | [labsheet1.md](./labsheets/labsheet1.md) | Introduction to openCV and basics | Uploaded |
| 2    | [labsheet2.md](./labsheets/labsheet2.md) | Edge detection, line detection, morphology, homography | Uploaded |

> The table above will be updated as new labsheets are released. Check back weekly, or watch/star this repo to get notified of updates.

### 📌 How to use a labsheet

1. Pull the latest changes: `git pull origin main`
2. Open the corresponding `labsheetN.md` file for the week
3. Follow the instructions and complete the tasks in your local `cv-env` environment
4. Save your work (code, notebooks, outputs) as instructed in the labsheet

---

## 🖥️ System Requirement Notice

All students **must ensure** their system has an **Ubuntu (Linux)** partition.

- 💾 **Minimum space required**: 50 GB
- 🐧 Native Ubuntu is recommended, but you may alternatively use:
  - **WSL2** (on Windows 10/11)
  - **VirtualBox** or **VMware** with Ubuntu

Ubuntu ensures smoother compatibility with vision toolchains and dependencies.

---

## 🐍 Python Environment Setup (Miniconda)

We'll use **Miniconda** to manage the Python environment for all demos and labsheets.

### 📥 Step 1: Install Miniconda

- Download: https://docs.conda.io/en/latest/miniconda.html
- Install it using default options
- After installation, restart your terminal and verify:

```bash
conda --version
```

---

### 🛠️ Step 2: Create Environment (Recommended via YAML)

Clone this repository and run:

```bash
git clone <this-repo-url>
cd <this-repo-folder>
conda env create -f environment.yml
conda activate cv-env
```

Alternatively, create and install manually:

```bash
conda create -n cv-env python=3.10
conda activate cv-env
pip install opencv-python opencv-contrib-python numpy matplotlib open3d typing_extensions
```

You may add necessary libraries later to your environment using `pip`.

---

### ✅ Step 3: Verify Setup

Run this minimal test script:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("OpenCV version:", cv2.__version__)
```

If this runs without errors and prints a version number, your environment is ready.

---

## 🧑‍💻 Recommended Code Editor: VS Code

We recommend using **Visual Studio Code** for writing and debugging code.

* Download: [https://code.visualstudio.com/](https://code.visualstudio.com/)
* Install the **Python** extension from the marketplace
* Install the **Jupyter** extension if labsheets include notebook-based tasks
* Configure the integrated terminal to use the `cv-env` Conda environment

---

## 📦 Sample Environment File (`environment.yml`)

```yaml
name: cv-env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - matplotlib
  - pip
  - pip:
      - opencv-python
      - opencv-contrib-python
      - open3d
      - typing_extensions
```

---

## 📂 Repository Structure

```
.
├── README.md
├── LICENSE
├── environment.yml
├── labsheets/
│   ├── labsheet1.md
│   ├── labsheet2.md
│   └── ...
└── demos/
    └── (in-class demo code, organized by week/topic)
```

---

## 🔄 Staying Up to Date

Since new labsheets are pushed weekly, make sure to sync your local copy regularly before each lab session:

```bash
git pull origin main
```

If you've made local changes and hit a merge conflict, consider working in your own fork or branch rather than committing directly to `main`.

---

## 🙋 Getting Help

- **Course-related doubts**: Reach out during lab hours or via the course's official communication channel (Teams/Slack/Email, as announced in class).
- **Repo/setup issues**: Open a GitHub [Issue](../../issues) describing your problem, including your OS, Python version, and the error message/traceback.

---

## 📄 License

This repository is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for full details.
