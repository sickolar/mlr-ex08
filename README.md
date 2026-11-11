# Exercise 8 — Leaf Classifier on Phone (`mlr-ex08`)

A modern browser-based leaf species classifier powered by **PyTorch → ONNX** and **ONNX Runtime Web**, running 100% locally on your phone or desktop browser.

## Live Web App (GitHub Pages)

- **Public App URL:** [https://sickolar.github.io/mlr-ex08/](https://sickolar.github.io/mlr-ex08/)

## Features

- **Live Rear Camera Classification:** Classifies `apple`, `cherry`, `chestnut`, and `maple` leaves in real time using the phone's rear camera (`facingMode: "environment"`), compatible with Safari on iPhone and Chrome on Android.
- **Exact Preprocessing Pipeline:** Central square crop → resize to `96 × 96` → scale RGB to `[0, 1]` (`/ 255`) → per-channel standardization with ImageNet `mean = [0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]` → `1 × 3 × 96 × 96` NCHW Float32 tensor.
- **Photo Upload & Notebook Verification:** Upload any leaf photo as a fallback or verify `check_image.jpg` directly against `check_expected.json`.
- **100% On-Device Privacy:** No camera frames or uploaded photos ever leave the device.
- **Local QR Code:** Scannable SVG QR code generated locally pointing to `https://sickolar.github.io/mlr-ex08/`.

## Repository Files

- `index.html` — Single-file modern web application (pastel brown & white theme)
- `leaf_model.onnx` — Exported MobileNetV3-Small ONNX model with embedded metadata
- `model_info.json` — Model class names, input resolution (`96`), and normalization constants
- `check_image.jpg` — Validation sample image from the notebook
- `check_expected.json` — Expected class probabilities from PyTorch in the notebook
- `REVIEW.md` — Verification review of the deployed web app
- `REPORT.md` — Task 4 experimental report on classifier improvements
