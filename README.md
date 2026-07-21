# Hand Orientation Generator

<image>
<img src="hand-orientation-generator/inputstereo-outputtranslation.png" alt="Samples"/>
</image>


A pseudo-dataset generation and PoseCNN-style training pipeline for estimating 6D hand pose from stereo images.

This repository provides a research-oriented pipeline for generating pseudo ground-truth labels of hand translation and orientation using stereo vision, hand landmarks, disparity-based 3D reconstruction, and temporal quaternion smoothing. The generated labels are then used to train PoseCNN-style segmentation, translation, and rotation branches.

> This project is still under active development. The generated labels should be treated as pseudo ground truth, not manually verified ground truth.

---

## Overview

The main goal of this project is to build a semi-automatic pipeline for generating hand pose labels from stereo image data. Instead of manually annotating 6D hand pose, the system estimates:

- hand segmentation mask,
- 3D hand landmarks,
- hand translation `[Tx, Ty, Tz]`,
- hand rotation matrix `R`,
- quaternion orientation `[qw, qx, qy, qz]`.

The generated pseudo labels are used to train and evaluate PoseCNN-inspired models for hand pose estimation.

---

## Pipeline

The pseudo-dataset generation pipeline follows these stages:

```text
Stereo side-by-side image
        ↓
Split left-right image
        ↓
Stereo rectification
        ↓
StereoSGBM + WLS disparity estimation
        ↓
MediaPipe hand landmark detection on rectified left image
        ↓
2D landmark → 3D landmark reconstruction using disparity + Q matrix
        ↓
Palm center and palm normal estimation
        ↓
Hand coordinate frame construction
        ↓
Rotation matrix and quaternion generation
        ↓
Temporal quaternion smoothing
        ↓
Pseudo ground-truth CSV + visualization outputs
```

