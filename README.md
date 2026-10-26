# Manifold-Stereo

Code for **Manifold-Guided Stereo-Monocular Refinement for Endoscopic Stereo Disparity Estimation** (NeurIPS 2026).

The implementation retains the original `EndoStereoFM` class and configuration names.

**[Paper (coming soon)]**  |  **[Model Zoo (coming soon)]**  |  **[Dataset](#数据集准备)**

**Manifold-Stereo** combines monocular depth priors ([Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)), cost-volume-conditioned latent residual flow, stereo-scale alignment, and geometric consistency gating for endoscopic stereo disparity estimation. The latent flow refines a monocular disparity candidate before its contribution is regulated during recurrent stereo refinement.

---

## Project Structure

```
Manifold-Stereo/
├── config/
│   └── train_scared.yaml          # Main training configuration
├── core/
│   ├── endo_stereo_fm.py          # Main model (EndoStereoFM class)
│   ├── stereo_datasets.py          # Dataset classes (SCARED, Hamlyn, SERV-CT)
│   ├── geo_encoder.py             # Geometric encoding volume
│   ├── depth_refine.py            # REMP refinement module
│   ├── feat_update.py             # GRU update blocks
│   ├── disp_warp.py              # Disparity warping
│   ├── stereo_submodule.py        # Basic building blocks (BasicConv, etc.)
│   └── utils/
│       ├── augmentor.py           # Data augmentation (FlowAugmentor)
│       ├── frame_utils.py         # I/O utilities (disparity readers, PFM, etc.)
│       ├── geom_utils.py          # Geometric helpers (disp2depth, etc.)
│       ├── file_io.py             # File I/O helpers
│       └── utils.py               # Misc utilities (InputPadder, etc.)
├── Depth-Anything-V2-list3/        # Depth Anything V2 submodule
│   └── depth_anything_v2/
│       ├── dpt.py                 # DPT decoder
│       └── dinov2_layers/        # DINOv2 building blocks
├── train_scared.py                # Training entry point
├── evaluate_scared.py             # SCARED evaluation
├── evaluate_hamlyn.py             # Hamlyn evaluation
├── evaluate_servct.py             # SERV-CT evaluation
├── vis_scared.py                  # Visualization scared hamlyn servct script()
├── inference_debug.py            # Debug/inference utility
└── model_stats.py                 # Model statistics
```

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Preparation](#数据集准备)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)

---

## Overview

Traditional stereo matching struggles in surgical environments due to:

- **Weakly textured tissues** with low visual features
- **Specular reflections** and blood occlusion
- **Large depth variance** across surgical phases

Manifold-Stereo addresses these by fusing a monocular depth prior with geometric stereo features through:

1. **Monocular Depth Encoder** — frozen Depth Anything V2 providing strong semantic priors
2. **Geometric Encoding Volume** — GWC (Group-wise Correlation Volume) + Hourglass 3D CNN
3. **Conditional Latent Residual Flow** — ODE-based latent refinement that decodes a correction to the monocular disparity candidate
4. **Geometric Consistency Gating** — spatial contribution weights based on left-right feature alignment, applied after stereo-scale alignment of the candidate

---

## Installation

### Requirements

- Python 3.8
- CUDA >= 11.0
- NVIDIA GPU with >= 16GB VRAM (for training)

### Environment Setup

```bash
# Create conda environment
conda create -n endo_stereo_fm python=3.8
conda activate endo_stereo_fm

# Install PyTorch
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install tqdm scipy opencv-python scikit-image tensorboard matplotlib timm==0.6.13
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install accelerate==1.0.1 hydra-core omegaconf

# Optional: for higher-precision ODE solving in Flow Matching
pip install torchdiffeq
```

### Pretrained Weights

1. Download **Depth Anything V2** pretrained weights:
  ```bash
   # Place the weight file at the project root
   # e.g., F:/depth_anything_v2_vitl.pth
  ```
   Available variants: [vits](https://huggingface.co/spaces/depthanything/Depth-Anything-V2/blob/main/depth_anything_v2_vits.pth), [vitb](https://huggingface.co/spaces/depthanything/Depth-Anything-V2/blob/main/depth_anything_v2_vitb.pth), [vitl](https://huggingface.co/spaces/depthanything/Depth-Anything-V2/blob/main/depth_anything_v2_vitl.pth), [vitg](https://huggingface.co/spaces/depthanything/Depth-Anything-V2/blob/main/depth_anything_v2_vitg.pth)
2. Update the weight path in `core/endo_stereo_fm.py` (line ~498) or set it in your config.

---

## 数据集准备

### SCARED (Training + Evaluation)

Download from the [SCARED dataset page](https://endovissub2019-scared.grand-challenge.org/) and organize as follows:

```
${SCARED_ROOT}/
├── train_data/
│   ├── dataset_1/
│   │   └── keyframe_1/
│   │       └── data/
│   │           ├── left_rectified/   # 000001.png, 000002.png, ...
│   │           ├── right_rectified/
│   │           └── disparity/         # 000001.png, 000002.png, ...  (16-bit, / 64.0)
│   └── dataset_2/
│       └── ...
├── test_data/
│   └── ...
└── splits/
    └── endovis/
        ├── train_files.txt    # "dataset_1/keyframe_1 0"
        ├── val_files.txt
        └── test_files.txt
```

Configure in `config/train_scared.yaml`:

```yaml
scared_root: '/path/to/endovis_data'
scared_splits_dir: '/path/to/splits/endovis'
```

### Hamlyn (Evaluation)

Download from the [Hamlyn Dataset](http://hamlyn.doc.ic.ac.uk/vision/) and organize as follows:

```
${HAMLYN_ROOT}/
├── calibration/
│   ├── 01/
│   │   ├── intrinsics.txt     # fx fy cx cy
│   │   └── extrinsics.txt     # 3x4 RT matrix
│   ├── 02/
│   └── ...
├── rectified01/
│   ├── image01/               # left images
│   ├── image02/               # right images
│   └── disparity01/           # left disparity PNG (16-bit, / 256.0)
├── rectified02/
└── ...
```

### SERV-CT (Evaluation)

Download from the [SERV-CT dataset](https://www.sciencedirect.com/science/article/abs/pii/S1361841520301641) and organize as follows:

```
${SERVCT_ROOT}/
├── Left_rectified/             # 001.png, 002.png, ...
├── Right_rectified/
├── Disparity/                  # 001.png, 002.png, ...  (16-bit, / 256.0)
└── Rectified_calibration/
    ├── 001.json                # {"Q": {"data": [...], "rows": 4, "cols": 4}}
    ├── 002.json
    └── ...
```

---

## Training

```bash
# Single-node multi-GPU training
CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_scared.py

# With custom config
CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_scared.py --config-dir ./config --config-name train_scared.yaml
```

Training outputs (checkpoints, logs) are saved to `./checkpoints/`.

**Training loss components:**

- Stereo disparity regression loss
- Flow matching latent loss (optional)
- Geodesic consistency loss (optional)
- Uncertainty calibration loss (optional)

---

## Evaluation

```bash
# SCARED
python evaluate_scared.py

# Hamlyn
python evaluate_hamlyn.py

# SERV-CT
python evaluate_servct.py
```

**Metrics computed:**


| Level         | Metrics                                             |
| ------------- | --------------------------------------------------- |
| **Disparity** | MAE, D1-ratio (1px, 2px, 3px, 5px threshold), Bad-X |
| **Depth**     | Abs Rel, Sq Rel, RMSE, RMSE(log), δ1/δ2/δ3          |


---

## Acknowledgements

This project builds upon several excellent works:

- **[MonSter](https://github.com/Junda24/MonSter)** — Monocular Stereo Matching
- **[RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo)** — Recurrent All-Pairs Field Transforms for Stereo
- **[IGEV](https://github.com/gangweiX/IGEV)** — Iterative Geometry Encoding Volume
- **[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)** — Monocular Depth Estimation

