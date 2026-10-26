<h1 align="center">Manifold-Stereo</h1>

<p align="center">
  <strong>Manifold-Guided Stereo-Monocular Refinement<br>for Endoscopic Stereo Disparity Estimation</strong>
</p>

<p align="center">NeurIPS 2026 · Official Implementation</p>

<p align="center">
  <a href="#method">Method</a> ·
  <a href="#setup">Setup</a> ·
  <a href="#training">Training</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#citation">Citation</a>
</p>

Manifold-Stereo combines calibrated stereo geometry with monocular depth priors for endoscopic disparity estimation. It refines a monocular candidate in a **cost-volume-induced local correction space**, then controls its contribution through **geometric consistency-gated recurrent refinement**.

## Method

![Figure 1: Cost-volume-guided local correction of a monocular disparity candidate.](assets/figures/figure1_motivation.png)

**Figure 1 — Motivation.** Local latent corrections use stereo evidence to refine a monocular candidate in regions with ambiguous correspondence. [Original PDF](assets/figures/figure1_motivation.pdf)

![Figure 2: The complete Manifold-Stereo pipeline, from stereo and monocular candidates to reliability-gated recurrent updates.](assets/figures/figure2_overview.png)

**Figure 2 — Framework.** Stereo and monocular candidate construction, conditional latent residual flow, and reliability-gated recurrent refinement. [Original PDF](assets/figures/figure2_overview.pdf)

- **Local correction:** an image-aligned latent field encodes stereo cost-volume geometry.
- **Residual refinement:** a conditional latent trajectory decodes corrections to the monocular candidate.
- **Reliability gating:** stereo-scale alignment and geometric consistency regulate the candidate during recurrent updates.

The model is implemented in [core/endo_stereo_fm.py](core/endo_stereo_fm.py). Class and configuration names retain the original `EndoStereoFM` / `flow_matching` identifiers; the method uses a conditional latent residual flow.

## Setup

Run the following commands from a machine with an NVIDIA GPU and a driver compatible with CUDA 11.8:

```bash
git clone https://github.com/Jiewen-Yang/Manifold-Stereo.git
cd Manifold-Stereo

conda create -n manifold-stereo python=3.10 -y
conda activate manifold-stereo
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install "numpy<2" "opencv-python<4.12" scipy scikit-image pillow imageio \
    matplotlib tqdm opt_einsum lz4 pyyaml hydra-core omegaconf accelerate==1.0.1 wandb torchdiffeq
```

### Prepare weights and data

1. Download the **Depth Anything V2 ViT-L** weights from the [upstream repository](https://github.com/DepthAnything/Depth-Anything-V2#pre-trained-models). In [core/endo_stereo_fm.py](core/endo_stereo_fm.py), replace the `F:/depth_anything_v2_{args.encoder}.pth` path with your local checkpoint path. The default configuration uses `encoder: vitl`.
2. Prepare rectified **SCARED** image pairs and disparity maps. The dataset readers for SCARED, Hamlyn, and SERV-CT are in [core/stereo_datasets.py](core/stereo_datasets.py); SCARED split lists are included in [splits/endovis/](splits/endovis/).
3. Update [config/train_scared.yaml](config/train_scared.yaml):

```yaml
scared_root: /path/to/SCARED
scared_splits_dir: ./splits/endovis
save_path: ./checkpoints/scared/
```

<details>
<summary>SCARED directory layout</summary>

```text
SCARED/
├── train_data/                       # dataset_1 ... dataset_7
│   └── dataset_1/keyframe_1/data/
│       ├── left_rectified/000001.png
│       ├── right_rectified/000001.png
│       └── disparity/000001.png
└── test_data/                        # dataset_8 and later
    └── dataset_8/keyframe_1/data/
        ├── left_rectified/000001.png
        ├── right_rectified/000001.png
        └── disparity/000001.png
```

Split entries use names such as `dataset1/keyframe1 1 l`; the loader maps these to `dataset_1/keyframe_1`. Disparity PNGs are decoded with a scale factor of 1/64.

</details>

Model checkpoints and datasets are not included in this repository. The Depth Anything V2 weights are required when initializing the model for both training and evaluation.

## Training

After configuring the local paths above, launch training from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline accelerate launch --multi_gpu --num_processes 2 \
    train_scared.py --config-name train_scared
```

Checkpoints and validation logs are written to `save_path`. Adjust GPU count, batch size, and iteration settings for your environment; remove `WANDB_MODE=offline` to use online Weights & Biases logging.

Two ablation configurations are included:

| Ablation | Hydra configuration name |
| --- | --- |
| Without geometric gating | `train_scared_exp01_no_geo_gating` |
| Without latent residual-flow correction | `train_scared_exp02_no_flow_matching` |

Select an ablation by changing `--config-name` in the training command.

## Evaluation

Use a trained Manifold-Stereo checkpoint and supply the dataset paths explicitly. Run from the repository root:

```bash
# SCARED
python evaluate_scared.py --model_path /path/to/model.pth \
    --data_root /path/to/SCARED --split_file ./splits/endovis/test_files.txt

# Hamlyn
python evaluate_hamlyn.py --model_path /path/to/model.pth \
    --data_root /path/to/Hamlyn

# SERV-CT
python evaluate_servct.py --model_path /path/to/model.pth \
    --data_root /path/to/SERV-CT
```

All three scripts accept `--config_path` and `--iters` (default: 32). Use the configuration that matches your checkpoint. See [vis_scared.py](vis_scared.py) for the visualization utility.

## Citation

```bibtex
@inproceedings{yang2026manifoldstereo,
  title     = {Manifold-Guided Stereo-Monocular Refinement for Endoscopic Stereo Disparity Estimation},
  author    = {Yang, Jiewen and Zhong, Yi and Pu, Bin and Dong, Xingbo and Zheng, Ming and Lin, Qika and He, Kai and Zhang, Xuechen},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements and license

We thank [MonSter](https://github.com/Junda24/MonSter), [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), [IGEV](https://github.com/gangweiX/IGEV), and [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for their open research and code.

See [LICENSE](LICENSE) for the included MonSter Model License; upstream components retain their respective licenses.

> MonSter model is licensed under the MonSter Model License, Copyright © Autel Robotics Co., Ltd. and Huazhong University of Science and Technology. All Rights Reserved.
