# 🎨 GAN Framework for WSI Artifact Augmentation

> A unified, modular framework for GAN-based image-to-image translation — designed for introducing realistic Whole Slide Image (WSI) artifacts into clean pathology tiles using 11 state-of-the-art GAN architectures.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square)](https://pytorch.org/)
[![Models](https://img.shields.io/badge/Models-11%20GAN%20Architectures-purple?style=flat-square)](#implemented-models)
[![Task](https://img.shields.io/badge/Task-WSI%20Artifact%20Augmentation-red?style=flat-square)]()

---

## Table of Contents

- [Background](#background)
  - [What is a GAN?](#what-is-a-gan)
  - [What is CycleGAN?](#what-is-cyclegan)
  - [WSI Artifacts](#wsi-artifacts-in-pathology)
  - [GAN-Based Artifact Augmentation](#gan-based-artifact-augmentation-for-wsi)
- [Implemented Models](#implemented-models)
- [Files to Add a New Model](#files-to-add-a-new-model)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [GPU Configuration](#gpu-configuration)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Acknowledgements](#acknowledgements)

---

## Background

### What is a GAN?

A **Generative Adversarial Network (GAN)** consists of two neural networks trained simultaneously in competition:

- **Generator (G)** — learns to produce fake images indistinguishable from real ones
- **Discriminator (D)** — learns to distinguish real images from generated fakes

The generator improves by fooling the discriminator; the discriminator improves by catching fakes. This adversarial dynamic pushes the generator to produce increasingly realistic outputs — without ever needing explicit pixel-level supervision.

### What is CycleGAN?

**CycleGAN** extends the GAN framework to **unpaired image-to-image translation** — enabling style transfer between two domains without any paired training examples. It uses two generators and two discriminators:

```
G_AB : Domain A → Domain B   (e.g. clean WSI tile → artifact tile)
G_BA : Domain B → Domain A   (e.g. artifact tile → clean WSI tile)
```

**Cycle consistency** enforces that translating an image to the other domain and back recovers the original: `G_BA(G_AB(x)) ≈ x`. This prevents the generator from ignoring the input content.

### 📊 Cycle-GAN Framework Results

| Sample 1 | Sample 2 | Sample 3 |
| :---: | :---: | :---: |
| <img src="images/sample_1.png" width="300" height="300" style="object-fit: cover;"> | <img src="images/sample_2.png" width="300" height="300" style="object-fit: cover;"> | <img src="images/sample_3.png" width="300" height="300" style="object-fit: cover;"> |


---

### WSI Artifacts in Pathology

Whole Slide Images are digitised at high resolution (×20–×40 magnification), making them susceptible to technical artifacts during tissue preparation, staining, and scanning:

| Artifact Type | Description | Impact |
|---|---|---|
| **Bubbles / Air pockets** | Air trapped under coverslip | Occludes tissue; misleads models |
| **Colour variation** | Scanner calibration, stain batch differences | Domain shift across cohorts |
| **Blur / Out-of-focus** | Tissue folds or scanner focus errors | Reduces feature quality |
| **Pen marks** | Pathologist annotations on glass | False positive regions |
| **Tissue folding** | Physical creasing creates overlapping layers | Distorted morphology |
| **Dark regions** | Over-staining or thick tissue | Feature suppression |
| **Scratches / Dust** | Glass surface contamination | Spurious patterns |

---

### GAN-Based Artifact Augmentation for WSI

```
Real clean tile  +  Real artifact tile  →  Synthetic tile with realistic artifact
```

Rather than simulating artifacts analytically, we learn the visual statistics of real artifacts from actual WSI data and transfer them to clean images — preserving tissue morphology while introducing realistic artifact patterns with no paired data required.

---

## Implemented Models

| # | Model | Year | Type | Key Feature | Best For |
|---|---|---|---|---|---|
| 1 | **Vanilla GAN** | 2014 | Single-domain | Simple CNN encoder-decoder | Quick baseline |
| 2 | **DCGAN** | 2015 | Single-domain | Strided conv + BatchNorm, no FC | Stable training baseline |
| 3 | **CycleGAN** | 2017 | Cross-domain | ResNet G + cycle + identity loss | Standard unpaired translation |
| 4 | **Pix2Pix** | 2017 | Paired | U-Net G + conditional PatchGAN | Paired clean↔artifact tiles |
| 5 | **StarGAN** | 2018 | Multi-domain | Single G+D for all N domains | Multiple artifact types at once |
| 6 | **Single-Domain GAN** | custom | Single-domain | Internal style variation | Artifact-only dataset |
| 7 | **CUT** | 2020 | Cross-domain | PatchNCE — no reverse G | Faster than CycleGAN |
| 8 | **DiffAugGAN** | 2020 | Cross-domain | Differentiable augmentation | Small datasets (<1k images) |
| 9 | **ResNet GAN** | 2020 | Cross-domain | Pretrained ResNet-50 encoder | Limited data + ImageNet warmstart |
| 10 | **BigGAN** | 2018 | Cross-domain | Self-attention + spectral norm + conditional BN | Large datasets, global coherence |
| 11 | **StyleGAN** | 2020 | Cross-domain | Mapping network + modulated conv + equalized LR | Highest quality production output |

All models share the same CLI interface — switch by changing only `--config`.

---

### Model Selection Guide

```
Small artifact dataset (<1000 images)?
  └─ DiffAugGAN  ← differentiable augmentation stabilises small-data training

Single artifact type, one domain?
  └─ DCGAN  or  Single-Domain GAN  ← self-contained style learning

Multiple artifact types in separate folders?
  └─ StarGAN  ← one model, conditioned on domain label

Unpaired clean + artifact tiles?
  ├─ CUT          ← fastest, good quality, no reverse G needed
  ├─ CycleGAN     ← robust, widely validated
  ├─ ResNet GAN   ← best when training data is limited (ImageNet warmstart)
  └─ BigGAN       ← best when dataset is large and diverse

Paired clean + artifact tiles (same slide)?
  └─ Pix2Pix  ← highest consistency when pairs are available

Maximum visual quality for production?
  └─ StyleGAN  ← modulated convolutions, style injection at every resolution
```

---

## Files to Add a New Model

Four files need to be created or modified. No changes to `train.py`, `test.py`, or `infer.py`.

### Files to CREATE

**1. Model file** — `modules/<model_name>/model.py`

```python
# modules/my_gan/model.py
import torch.nn as nn

class MyGAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.MyGAN
        # Declare generators and discriminators
        # Cross-domain naming: G_AB, G_BA, D_A, D_B
        # Single-domain naming: G, D

    # For cross-domain models used in infer.py --direction A2B / B2A:
    # G_AB must exist for A2B; G_BA must exist for B2A
```

**2. `__init__.py`** — `modules/<model_name>/__init__.py` (empty file)

**3. Config file** — `configs/<model_name>.yaml`

```yaml
# configs/my_gan.yaml
_base_: base.yaml

General:
  model_name: my_gan        # must match key in MODEL_REGISTRY

Dataset:
  domain_a_dir: /path/to/source
  domain_b_dir: /path/to/target   # null for single-domain

Training:
  epochs: 200
  batch_size: 1

MyGAN:                      # model-specific hyperparameters
  generator_features: 64
  discriminator_features: 64
```

### File to MODIFY

**4. Registry** — `modules/__init__.py`

```python
# Add two lines:
from modules.my_gan.model import MyGAN          # ← import
MODEL_REGISTRY["my_gan"]  = MyGAN              # ← register
```

Then train immediately:

```bash
python train.py --config configs/my_gan.yaml \
    --options Dataset.domain_a_dir=/data/source Dataset.domain_b_dir=/data/target
```

---

## Project Structure

```
gan_framework/
│
├── configs/
│   ├── base.yaml               ← Shared defaults (all models inherit this)
│   ├── vanilla_gan.yaml
│   ├── dcgan.yaml              ← NEW
│   ├── cycle_gan.yaml
│   ├── single_domain_gan.yaml
│   ├── pix2pix.yaml
│   ├── cut.yaml
│   ├── stargan.yaml
│   ├── diffaug_gan.yaml
│   ├── resnet_gan.yaml         ← NEW
│   ├── biggan.yaml             ← NEW
│   └── stylegan.yaml           ← NEW
│
├── modules/
│   ├── __init__.py             ← MODEL_REGISTRY (add new models here)
│   ├── vanilla_gan/model.py
│   ├── dcgan/model.py          ← NEW  DCGAN: strided conv, BatchNorm
│   ├── cycle_gan/model.py
│   ├── single_domain_gan/model.py
│   ├── pix2pix/model.py
│   ├── cut/model.py
│   ├── stargan/model.py
│   ├── diffaug_gan/model.py
│   ├── resnet_gan/model.py     ← NEW  ResNet-50 pretrained encoder
│   ├── biggan/model.py         ← NEW  Self-attention + spectral norm + cBN
│   └── stylegan/model.py       ← NEW  Mapping network + modulated conv
│
├── utils/
│   ├── config.py               ← YAML loader + CLI --options override
│   ├── dataset.py              ← Single/Unpaired/Paired/Multi datasets (recursive subfolder)
│   ├── losses.py               ← All loss functions
│   ├── optimizer.py            ← Adam + schedulers
│   └── logger.py               ← Logger, CSV, checkpoints, sample saver
│
├── logs/                       ← Auto-created: logs/<timestamp>_<model>/
├── train.py
├── test.py
├── infer.py
├── environment.yml
└── requirements.txt
```

---

## Environment Setup

#### Using conda (recommended)

```bash
conda env create -f environment.yml
conda activate gan_framework
```

#### Using pip

```bash
pip install -r requirements.txt
```

---

## Configuration

All settings live in `configs/`. Every model YAML **inherits `base.yaml`** and only overrides what it needs.

### Key `base.yaml` fields

```yaml
General:
  seed: 42
  device: 0                    # see GPU Configuration below

Dataset:
  domain_a_dir: /path/to/real_wsi_tiles
  domain_b_dir: /path/to/artifact_tiles  # null for single-domain
  image_size: 256
  # Supports subfolders recursively — reads .png .jpg .jpeg .bmp .tiff .tif .webp

Training:
  epochs: 300
  batch_size: 1
  lr_generator:     2.0e-4
  lr_discriminator: 2.0e-4
  lambda_cycle: 10.0
  lambda_identity: 5.0
  scheduler: linear_decay      # linear_decay | cosine | none
  decay_start_epoch: 100
```

Override any field without editing YAML:

```bash
python train.py --config configs/cycle_gan.yaml \
    --options Training.epochs=100 General.device=2 Training.lambda_cycle=5.0
```

---

## GPU Configuration

`General.device` accepts four formats — set in `base.yaml` or override at runtime.

| Value | Meaning |
|---|---|
| `0` | Single GPU — cuda:0 |
| `2` | Single GPU — cuda:2 |
| `[0, 1]` | DataParallel on GPUs 0 and 1 |
| `[0, 2, 3]` | DataParallel on GPUs 0, 2, 3 |
| `all` | DataParallel on all available GPUs |
| `cpu` or `-1` | CPU |

```bash
# Single GPU
python train.py --config configs/stylegan.yaml --options General.device=0

# Specific GPUs (quote brackets to protect from shell)
python train.py --config configs/biggan.yaml --options "General.device=[0,1]"

# All GPUs
python train.py --config configs/resnet_gan.yaml --options General.device=all

# CPU
python train.py --config configs/dcgan.yaml --options General.device=cpu
```

> **Multi-GPU tip:** GAN training defaults to `batch_size: 1`. Set a larger batch when using DataParallel — e.g. `Training.batch_size=4` for 2 GPUs. `test.py` and `infer.py` auto-scale batch size by GPU count.

---

## Usage

### Training

```bash
# DCGAN — single-domain artifact style learning
python train.py --config configs/dcgan.yaml \
    --options Dataset.domain_a_dir=/data/artifact_tiles General.device=0

# CycleGAN — cross-domain (most common workflow)
python train.py --config configs/cycle_gan.yaml \
    --options Dataset.domain_a_dir=/data/real_tiles \
              Dataset.domain_b_dir=/data/artifact_tiles \
              Training.epochs=300 General.device=0

# ResNet GAN — pretrained backbone, less data needed
python train.py --config configs/resnet_gan.yaml \
    --options Dataset.domain_a_dir=/data/real_tiles \
              Dataset.domain_b_dir=/data/artifact_tiles \
              General.device=1

# BigGAN — large dataset, global spatial coherence
python train.py --config configs/biggan.yaml \
    --options Dataset.domain_a_dir=/data/real_tiles \
              Dataset.domain_b_dir=/data/artifact_tiles \
              Training.batch_size=4 General.device=2

# StyleGAN — highest quality production output
python train.py --config configs/stylegan.yaml \
    --options Dataset.domain_a_dir=/data/real_tiles \
              Dataset.domain_b_dir=/data/artifact_tiles \
              Training.batch_size=4 "General.device=[0,1]"

# CUT — fastest cross-domain training
python train.py --config configs/cut.yaml \
    --options Dataset.domain_a_dir=/data/real_tiles \
              Dataset.domain_b_dir=/data/artifact_tiles

# StarGAN — all artifact types in one model
python train.py --config configs/stargan.yaml \
    --options Dataset.domain_a_dir=/data/multi_artifact_root \
              StarGAN.num_domains=3

# DiffAugGAN — small dataset
python train.py --config configs/diffaug_gan.yaml \
    --options Dataset.domain_a_dir=/data/real_tiles \
              Dataset.domain_b_dir=/data/few_artifact_tiles

# Pix2Pix — paired tiles
python train.py --config configs/pix2pix.yaml \
    --options Dataset.domain_a_dir=/data/real_tiles \
              Dataset.domain_b_dir=/data/paired_artifacts
```

---

### Evaluation

```bash
# Standard evaluation + FID score
python test.py --config configs/stylegan.yaml \
    --checkpoint logs/.../checkpoints/latest.pth \
    --fid \
    --options Dataset.domain_a_dir=/data/test_real \
              Dataset.domain_b_dir=/data/test_artifacts \
              General.device=0

# Multi-GPU evaluation
python test.py --config configs/biggan.yaml \
    --checkpoint logs/.../checkpoints/latest.pth \
    --options "General.device=[0,1]"
```

---

### Inference

```bash
# Add artifacts to clean tiles — A2B
python infer.py --config configs/stylegan.yaml \
    --checkpoint logs/.../checkpoints/latest.pth \
    --input_dir /data/clean_tiles \
    --output_dir /data/generated_artifacts \
    --num_images 5000 \
    --direction A2B \
    --options General.device=0

# Restore clean tiles from artifacts — B2A
python infer.py --config configs/resnet_gan.yaml \
    --checkpoint logs/.../checkpoints/latest.pth \
    --input_dir /data/artifact_tiles \
    --output_dir /data/restored_tiles \
    --direction B2A \
    --options General.device=0

# Maximum throughput on all GPUs
python infer.py --config configs/cycle_gan.yaml \
    --checkpoint logs/.../checkpoints/latest.pth \
    --input_dir /data/clean_tiles \
    --output_dir /data/artifacts \
    --options General.device=all
```

---

## Acknowledgements

- [DCGAN](https://arxiv.org/abs/1511.06434) — Radford et al., 2015
- [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) — Zhu et al., 2017
- [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) — Isola et al., 2017
- [BigGAN](https://arxiv.org/abs/1809.11096) — Brock et al., 2018
- [StarGAN](https://github.com/yunjey/stargan) — Choi et al., 2018
- [CUT](https://github.com/taesungp/contrastive-unpaired-translation) — Park et al., 2020
- [DiffAugGAN](https://github.com/mit-han-lab/data-efficient-gans) — Zhao et al., 2020
- [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch) — Karras et al., 2020

---

<p align="center">
  Built for realistic WSI artifact augmentation in computational pathology.
</p>
