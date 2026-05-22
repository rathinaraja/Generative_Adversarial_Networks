# 🎨 GAN Framework for WSI Artifact Augmentation

> A unified, modular framework for GAN-based image-to-image translation — designed for introducing realistic Whole Slide Image (WSI) artifacts into clean pathology tiles using classical GAN, single-domain, and cross-domain cycle-consistent models.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square)](https://pytorch.org/)
[![Models](https://img.shields.io/badge/Models-7%20GAN%20Architectures-purple?style=flat-square)](#implemented-models)
[![Task](https://img.shields.io/badge/Task-WSI%20Artifact%20Augmentation-red?style=flat-square)]()

---

## Table of Contents

- [Background](#background)
  - [What is a GAN?](#what-is-a-gan)
  - [What is CycleGAN?](#what-is-cyclegan)
  - [WSI Artifacts](#wsi-artifacts-in-pathology)
  - [GAN-Based Artifact Augmentation](#gan-based-artifact-augmentation-for-wsi)
- [Implemented Models](#implemented-models)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Adding a New Model](#adding-a-new-model)
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

---

### WSI Artifacts in Pathology

Whole Slide Images are digitised at very high resolution (×20–×40 magnification), making them susceptible to a range of **technical artifacts** introduced during tissue preparation, staining, and scanning:

| Artifact Type | Description | Impact |
|---|---|---|
| **Bubbles / Air pockets** | Air trapped under coverslip creates circular clear regions | Occludes tissue; misleads models |
| **Colour variation** | Scanner calibration, stain batch, and tissue thickness differences | Domain shift across cohorts |
| **Blur / Out-of-focus** | Tissue folds or scanner focus errors | Reduces feature quality |
| **Pen marks** | Pathologist annotations on the glass slide | False positive regions |
| **Tissue folding** | Physical creasing creates overlapping layers | Distorted morphology |
| **Dark regions** | Over-staining or thick tissue | Feature suppression |
| **Scratches / Dust** | Glass surface contamination | Spurious patterns |

These artifacts occur unpredictably in real clinical WSI collections. Deep learning models trained on clean data fail silently when deployed on artifact-containing slides — a major robustness problem.

---

### GAN-Based Artifact Augmentation for WSI

The key insight of this framework is to **use real artifact images as style teachers**:

```
Real clean tile  +  Real artifact tile  →  Synthetic tile with realistic artifact
```

Rather than simulating artifacts analytically (which looks artificial), we learn the visual statistics of real artifacts from actual WSI data and transfer them to clean images. This produces augmented training data that:

- Preserves the tissue morphology of the clean source image
- Introduces visually realistic, data-driven artifact patterns
- Covers the full distribution of artifact intensities and types
- Requires no paired data (clean tile + same tile with artifact)

The framework supports four strategies depending on your data availability:

| Strategy | Model | Data Required | Use Case |
|---|---|---|---|
| **Single-domain** | `single_domain_gan` | Only artifact tiles | Learn internal variation from artifacts alone |
| **Cross-domain (unpaired)** | `cycle_gan`, `cut`, `diffaug_gan` | Clean tiles + artifact tiles (unpaired) | Transfer artifact style onto clean tiles |
| **Cross-domain (paired)** | `pix2pix` | Matched (clean, artifact) tile pairs | Highest quality when pairs are available |
| **Multi-domain** | `stargan` | N groups of artifact-type folders | One model for all artifact types |

---

## Implemented Models

| # | Model | Year | Type | Key Feature |
|---|---|---|---|---|
| 1 | **Vanilla GAN** | 2014 | Single-domain | Baseline CNN encoder-decoder generator |
| 2 | **CycleGAN** | 2017 | Cross-domain | ResNet generator + cycle + identity loss |
| 3 | **Single-Domain GAN** | custom | Single-domain | Internal style variation within one domain |
| 4 | **Pix2Pix** | 2017 | Paired cross-domain | U-Net generator + conditional PatchGAN |
| 5 | **CUT** | 2020 | Cross-domain | PatchNCE contrastive loss — no reverse G needed |
| 6 | **StarGAN** | 2018 | Multi-domain | Single G+D pair handles all N artifact domains |
| 7 | **DiffAugGAN** | 2020 | Cross-domain | Differentiable augmentation — best for small datasets |

Switch between all models by changing only the `--config` argument — no code changes needed.

---

## Project Structure

```
gan_framework/
│
├── configs/
│   ├── base.yaml               ← Shared defaults (seed, device, augmentation, training)
│   ├── vanilla_gan.yaml
│   ├── cycle_gan.yaml          ← ResNet CycleGAN (cross-domain)
│   ├── single_domain_gan.yaml  ← Internal style variation
│   ├── pix2pix.yaml            ← Paired translation (U-Net)
│   ├── cut.yaml                ← Contrastive Unpaired Translation
│   ├── stargan.yaml            ← Multi-domain
│   └── diffaug_gan.yaml        ← Limited data (<1k images)
│
├── modules/
│   ├── __init__.py             ← MODEL_REGISTRY + get_model(cfg)
│   ├── vanilla_gan/model.py
│   ├── cycle_gan/model.py      ← ResNet generator + PatchGAN (from your scripts)
│   ├── single_domain_gan/model.py
│   ├── pix2pix/model.py        ← U-Net generator + conditional discriminator
│   ├── cut/model.py            ← PatchNCE feature projection heads
│   ├── stargan/model.py        ← Multi-domain generator with domain label input
│   └── diffaug_gan/model.py    ← Differentiable augmentation wrapper
│
├── utils/
│   ├── config.py               ← YAML loader with _base_ inheritance + CLI override
│   ├── dataset.py              ← Single/Unpaired/Paired/MultiDomain datasets + ImagePool
│   ├── losses.py               ← LSGAN, VanillaGAN, Hinge, cycle, identity, PatchNCE, DiffAug
│   ├── optimizer.py            ← Adam + linear decay / cosine scheduler
│   └── logger.py               ← Console logger, CSVLogger, CheckpointManager, sample saver
│
├── logs/                       ← Auto-created: logs/<timestamp>_<model>/
│   └── 20240601_cycle_gan/
│       ├── checkpoints/latest.pth
│       ├── samples/epoch_0010/
│       ├── metrics.csv
│       └── cycle_gan.yaml      ← Config snapshot
│
├── train.py                    ← Training entry point
├── test.py                     ← Evaluation + optional FID
├── infer.py                    ← Batch artifact generation
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

All settings live in `configs/`. Each model YAML **inherits from `base.yaml`** and only declares overrides.

### Key `base.yaml` fields

```yaml
General:
  seed: 42
  device: 0                    # GPU index; -1 = CPU

Dataset:
  domain_a_dir: /path/to/real_wsi_tiles      # clean source domain
  domain_b_dir: /path/to/artifact_tiles      # artifact target domain (null for single-domain)
  image_size: 256

Training:
  epochs: 300
  batch_size: 1
  lr_generator:     2.0e-4
  lr_discriminator: 2.0e-4
  lambda_cycle: 10.0           # cycle consistency loss weight
  lambda_identity: 5.0         # identity loss weight
  scheduler: linear_decay      # linear_decay | cosine | none
  decay_start_epoch: 100       # hold LR until this epoch, then decay to 0
```

Override any field at the command line without editing YAML:

```bash
python train.py --config configs/cycle_gan.yaml \
    --options Training.epochs=100 General.device=2 Training.lambda_cycle=5.0
```

---

## Usage

### Training

```bash
# CycleGAN — cross-domain artifact transfer (most common)
python train.py --config configs/cycle_gan.yaml \
    --options Dataset.domain_a_dir=/data/real_tiles \
              Dataset.domain_b_dir=/data/artifact_tiles \
              Training.epochs=300 General.device=0

# Single-domain — learn style variation from artifact tiles alone
python train.py --config configs/single_domain_gan.yaml \
    --options Dataset.domain_a_dir=/data/artifact_tiles General.device=1

# CUT — faster than CycleGAN, trains in ~half the time
python train.py --config configs/cut.yaml \
    --options Dataset.domain_a_dir=/data/real_tiles \
              Dataset.domain_b_dir=/data/artifact_tiles

# Pix2Pix — highest quality when paired tiles are available
python train.py --config configs/pix2pix.yaml \
    --options Dataset.domain_a_dir=/data/real_tiles \
              Dataset.domain_b_dir=/data/paired_artifact_tiles

# StarGAN — one model for bubble + colour + blur artifacts
python train.py --config configs/stargan.yaml \
    --options Dataset.domain_a_dir=/data/multi_artifact_root \
              StarGAN.num_domains=3

# DiffAugGAN — small dataset (<1000 artifact images)?
python train.py --config configs/diffaug_gan.yaml \
    --options Dataset.domain_a_dir=/data/real_tiles \
              Dataset.domain_b_dir=/data/few_artifacts
```

Training outputs are auto-saved to `logs/<timestamp>_<model>/`:

```
logs/20240601_120000_cycle_gan/
├── checkpoints/latest.pth
├── samples/epoch_0010/real_A.png  fake_B.png  cycled_A.png  ...
└── metrics.csv
```

---

### Evaluation

```bash
# Generate test outputs and compute FID score
python test.py --config configs/cycle_gan.yaml \
    --checkpoint logs/.../checkpoints/latest.pth \
    --fid \
    --options Dataset.domain_a_dir=/data/test_real \
              Dataset.domain_b_dir=/data/test_artifacts
```

---

### Inference

Generate artifact images from a folder of clean WSI tiles:

```bash
# Add artifacts to 5000 clean tiles (A → B)
python infer.py --config configs/cycle_gan.yaml \
    --checkpoint logs/.../checkpoints/latest.pth \
    --input_dir /data/clean_tiles \
    --output_dir /data/augmented_artifacts \
    --num_images 5000 \
    --direction A2B \
    --options General.device=0

# Restore clean appearance from artifact tiles (B → A)
python infer.py --config configs/cycle_gan.yaml \
    --checkpoint logs/.../checkpoints/latest.pth \
    --input_dir /data/artifact_tiles \
    --output_dir /data/restored_tiles \
    --direction B2A
```

---

## Adding a New Model

Four steps — `train.py`, `test.py`, and `infer.py` require zero modifications.

**Step 1** — Create model file:
```
modules/my_gan/model.py
modules/my_gan/__init__.py
```

**Step 2** — Implement the standard interface:
```python
class MyGAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Declare G, D (follow naming: G_AB, G_BA, D_A, D_B or G, D)
```

**Step 3** — Register in `modules/__init__.py`:
```python
from modules.my_gan.model import MyGAN
MODEL_REGISTRY["my_gan"] = MyGAN
```

**Step 4** — Create YAML config:
```yaml
# configs/my_gan.yaml
_base_: base.yaml
General:
  model_name: my_gan
MyGAN:
  generator_features: 64
```

Then train:
```bash
python train.py --config configs/my_gan.yaml
```

---

## Acknowledgements

- [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) — Zhu et al., 2017
- [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) — Isola et al., 2017
- [CUT](https://github.com/taesungp/contrastive-unpaired-translation) — Park et al., 2020
- [StarGAN](https://github.com/yunjey/stargan) — Choi et al., 2018
- [DiffAugGAN](https://github.com/mit-han-lab/data-efficient-gans) — Zhao et al., 2020

---

<p align="center">
  Built for realistic WSI artifact augmentation in computational pathology.
</p>
