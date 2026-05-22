"""utils/dataset.py — Dataset classes for GAN training and inference."""
import os
import sys
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def _collect_images(root: str) -> List[str]:
    root = os.path.abspath(root)
    if not os.path.exists(root):
        print(f"❌ Path does not exist: {root}")
        sys.exit(1)
    paths = []
    for dirpath, _, fnames in os.walk(root):
        for f in fnames:
            if Path(f).suffix.lower() in VALID_EXTS:
                paths.append(os.path.join(dirpath, f))
    print(f"✅ Collected {len(paths)} images from {root}")
    return sorted(paths)


def build_transform(cfg, augment: bool = True) -> transforms.Compose:
    a   = cfg.Augmentation
    sz  = cfg.Dataset.image_size
    ops = [transforms.Resize((sz, sz))]
    if augment:
        if a.get("random_crop"):
            ops += [transforms.RandomCrop(sz, padding=sz // 8)]
        if a.get("horizontal_flip"):
            ops += [transforms.RandomHorizontalFlip()]
        if a.get("vertical_flip"):
            ops += [transforms.RandomVerticalFlip()]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=list(a.normalize_mean), std=list(a.normalize_std)),
    ]
    return transforms.Compose(ops)


# ─────────────────────────────────────────────────────────────────────────────

class SingleDomainDataset(Dataset):
    """One folder of images — used for vanilla GAN and single-domain GAN."""

    def __init__(self, root: str, transform: Callable):
        self.paths     = _collect_images(root)
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception as e:
            print(f"⚠️  {self.paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        return self.transform(img), self.paths[idx]


class UnpairedDataset(Dataset):
    """Two independent folders — domain A and domain B (CycleGAN / CUT / DiffAugGAN)."""

    def __init__(self, root_a: str, root_b: str, transform: Callable):
        self.paths_a   = _collect_images(root_a)
        self.paths_b   = _collect_images(root_b)
        self.transform = transform

    def __len__(self): return max(len(self.paths_a), len(self.paths_b))

    def __getitem__(self, idx):
        path_a = self.paths_a[idx % len(self.paths_a)]
        path_b = self.paths_b[random.randint(0, len(self.paths_b) - 1)]
        try:
            img_a = self.transform(Image.open(path_a).convert("RGB"))
            img_b = self.transform(Image.open(path_b).convert("RGB"))
        except Exception as e:
            print(f"⚠️  {e}")
            return self.__getitem__((idx + 1) % len(self))
        return img_a, img_b, path_a, path_b


class PairedDataset(Dataset):
    """Two folders with matching filenames — Pix2Pix."""

    def __init__(self, root_a: str, root_b: str, transform: Callable):
        self.paths_a   = _collect_images(root_a)
        self.root_b    = root_b
        self.transform = transform
        # Build paired list
        self.pairs: List[Tuple[str, str]] = []
        for pa in self.paths_a:
            fname = os.path.basename(pa)
            pb    = os.path.join(root_b, fname)
            if os.path.exists(pb):
                self.pairs.append((pa, pb))
        print(f"✅ Paired dataset: {len(self.pairs)} valid pairs")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        pa, pb = self.pairs[idx]
        try:
            ia = self.transform(Image.open(pa).convert("RGB"))
            ib = self.transform(Image.open(pb).convert("RGB"))
        except Exception as e:
            print(f"⚠️  {e}")
            return self.__getitem__((idx + 1) % len(self))
        return ia, ib, pa, pb


class MultiDomainDataset(Dataset):
    """Root folder with one subfolder per domain — StarGAN."""

    def __init__(self, root: str, transform: Callable):
        self.transform = transform
        self.records: List[Tuple[str, int]] = []
        classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.num_domains  = len(classes)
        for cls in classes:
            cls_dir = os.path.join(root, cls)
            for dirpath, _, fnames in os.walk(cls_dir):
                for f in fnames:
                    if Path(f).suffix.lower() in VALID_EXTS:
                        self.records.append((os.path.join(dirpath, f), self.class_to_idx[cls]))
        print(f"✅ MultiDomain: {len(self.records)} images, {self.num_domains} domains: {classes}")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        try:
            img = self.transform(Image.open(path).convert("RGB"))
        except Exception as e:
            print(f"⚠️  {e}")
            return self.__getitem__((idx + 1) % len(self))
        return img, label, path


class InferenceDataset(Dataset):
    """Single folder for inference — no labels."""

    def __init__(self, root: str, transform: Callable):
        self.paths     = _collect_images(root)
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception as e:
            print(f"⚠️  {self.paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        return self.transform(img), self.paths[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Fake image pool (stabilises discriminator training)
# ─────────────────────────────────────────────────────────────────────────────

import torch

class ImagePool:
    """Stores a rolling buffer of generated images to break temporal correlation."""

    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        self.images: list = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return images
        returned = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img)
                returned.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    old = self.images[idx].clone()
                    self.images[idx] = img
                    returned.append(old)
                else:
                    returned.append(img)
        return torch.cat(returned, dim=0)
