"""utils/logger.py — Console logger, CSV logger, checkpoint manager, visual sampler."""
import csv
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
from torchvision.utils import save_image


def get_logger(name: str = "gan_framework") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def make_run_dir(cfg) -> str:
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = cfg.General.get("model_name", "model")
    path  = Path(cfg.Logs.log_root_dir) / f"{ts}_{model}"
    path.mkdir(parents=True, exist_ok=True)

    yaml_src = getattr(cfg, "_yaml_path", None)
    if yaml_src and Path(yaml_src).exists():
        shutil.copy(yaml_src, path / Path(yaml_src).name)
    return str(path)


class CSVLogger:
    def __init__(self, path: str, fieldnames: list):
        self.path       = path
        self.fieldnames = fieldnames
        if not Path(path).exists():
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames,
                           extrasaction="ignore").writerow(row)


class CheckpointManager:
    def __init__(self, run_dir: str, keep_k: int = 3):
        self.ckpt_dir = Path(run_dir) / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        self.keep_k   = keep_k
        self._history: list = []

    def save(self, state: dict, epoch: int, tag: str = "") -> str:
        fname = self.ckpt_dir / f"epoch_{epoch:04d}{('_'+tag) if tag else ''}.pth"
        torch.save(state, fname)
        self._history.append(str(fname))
        while len(self._history) > self.keep_k:
            old = self._history.pop(0)
            if Path(old).exists():
                Path(old).unlink()
        shutil.copy(fname, self.ckpt_dir / "latest.pth")
        return str(fname)


def save_sample_images(run_dir: str, epoch: int, images: dict, denorm: bool = True):
    """
    Save a grid of sample images.
    images: dict of {name: tensor (B,C,H,W)} e.g.
        {"real_A": ..., "fake_B": ..., "cycled_A": ...}
    """
    sample_dir = Path(run_dir) / "samples" / f"epoch_{epoch:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in images.items():
        if tensor is None:
            continue
        img = (tensor * 0.5 + 0.5).clamp(0, 1) if denorm else tensor.clamp(0, 1)
        save_image(img, sample_dir / f"{name}.png", nrow=min(4, img.shape[0]))
