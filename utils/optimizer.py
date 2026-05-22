"""utils/optimizer.py — Optimizer and scheduler factory for GAN training."""
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


def build_optimizer(params, lr: float, beta1: float, beta2: float) -> torch.optim.Optimizer:
    return Adam(params, lr=lr, betas=(beta1, beta2))


def build_scheduler(optimizer: torch.optim.Optimizer, cfg, name: str = None):
    t    = cfg.Training
    name = name or str(t.get("scheduler", "linear_decay")).lower()

    if name == "linear_decay":
        # Hold LR constant until decay_start_epoch, then linearly decay to 0
        total       = t.epochs
        decay_start = t.get("decay_start_epoch", total // 2)

        def lr_lambda(epoch):
            if epoch < decay_start:
                return 1.0
            return max(0.0, 1.0 - (epoch - decay_start) / (total - decay_start))

        return LambdaLR(optimizer, lr_lambda)

    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=t.epochs, eta_min=1e-6)

    if name == "none":
        return None

    raise ValueError(f"Unknown scheduler: {name}")
