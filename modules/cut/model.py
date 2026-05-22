"""modules/cut/model.py — CUT: Contrastive Unpaired Translation (Park et al., 2020).
Single-sided unpaired translation — trains 2× faster than CycleGAN with equal quality.
Uses PatchNCE loss instead of cycle consistency — no reverse generator F needed.
"""
import torch
import torch.nn as nn
from modules.cycle_gan.model import ResNetGenerator, PatchDiscriminator


class MLP(nn.Module):
    """Two-layer MLP projection head for PatchNCE feature matching."""
    def __init__(self, in_dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(True),
            nn.Linear(in_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class CUT(nn.Module):
    """
    Generator G: A → B  (e.g. real WSI → artifact WSI)
    Discriminator D_B: real B vs fake B
    Feature projection heads: one per NCE layer
    """
    def __init__(self, cfg):
        super().__init__()
        c         = cfg.CUT
        self.G    = ResNetGenerator(c.generator_features, c.num_residual_blocks)
        self.D_B  = PatchDiscriminator(c.discriminator_features)
        # NCE projection MLPs — one per feature extraction layer
        self.mlps = nn.ModuleList([MLP() for _ in c.nce_layers])
        self.nce_layers = list(c.nce_layers)

    def get_features(self, x: torch.Tensor, model: nn.Module) -> list:
        """Extract intermediate feature maps at nce_layers positions."""
        feats  = []
        layers = list(model.initial) + list(model.down) + list(model.res) + list(model.up) + list(model.final)
        for i, layer in enumerate(layers):
            x = layer(x)
            if i in self.nce_layers:
                feats.append(x)
            if len(feats) == len(self.nce_layers):
                break
        return feats
