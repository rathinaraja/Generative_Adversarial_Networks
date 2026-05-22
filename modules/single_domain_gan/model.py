"""modules/single_domain_gan/model.py — Single-domain CycleGAN (from your single_domain_gan.py).
Learns internal style variation within ONE domain (no paired domain_b needed).
G learns a style transform; F learns the inverse; cycle consistency keeps content.
"""
from modules.cycle_gan.model import ResNetGenerator, PatchDiscriminator
import torch.nn as nn


class SingleDomainGAN(nn.Module):
    """
    Two generators (G, F) and two discriminators operating on the
    SAME domain — G augments images with learned style variation,
    F reconstructs the original. Enforces cycle consistency.
    """
    def __init__(self, cfg):
        super().__init__()
        c = cfg.SingleDomainGAN
        f, n = c.generator_features, c.num_residual_blocks
        self.G = ResNetGenerator(f, n)   # source → stylised
        self.F = ResNetGenerator(f, n)   # stylised → source (inverse)
        self.D_real  = PatchDiscriminator(c.discriminator_features)  # real vs generated
        self.D_style = PatchDiscriminator(c.discriminator_features)  # style side
