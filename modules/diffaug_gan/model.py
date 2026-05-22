"""modules/diffaug_gan/model.py — DiffAugGAN: Differentiable Augmentation GAN.
Applies differentiable augmentation to both real and fake images before
passing to the discriminator — critical for small WSI artifact datasets.
Same ResNet generator as CycleGAN; discriminator wraps DiffAug.
"""
import torch
import torch.nn as nn
from modules.cycle_gan.model import ResNetGenerator, PatchDiscriminator
from utils.losses import diff_augment


class AugDiscriminator(nn.Module):
    """PatchGAN discriminator with differentiable augmentation applied to inputs."""
    def __init__(self, features: int = 64, aug_policy: str = "color,translation,cutout"):
        super().__init__()
        self.disc       = PatchDiscriminator(features)
        self.aug_policy = aug_policy

    def forward(self, x: torch.Tensor, augment: bool = True) -> torch.Tensor:
        if augment and self.training:
            x = diff_augment(x, self.aug_policy)
        return self.disc(x)


class DiffAugGAN(nn.Module):
    """
    CycleGAN architecture with DiffAug discriminators.
    Strongly recommended when WSI artifact dataset is < 1000 images.
    """
    def __init__(self, cfg):
        super().__init__()
        c    = cfg.DiffAugGAN
        f, n = c.generator_features, c.num_residual_blocks
        pol  = c.aug_policy
        self.G_AB = ResNetGenerator(f, n)
        self.G_BA = ResNetGenerator(f, n)
        self.D_A  = AugDiscriminator(c.discriminator_features, pol)
        self.D_B  = AugDiscriminator(c.discriminator_features, pol)
