"""modules/dcgan/model.py — DCGAN: Deep Convolutional GAN (Radford et al., 2015).

Architecture improvements over vanilla GAN:
  - Strided convolutions instead of pooling
  - BatchNorm in both generator and discriminator
  - No fully-connected layers (fully convolutional)
  - ReLU in generator, LeakyReLU in discriminator
  - Tanh output in generator

In the WSI artifact context:
  - Single-domain mode: learns to re-style tiles within one domain
  - Used as a baseline before trying CycleGAN or CUT
"""
import torch
import torch.nn as nn


def _weights_init(m):
    """DCGAN paper weight initialisation: N(0, 0.02)."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGenerator(nn.Module):
    """
    Fully convolutional generator.
    Takes noise z or an input image and upsamples to image_size × image_size.
    When used for image-to-image translation (artifact aug), input is a real
    tile rather than random noise.
    """
    def __init__(self, features: int = 64, image_size: int = 256):
        super().__init__()
        # Number of upsampling steps to reach image_size
        # 4 → 8 → 16 → 32 → 64 → 128 → 256  (6 steps for 256px)
        n_up = 0
        sz   = 4
        while sz < image_size:
            sz  *= 2
            n_up += 1

        f    = features * (2 ** (n_up - 1))
        layers = [
            # Bottleneck: 3-channel input → feature maps via strided downconv
            nn.Conv2d(3, f, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
        ]
        # Additional downsampling to reach 4×4 bottleneck
        for _ in range(n_up - 2):
            f_next = min(f * 2, features * 32)
            layers += [
                nn.Conv2d(f, f_next, 4, 2, 1, bias=False),
                nn.BatchNorm2d(f_next),
                nn.LeakyReLU(0.2, True),
            ]
            f = f_next

        self.encoder = nn.Sequential(*layers)

        # Decoder: transposed convolutions up to image_size
        dec = []
        for i in range(n_up):
            f_out = max(f // 2, features)
            dec += [
                nn.ConvTranspose2d(f, f_out, 4, 2, 1, bias=False),
                nn.BatchNorm2d(f_out) if i < n_up - 1 else nn.Identity(),
                nn.ReLU(True)         if i < n_up - 1 else nn.Tanh(),
            ]
            f = f_out

        # Final channel reduction to 3
        dec += [nn.Conv2d(f, 3, 3, 1, 1, bias=False), nn.Tanh()]
        self.decoder = nn.Sequential(*dec)

        self.apply(_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class DCDiscriminator(nn.Module):
    """
    Fully convolutional discriminator.
    Outputs a patch-level real/fake score.
    """
    def __init__(self, features: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1 — no BN on first layer (DCGAN paper)
            nn.Conv2d(3, features,   4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features,   features*2, 4, 2, 1, bias=False), nn.BatchNorm2d(features*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features*2, features*4, 4, 2, 1, bias=False), nn.BatchNorm2d(features*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features*4, features*8, 4, 2, 1, bias=False), nn.BatchNorm2d(features*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features*8, 1, 4, 1, 1, bias=False),
        )
        self.apply(_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DCGAN(nn.Module):
    """DCGAN wrapper — single-domain style transform for WSI artifact augmentation."""
    def __init__(self, cfg):
        super().__init__()
        c          = cfg.DCGAN
        self.G     = DCGenerator(c.generator_features, cfg.Dataset.image_size)
        self.D     = DCDiscriminator(c.discriminator_features)

    def generate(self, x):     return self.G(x)
    def discriminate(self, x): return self.D(x)
