"""modules/cycle_gan/model.py — CycleGAN with ResNet generator (your Cycled_GAN_Cross_Domain_ResNet.py)."""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
        )
    def forward(self, x): return x + self.block(x)


class ResNetGenerator(nn.Module):
    """ResNet generator — directly from your cross-domain training script."""
    def __init__(self, features: int = 64, n_res: int = 9):
        super().__init__()
        self.initial    = nn.Sequential(nn.Conv2d(3, features, 7, 1, 3),
                                        nn.InstanceNorm2d(features), nn.ReLU(True))
        self.down       = nn.Sequential(
            nn.Conv2d(features,   features*2, 3, 2, 1), nn.InstanceNorm2d(features*2), nn.ReLU(True),
            nn.Conv2d(features*2, features*4, 3, 2, 1), nn.InstanceNorm2d(features*4), nn.ReLU(True),
        )
        self.res        = nn.Sequential(*[ResidualBlock(features * 4) for _ in range(n_res)])
        self.up         = nn.Sequential(
            nn.ConvTranspose2d(features*4, features*2, 3, 2, 1, 1), nn.InstanceNorm2d(features*2), nn.ReLU(True),
            nn.ConvTranspose2d(features*2, features,   3, 2, 1, 1), nn.InstanceNorm2d(features),   nn.ReLU(True),
        )
        self.final      = nn.Sequential(nn.Conv2d(features, 3, 7, 1, 3), nn.Tanh())
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial(x)
        x = self.down(x)
        x = self.res(x)
        x = self.up(x)
        return self.final(x)


class PatchDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator."""
    def __init__(self, features: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, features,   4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features,   features*2, 4, 2, 1), nn.InstanceNorm2d(features*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features*2, features*4, 4, 2, 1), nn.InstanceNorm2d(features*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features*4, features*8, 4, 1, 1), nn.InstanceNorm2d(features*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features*8, 1, 4, 1, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x): return self.model(x)


class CycleGAN(nn.Module):
    """
    Full CycleGAN:
      G_AB : domain A → domain B  (real WSI → artifact)
      G_BA : domain B → domain A  (artifact → real WSI)
      D_A  : discriminates domain A images
      D_B  : discriminates domain B images
    """
    def __init__(self, cfg):
        super().__init__()
        c = cfg.CycleGAN
        f, n = c.generator_features, c.num_residual_blocks
        self.G_AB = ResNetGenerator(f, n)
        self.G_BA = ResNetGenerator(f, n)
        self.D_A  = PatchDiscriminator(c.discriminator_features)
        self.D_B  = PatchDiscriminator(c.discriminator_features)
