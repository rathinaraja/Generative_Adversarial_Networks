"""modules/vanilla_gan/model.py — Classic GAN with simple CNN generator/discriminator."""
import torch
import torch.nn as nn


class Generator(nn.Module):
    """Encoder-decoder generator (from your original single_domain_gan.py)."""
    def __init__(self, features: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, features, 7, 1, 3),           nn.BatchNorm2d(features),   nn.ReLU(True),
            nn.Conv2d(features, features*2, 3, 2, 1),  nn.BatchNorm2d(features*2), nn.ReLU(True),
            nn.Conv2d(features*2, features*4, 3, 2, 1),nn.BatchNorm2d(features*4), nn.ReLU(True),
            nn.ConvTranspose2d(features*4, features*2, 3, 2, 1, 1), nn.BatchNorm2d(features*2), nn.ReLU(True),
            nn.ConvTranspose2d(features*2, features,   3, 2, 1, 1), nn.BatchNorm2d(features),   nn.ReLU(True),
            nn.Conv2d(features, 3, 7, 1, 3), nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x): return self.model(x)


class Discriminator(nn.Module):
    """PatchGAN discriminator."""
    def __init__(self, features: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, features,   4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features,   features*2, 4, 2, 1), nn.InstanceNorm2d(features*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features*2, features*4, 4, 2, 1), nn.InstanceNorm2d(features*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features*4, features*8, 4, 2, 1), nn.InstanceNorm2d(features*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features*8, 1, 4, 1, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x): return self.model(x)


class VanillaGAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.VanillaGAN
        self.G = Generator(c.generator_features)
        self.D = Discriminator(c.discriminator_features)

    def generate(self, x): return self.G(x)
    def discriminate(self, x): return self.D(x)
