"""modules/stargan/model.py — StarGAN: unified multi-domain image translation.
One generator handles ALL domain translations simultaneously.
Domain label is provided as an additional channel to the generator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.cycle_gan.model import ResidualBlock


class StarGenerator(nn.Module):
    """Accepts image + one-hot domain label as input channels."""
    def __init__(self, num_domains: int, features: int = 64, n_res: int = 6):
        super().__init__()
        in_c = 3 + num_domains
        self.initial = nn.Sequential(
            nn.Conv2d(in_c, features, 7, 1, 3), nn.InstanceNorm2d(features), nn.ReLU(True)
        )
        self.down = nn.Sequential(
            nn.Conv2d(features,   features*2, 4, 2, 1), nn.InstanceNorm2d(features*2), nn.ReLU(True),
            nn.Conv2d(features*2, features*4, 4, 2, 1), nn.InstanceNorm2d(features*4), nn.ReLU(True),
        )
        self.res = nn.Sequential(*[ResidualBlock(features*4) for _ in range(n_res)])
        self.up  = nn.Sequential(
            nn.ConvTranspose2d(features*4, features*2, 4, 2, 1), nn.InstanceNorm2d(features*2), nn.ReLU(True),
            nn.ConvTranspose2d(features*2, features,   4, 2, 1), nn.InstanceNorm2d(features),   nn.ReLU(True),
        )
        self.final = nn.Sequential(nn.Conv2d(features, 3, 7, 1, 3), nn.Tanh())

    def forward(self, x: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, 3, H, W)
        domain : (B, num_domains)  — one-hot target domain label
        """
        B, D = domain.shape
        H, W = x.shape[2], x.shape[3]
        label_map = domain.view(B, D, 1, 1).expand(B, D, H, W)
        x = torch.cat([x, label_map], dim=1)
        x = self.initial(x)
        x = self.down(x)
        x = self.res(x)
        x = self.up(x)
        return self.final(x)


class StarDiscriminator(nn.Module):
    """Multi-task discriminator: real/fake + domain classification."""
    def __init__(self, num_domains: int, features: int = 64, n_layers: int = 6):
        super().__init__()
        layers = [nn.Conv2d(3, features, 4, 2, 1), nn.LeakyReLU(0.01, True)]
        f = features
        for _ in range(1, n_layers):
            layers += [nn.Conv2d(f, f*2, 4, 2, 1), nn.LeakyReLU(0.01, True)]
            f = f * 2
        self.shared     = nn.Sequential(*layers)
        self.src_head   = nn.Conv2d(f, 1, 3, 1, 1)          # real/fake patch output
        self.cls_head   = nn.Conv2d(f, num_domains, 2, 1, 0) # domain classifier

    def forward(self, x):
        h   = self.shared(x)
        src = self.src_head(h)
        cls = self.cls_head(h)
        return src, cls.view(cls.shape[0], -1)


class StarGAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.StarGAN
        self.num_domains = c.num_domains
        self.G = StarGenerator(c.num_domains, c.generator_features, c.num_residual_blocks)
        self.D = StarDiscriminator(c.num_domains, c.discriminator_features)
