"""modules/pix2pix/model.py — Pix2Pix: paired image-to-image translation."""
import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True, bn=True, dropout=False, inner=False, outer=False):
        super().__init__()
        self.outer   = outer
        self.inner   = inner
        use_bias     = not bn
        if down:
            conv = nn.Conv2d(in_c, out_c, 4, 2, 1, bias=use_bias)
        else:
            conv = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=use_bias)
        relu  = nn.ReLU(True) if not down else nn.LeakyReLU(0.2, True)
        norm  = [nn.BatchNorm2d(out_c)] if bn else []
        drop  = [nn.Dropout(0.5)] if dropout else []

        if outer:
            self.model = nn.Sequential(conv)
            self.out   = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(in_c * 2, out_c, 4, 2, 1), nn.Tanh())
        elif inner:
            self.model = nn.Sequential(nn.LeakyReLU(0.2, True), conv, relu, *norm)
        else:
            down_part = [nn.LeakyReLU(0.2, True), conv] + norm
            up_part   = [relu, conv, *norm, *drop]
            self.model     = nn.Sequential(*down_part)
            self.up_model  = nn.Sequential(*up_part)

    def forward(self, x, skip=None):
        if self.outer:
            x = self.model(x)
            return x
        elif self.inner:
            return self.model(x)
        else:
            x = self.model(x)
            if skip is not None:
                return torch.cat([x, skip], dim=1)
            return x


class UNetGenerator(nn.Module):
    """U-Net generator for Pix2Pix — skip connections preserve spatial detail."""
    def __init__(self, features: int = 64):
        super().__init__()
        f = features
        # Encoder
        self.e1  = nn.Sequential(nn.Conv2d(3, f, 4, 2, 1))                                    # 128
        self.e2  = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(f,   f*2, 4, 2, 1), nn.BatchNorm2d(f*2))  # 64
        self.e3  = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(f*2, f*4, 4, 2, 1), nn.BatchNorm2d(f*4))  # 32
        self.e4  = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(f*4, f*8, 4, 2, 1), nn.BatchNorm2d(f*8))  # 16
        self.e5  = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(f*8, f*8, 4, 2, 1), nn.BatchNorm2d(f*8))  # 8
        self.e6  = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(f*8, f*8, 4, 2, 1), nn.BatchNorm2d(f*8))  # 4
        self.e7  = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(f*8, f*8, 4, 2, 1), nn.BatchNorm2d(f*8))  # 2
        self.e8  = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(f*8, f*8, 4, 2, 1))                       # 1 (bottleneck)
        # Decoder
        self.d1  = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(f*8,  f*8, 4, 2, 1), nn.BatchNorm2d(f*8),  nn.Dropout(0.5))
        self.d2  = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(f*16, f*8, 4, 2, 1), nn.BatchNorm2d(f*8),  nn.Dropout(0.5))
        self.d3  = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(f*16, f*8, 4, 2, 1), nn.BatchNorm2d(f*8),  nn.Dropout(0.5))
        self.d4  = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(f*16, f*8, 4, 2, 1), nn.BatchNorm2d(f*8))
        self.d5  = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(f*16, f*4, 4, 2, 1), nn.BatchNorm2d(f*4))
        self.d6  = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(f*8,  f*2, 4, 2, 1), nn.BatchNorm2d(f*2))
        self.d7  = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(f*4,  f,   4, 2, 1), nn.BatchNorm2d(f))
        self.d8  = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(f*2,  3,   4, 2, 1), nn.Tanh())

    def forward(self, x):
        import torch
        e1 = self.e1(x);   e2 = self.e2(e1)
        e3 = self.e3(e2);  e4 = self.e4(e3)
        e5 = self.e5(e4);  e6 = self.e6(e5)
        e7 = self.e7(e6);  e8 = self.e8(e7)
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        return self.d8(torch.cat([d7, e1], 1))


class ConditionalDiscriminator(nn.Module):
    """PatchGAN discriminator conditioned on input image (Pix2Pix-style)."""
    def __init__(self, features: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Conv2d(6, features, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        f = features
        for _ in range(n_layers - 1):
            layers += [nn.Conv2d(f, f*2, 4, 2, 1), nn.InstanceNorm2d(f*2), nn.LeakyReLU(0.2, True)]
            f = f * 2
        layers += [nn.Conv2d(f, f*2, 4, 1, 1), nn.InstanceNorm2d(f*2), nn.LeakyReLU(0.2, True),
                   nn.Conv2d(f*2, 1, 4, 1, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x, cond):
        import torch
        return self.model(torch.cat([x, cond], dim=1))


class Pix2Pix(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.Pix2Pix
        self.G = UNetGenerator(c.generator_features)
        self.D = ConditionalDiscriminator(c.discriminator_features, c.n_layers_disc)
