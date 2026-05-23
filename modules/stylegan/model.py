"""modules/stylegan/model.py — StyleGAN2-inspired architecture (Karras et al., 2020).

Key StyleGAN2 innovations used here:
  - Mapping network: maps domain/style code z → intermediate latent w
  - Adaptive Instance Normalisation (AdaIN) replaced by Weight Demodulation
  - Style injection at every resolution level via modulated convolutions
  - Equalized learning rate (all weights scaled at runtime)
  - No progressive growing (StyleGAN2 uses skip-connection generator)
  - Path-length regularisation (optional, off by default for speed)

Simplified for image-to-image translation:
  - Encoder extracts content code from input image (no pure noise input)
  - Style code is generated from domain label via mapping network
  - Style modulates the decoder at each resolution

In the WSI artifact context:
  - The encoder preserves tissue morphology (content)
  - The style code encodes the artifact domain (colour, texture, pattern)
  - Produces the highest visual quality among all models in this framework
  - Recommended for final production-quality artifact synthesis
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Equalized learning rate helpers
# ─────────────────────────────────────────────────────────────────────────────

class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate."""
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, lr_mul: float = 1.0):
        super().__init__()
        self.weight  = nn.Parameter(torch.randn(out_dim, in_dim) / lr_mul)
        self.bias    = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.scale   = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul  = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.scale,
                        self.bias * self.lr_mul if self.bias is not None else None)


class EqualConv2d(nn.Module):
    """Conv2d with equalized learning rate."""
    def __init__(self, in_c, out_c, kernel, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight  = nn.Parameter(torch.randn(out_c, in_c, kernel, kernel))
        self.bias    = nn.Parameter(torch.zeros(out_c)) if bias else None
        self.stride  = stride
        self.padding = padding
        self.scale   = 1 / math.sqrt(in_c * kernel * kernel)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias,
                        self.stride, self.padding)


# ─────────────────────────────────────────────────────────────────────────────
# Mapping network: domain/style label → intermediate latent w
# ─────────────────────────────────────────────────────────────────────────────

class MappingNetwork(nn.Module):
    """
    Maps a domain label (one-hot or embedding) to intermediate style vector w.
    w is injected at each decoder layer to modulate style.
    """
    def __init__(self, num_domains: int, style_dim: int = 512, n_layers: int = 8):
        super().__init__()
        self.embed = nn.Embedding(num_domains, style_dim)
        layers = []
        for _ in range(n_layers):
            layers += [EqualLinear(style_dim, style_dim), nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(*layers)

    def forward(self, domain: torch.Tensor) -> torch.Tensor:
        z = self.embed(domain)
        z = F.normalize(z, dim=1)
        return self.net(z)


# ─────────────────────────────────────────────────────────────────────────────
# Modulated convolution (StyleGAN2 weight demodulation)
# ─────────────────────────────────────────────────────────────────────────────

class ModulatedConv2d(nn.Module):
    """
    Modulated convolution with weight demodulation (StyleGAN2).
    Style vector w modulates the conv weights; demodulation normalises them
    to prevent artefact blobs ('blob artefacts' from AdaIN in StyleGAN1).
    """
    def __init__(self, in_c: int, out_c: int, kernel: int, style_dim: int,
                 upsample: bool = False, demodulate: bool = True):
        super().__init__()
        self.out_c      = out_c
        self.kernel     = kernel
        self.upsample   = upsample
        self.demodulate = demodulate
        self.scale      = 1 / math.sqrt(in_c * kernel * kernel)
        self.padding    = kernel // 2

        self.weight     = nn.Parameter(torch.randn(1, out_c, in_c, kernel, kernel))
        self.modulation = EqualLinear(style_dim, in_c, bias=True)
        nn.init.ones_(self.modulation.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Modulate
        s = self.modulation(style).view(B, 1, C, 1, 1)
        w = self.weight * self.scale * s       # (B, out_c, in_c, k, k)
        # Demodulate
        if self.demodulate:
            d = torch.rsqrt((w ** 2).sum([2, 3, 4], keepdim=True) + 1e-8)
            w = w * d
        # Reshape for grouped conv
        w  = w.view(B * self.out_c, C, self.kernel, self.kernel)
        x  = x.view(1, B * C, H, W)
        if self.upsample:
            x = F.interpolate(x.view(B, C, H, W), scale_factor=2, mode="bilinear",
                              align_corners=False).view(1, B * C, H * 2, W * 2)
        out = F.conv2d(x, w, padding=self.padding, groups=B)
        return out.view(B, self.out_c, out.shape[-2], out.shape[-1])


# ─────────────────────────────────────────────────────────────────────────────
# StyleGAN2 generator blocks
# ─────────────────────────────────────────────────────────────────────────────

class StyleBlock(nn.Module):
    """One style-modulated resolution block."""
    def __init__(self, in_c: int, out_c: int, style_dim: int, upsample: bool = False):
        super().__init__()
        self.conv1 = ModulatedConv2d(in_c,  out_c, 3, style_dim, upsample)
        self.conv2 = ModulatedConv2d(out_c, out_c, 3, style_dim)
        self.skip  = EqualConv2d(in_c, out_c, 1, bias=False)
        self.act   = nn.LeakyReLU(0.2, True)
        self.up    = upsample

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x, w))
        h = self.act(self.conv2(h, w))
        # Skip
        s = self.skip(x)
        if self.up:
            s = F.interpolate(s, scale_factor=2, mode="bilinear", align_corners=False)
        return (h + s) / math.sqrt(2)


# ─────────────────────────────────────────────────────────────────────────────
# Full Generator and Discriminator
# ─────────────────────────────────────────────────────────────────────────────

class StyleGenerator(nn.Module):
    """
    StyleGAN2-inspired image-to-image generator.
    Content from input image + style from domain label → translated image.
    """
    def __init__(self, features: int = 64, style_dim: int = 512, num_domains: int = 2):
        super().__init__()
        f = features
        # Content encoder
        self.enc = nn.Sequential(
            EqualConv2d(3, f,   7, 1, 3), nn.LeakyReLU(0.2, True),
            EqualConv2d(f, f*2, 3, 2, 1), nn.LeakyReLU(0.2, True),
            EqualConv2d(f*2, f*4, 3, 2, 1), nn.LeakyReLU(0.2, True),
        )
        # Style decoder with modulated convs
        self.style_blocks = nn.ModuleList([
            StyleBlock(f*4, f*4, style_dim),
            StyleBlock(f*4, f*2, style_dim, upsample=True),
            StyleBlock(f*2, f,   style_dim, upsample=True),
        ])
        self.to_rgb = EqualConv2d(f, 3, 7, 1, 3)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        for blk in self.style_blocks:
            h = blk(h, w)
        return torch.tanh(self.to_rgb(h))


class StyleDiscriminator(nn.Module):
    """
    Residual discriminator with minibatch std dev for diversity.
    """
    def __init__(self, features: int = 64):
        super().__init__()
        f = features
        self.blocks = nn.Sequential(
            EqualConv2d(3,   f,   4, 2, 1), nn.LeakyReLU(0.2, True),
            EqualConv2d(f,   f*2, 4, 2, 1), nn.LeakyReLU(0.2, True),
            EqualConv2d(f*2, f*4, 4, 2, 1), nn.LeakyReLU(0.2, True),
            EqualConv2d(f*4, f*8, 4, 2, 1), nn.LeakyReLU(0.2, True),
        )
        # +1 for minibatch std channel
        self.final = nn.Sequential(
            EqualConv2d(f*8 + 1, f*8, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            EqualLinear(f*8, 1),
        )

    @staticmethod
    def _minibatch_std(x: torch.Tensor) -> torch.Tensor:
        """Appends a channel of the batch-wide std dev to stabilise diversity."""
        std = x.std(dim=0, keepdim=True).mean().expand(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, std], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)
        h = self._minibatch_std(h)
        return self.final(h)


class StyleGAN(nn.Module):
    """
    StyleGAN2-inspired model for high-quality WSI artifact transfer.
    Mapping network converts domain label → style vector w.
    G_AB/G_BA share the mapping network but have separate style decoders.
    """
    def __init__(self, cfg):
        super().__init__()
        c          = cfg.StyleGAN
        nd         = c.get("num_domains", 2)
        sd         = c.get("style_dim", 512)
        self.map   = MappingNetwork(nd, sd, c.get("mapping_layers", 8))
        self.G_AB  = StyleGenerator(c.generator_features, sd, nd)
        self.G_BA  = StyleGenerator(c.generator_features, sd, nd)
        self.D_A   = StyleDiscriminator(c.discriminator_features)
        self.D_B   = StyleDiscriminator(c.discriminator_features)

    def get_style(self, domain: torch.Tensor) -> torch.Tensor:
        return self.map(domain)
