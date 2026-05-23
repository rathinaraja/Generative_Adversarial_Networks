"""modules/resnet_gan/model.py — ResNet GAN: ResNet-50 backbone encoder-decoder GAN.

Architecture:
  Generator  : Pretrained ResNet-50 encoder (ImageNet) + transposed-conv decoder
  Discriminator: PatchGAN with InstanceNorm

This is the production-quality model from your original
Cycled_GAN_Cross_Domain_ResNet.py / Generating_images_ResNet.py scripts,
now with a pretrained ResNet-50 backbone for richer feature representations.

Key advantages over vanilla CycleGAN:
  - Pretrained ImageNet features → better texture and colour transfer
  - Deeper encoder captures high-level tissue structure
  - Transfer learning stabilises early training

In the WSI artifact context:
  - Use as a drop-in upgrade over cycle_gan when training data is limited
  - Pretrained encoder requires fewer epochs to converge
"""
import torch
import torch.nn as nn
from torchvision import models


# ─────────────────────────────────────────────────────────────────────────────
# Residual block (used in decoder)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels),
        )
    def forward(self, x): return x + self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class ResNetGenerator(nn.Module):
    """
    ResNet-50 encoder (optionally pretrained) + residual bottleneck + upsampling decoder.
    Produces the same spatial resolution as the input.
    """
    def __init__(self, pretrained: bool = True, n_res: int = 9, features: int = 256):
        super().__init__()

        # ── Encoder: ResNet-50 up to layer2 (stride 8, feature dim 512) ──────
        resnet         = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.enc_init  = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.enc_l1    = resnet.layer1    # 256-ch, stride 4
        self.enc_l2    = resnet.layer2    # 512-ch, stride 8

        # Adapt 512 → features for residual bottleneck
        self.adapt     = nn.Sequential(
            nn.Conv2d(512, features, 1, bias=False),
            nn.InstanceNorm2d(features),
            nn.ReLU(True),
        )

        # ── Bottleneck: residual blocks ───────────────────────────────────────
        self.res        = nn.Sequential(*[ResidualBlock(features) for _ in range(n_res)])

        # ── Decoder: upsample × 8 back to original resolution ────────────────
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(features, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256), nn.ReLU(True),              # ×2
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128), nn.ReLU(True),              # ×4
            nn.ConvTranspose2d(128, 64,  4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),  nn.ReLU(True),              # ×8
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh(),
        )

        # Initialise non-pretrained layers
        for m in list(self.adapt.modules()) + list(self.res.modules()) + list(self.decoder.modules()):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc_init(x)
        h = self.enc_l1(h)
        h = self.enc_l2(h)
        h = self.adapt(h)
        h = self.res(h)
        return self.decoder(h)


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator  (same PatchGAN as CycleGAN)
# ─────────────────────────────────────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    def __init__(self, features: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, features,   4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features,   features*2, 4, 2, 1), nn.InstanceNorm2d(features*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features*2, features*4, 4, 2, 1), nn.InstanceNorm2d(features*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features*4, features*8, 4, 1, 1), nn.InstanceNorm2d(features*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features*8, 1, 4, 1, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x): return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class ResNetGAN(nn.Module):
    """
    ResNet GAN for cross-domain WSI artifact transfer.
    G_AB : clean tile  → artifact tile
    G_BA : artifact tile → clean tile
    D_A  : discriminates domain A (clean)
    D_B  : discriminates domain B (artifact)
    """
    def __init__(self, cfg):
        super().__init__()
        c         = cfg.ResNetGAN
        pretrained = c.get("pretrained", True)
        self.G_AB = ResNetGenerator(pretrained, c.num_residual_blocks, c.bottleneck_features)
        self.G_BA = ResNetGenerator(pretrained, c.num_residual_blocks, c.bottleneck_features)
        self.D_A  = PatchDiscriminator(c.discriminator_features)
        self.D_B  = PatchDiscriminator(c.discriminator_features)
