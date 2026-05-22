"""utils/losses.py — All GAN loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial losses
# ─────────────────────────────────────────────────────────────────────────────

class LSGANLoss(nn.Module):
    """Least-squares GAN loss — more stable than vanilla BCE."""
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def __call__(self, pred, is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


class VanillaGANLoss(nn.Module):
    """Standard BCE-with-logits GAN loss."""
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, pred, is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


class HingeLoss(nn.Module):
    """Hinge loss — used in SAGAN, BigGAN."""
    def __call__(self, pred, is_real: bool) -> torch.Tensor:
        if is_real:
            return F.relu(1.0 - pred).mean()
        return F.relu(1.0 + pred).mean()


def gradient_penalty(discriminator, real: torch.Tensor, fake: torch.Tensor,
                     device: torch.device, lambda_gp: float = 10.0) -> torch.Tensor:
    """WGAN-GP gradient penalty."""
    B   = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_interp = discriminator(interpolated)
    grads    = torch.autograd.grad(
        outputs=d_interp, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True
    )[0]
    grads   = grads.view(B, -1)
    penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * penalty


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction / consistency losses
# ─────────────────────────────────────────────────────────────────────────────

def cycle_consistency_loss(real: torch.Tensor, cycled: torch.Tensor,
                           weight: float = 10.0) -> torch.Tensor:
    return weight * F.l1_loss(cycled, real)


def identity_loss(real: torch.Tensor, same: torch.Tensor,
                  weight: float = 5.0) -> torch.Tensor:
    return weight * F.l1_loss(same, real)


def l1_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor,
                            weight: float = 100.0) -> torch.Tensor:
    return weight * F.l1_loss(pred, target)


# ─────────────────────────────────────────────────────────────────────────────
# PatchNCE loss for CUT
# ─────────────────────────────────────────────────────────────────────────────

class PatchNCELoss(nn.Module):
    """
    Contrastive patch-level loss for CUT (Contrastive Unpaired Translation).
    Maximises mutual information between corresponding patches in source
    and translated image feature maps.
    """
    def __init__(self, temperature: float = 0.07, num_patches: int = 256):
        super().__init__()
        self.temperature = temperature
        self.num_patches = num_patches
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor) -> torch.Tensor:
        B, C = feat_q.shape[:2]
        feat_q = feat_q.view(B, C, -1).permute(0, 2, 1)   # (B, N, C)
        feat_k = feat_k.view(B, C, -1).permute(0, 2, 1)

        # Sample patches
        N = min(self.num_patches, feat_q.shape[1])
        idx = torch.randperm(feat_q.shape[1], device=feat_q.device)[:N]
        feat_q = F.normalize(feat_q[:, idx, :], dim=-1)
        feat_k = F.normalize(feat_k[:, idx, :], dim=-1)

        # Positive: same spatial location across q and k
        logits = torch.bmm(feat_q, feat_k.transpose(1, 2)) / self.temperature   # (B,N,N)
        labels = torch.arange(N, device=feat_q.device).unsqueeze(0).expand(B, -1)
        return self.cross_entropy(logits.view(B * N, N), labels.reshape(B * N))


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable Augmentation (DiffAugGAN)
# ─────────────────────────────────────────────────────────────────────────────

def diff_augment(x: torch.Tensor, policy: str = "color,translation,cutout") -> torch.Tensor:
    """Apply differentiable augmentations to discriminator inputs."""
    fns = {
        "color":       _rand_color,
        "translation": _rand_translation,
        "cutout":      _rand_cutout,
    }
    for p in policy.split(","):
        p = p.strip()
        if p in fns:
            x = fns[p](x)
    return x


def _rand_color(x: torch.Tensor) -> torch.Tensor:
    x = x + torch.rand(x.shape[0], 3, 1, 1, device=x.device) - 0.5
    x = x * (torch.rand(x.shape[0], 3, 1, 1, device=x.device) * 2)
    return x.clamp(-1, 1)


def _rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    B, C, H, W = x.shape
    dx = torch.randint(-int(W * ratio), int(W * ratio) + 1, (B,))
    dy = torch.randint(-int(H * ratio), int(H * ratio) + 1, (B,))
    shifted = []
    for i in range(B):
        shifted.append(torch.roll(torch.roll(x[i], dx[i].item(), -1), dy[i].item(), -2))
    return torch.stack(shifted)


def _rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    B, C, H, W = x.shape
    cut_h = int(H * ratio)
    cut_w = int(W * ratio)
    out   = x.clone()
    for i in range(B):
        cy = torch.randint(0, H, (1,)).item()
        cx = torch.randint(0, W, (1,)).item()
        y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
        x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
        out[i, :, y1:y2, x1:x2] = 0
    return out
