"""modules/__init__.py — Model registry."""
from modules.vanilla_gan.model       import VanillaGAN
from modules.cycle_gan.model         import CycleGAN
from modules.single_domain_gan.model import SingleDomainGAN
from modules.pix2pix.model           import Pix2Pix
from modules.cut.model               import CUT
from modules.stargan.model           import StarGAN
from modules.diffaug_gan.model       import DiffAugGAN
from modules.dcgan.model             import DCGAN
from modules.resnet_gan.model        import ResNetGAN
from modules.biggan.model            import BigGAN
from modules.stylegan.model          import StyleGAN

MODEL_REGISTRY = {
    # ── Original models ───────────────────────────────────────────────────────
    "vanilla_gan":       VanillaGAN,
    "cycle_gan":         CycleGAN,
    "single_domain_gan": SingleDomainGAN,
    "pix2pix":           Pix2Pix,
    "cut":               CUT,
    "stargan":           StarGAN,
    "diffaug_gan":       DiffAugGAN,
    # ── New models ────────────────────────────────────────────────────────────
    "dcgan":             DCGAN,
    "resnet_gan":        ResNetGAN,
    "biggan":            BigGAN,
    "stylegan":          StyleGAN,
}


def get_model(cfg):
    name = str(cfg.General.model_name).lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'.\nAvailable: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](cfg)
