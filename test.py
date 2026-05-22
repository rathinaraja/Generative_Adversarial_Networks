"""test.py — Evaluate a trained GAN on a held-out set.
Generates images from domain A → B (and B → A for cycle models), saves them, and
computes FID if pytorch-fid is installed.
Usage:
  python test.py --config configs/cycle_gan.yaml \
      --checkpoint logs/20240601_cycle_gan/checkpoints/latest.pth \
      --options Dataset.domain_a_dir=/data/test_real Dataset.domain_b_dir=/data/test_artifacts General.device=0
  # Compute FID (requires pytorch-fid)
  python test.py --config configs/cycle_gan.yaml --checkpoint logs/.../checkpoints/latest.pth --fid
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from utils.config  import load_config, print_config
from utils.dataset import SingleDomainDataset, UnpairedDataset, build_transform
from utils.logger  import get_logger
from modules       import get_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fid",        action="store_true", help="Compute FID score")
    p.add_argument("--options",    nargs="+", default=None)
    return p.parse_args()

def main():
    args   = parse_args()
    cfg    = load_config(args.config, args.options)
    logger = get_logger()
    print_config(cfg)

    dev    = cfg.General.device
    device = torch.device(f"cuda:{dev}" if torch.cuda.is_available() and dev >= 0 else "cpu")
    model  = get_model(cfg).to(device)
    ckpt   = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    logger.info(f"Loaded: {args.checkpoint}")

    tf      = build_transform(cfg, augment=False)
    ds      = cfg.Dataset
    name    = str(cfg.General.model_name).lower()
    out_dir = os.path.join(os.path.dirname(args.checkpoint), "test_output")
    os.makedirs(os.path.join(out_dir, "fake_B"), exist_ok=True)

    if ds.domain_b_dir:
        dataset = UnpairedDataset(ds.domain_a_dir, ds.domain_b_dir, tf)
        os.makedirs(os.path.join(out_dir, "fake_A"), exist_ok=True)
    else:
        dataset = SingleDomainDataset(ds.domain_a_dir, tf)

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=cfg.General.num_workers)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Generating")):
            real_A = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)

            if name in ("cycle_gan", "diffaug_gan", "cut"):
                fake_B = model.G_AB(real_A)
                for j, img in enumerate(fake_B):
                    save_image((img * 0.5 + 0.5).clamp(0, 1), os.path.join(out_dir, "fake_B", f"{i*4+j:05d}.png"))
                if hasattr(model, "G_BA") and ds.domain_b_dir:
                    real_B = batch[1].to(device)
                    fake_A = model.G_BA(real_B)
                    for j, img in enumerate(fake_A):
                        save_image((img * 0.5 + 0.5).clamp(0, 1), os.path.join(out_dir, "fake_A", f"{i*4+j:05d}.png"))
            else:
                gen = model.G if hasattr(model, "G") else model.G_AB
                fake = gen(real_A)
                for j, img in enumerate(fake):
                    save_image((img * 0.5 + 0.5).clamp(0, 1), os.path.join(out_dir, "fake_B", f"{i*4+j:05d}.png"))

    logger.info(f"✅ Generated images saved to {out_dir}")

    if args.fid:
        try:
            import subprocess
            result = subprocess.run(
                ["python", "-m", "pytorch_fid", ds.domain_b_dir,
                 os.path.join(out_dir, "fake_B"), "--device", f"cuda:{dev}"], capture_output=True, text=True
            )
            logger.info(f"FID score:\n{result.stdout}")
        except Exception as e:
            logger.warning(f"FID computation failed: {e}. Install: pip install pytorch-fid")

if __name__ == "__main__":
    main()