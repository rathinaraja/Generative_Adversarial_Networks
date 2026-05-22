"""infer.py — Generate artifact images from clean WSI tiles using a trained GAN.
Usage:
  # Generate artifacts from real tiles (A → B)
  python infer.py --config configs/cycle_gan.yaml \
      --checkpoint logs/.../checkpoints/latest.pth \
      --input_dir /data/real_tiles \
      --output_dir /data/generated_artifacts \
      --num_images 5000 \
      --options General.device=0
  # Restore clean images from artifact tiles (B → A)
  python infer.py --config configs/cycle_gan.yaml \
      --checkpoint logs/.../checkpoints/latest.pth \
      --input_dir /data/artifact_tiles \
      --output_dir /data/restored_tiles \
      --direction B2A
"""
import argparse
import os
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.data import Subset

from utils.config  import load_config, print_config
from utils.dataset import InferenceDataset, build_transform
from utils.logger  import get_logger
from modules       import get_model

def parse_args():
    p = argparse.ArgumentParser(description="GAN Inference — Generate Artifact Images")
    p.add_argument("--config",      required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--input_dir",   required=True,  help="Source images folder")
    p.add_argument("--output_dir",  required=True,  help="Output folder for generated images")
    p.add_argument("--num_images",  type=int, default=None, help="Max images to generate (None = all)")
    p.add_argument("--direction",   default="A2B", choices=["A2B", "B2A"],
                   help="A2B: add artifacts   B2A: remove artifacts")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--options",     nargs="+", default=None)
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

    # Select generator
    name = str(cfg.General.model_name).lower()
    if args.direction == "A2B":
        gen = model.G_AB if hasattr(model, "G_AB") else model.G
    else:
        gen = model.G_BA if hasattr(model, "G_BA") else model.F

    tf      = build_transform(cfg, augment=False)
    dataset = InferenceDataset(args.input_dir, tf)

    if args.num_images and args.num_images < len(dataset):
        indices = random.sample(range(len(dataset)), args.num_images)        
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=cfg.General.num_workers, pin_memory=True)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Generating {len(dataset)} images ({args.direction}) → {args.output_dir}")

    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating", ncols=90):
            imgs   = batch[0] if isinstance(batch, (list, tuple)) else batch
            paths  = batch[1] if isinstance(batch, (list, tuple)) else [None] * len(imgs)
            imgs   = imgs.to(device)
            fake   = gen(imgs)

            for j, (src_path, out_img) in enumerate(zip(paths, fake)):
                out_img = (out_img * 0.5 + 0.5).clamp(0, 1)
                if src_path and src_path != "None":
                    fname = os.path.splitext(os.path.basename(src_path))[0]
                    out_fname = f"{fname}_{args.direction}_{count+j:05d}.png"
                else:
                    out_fname = f"generated_{count+j:05d}.png"
                save_image(out_img, os.path.join(args.output_dir, out_fname))

            count += len(imgs)

    logger.info(f"✅ {count} images saved to {args.output_dir}")

if __name__ == "__main__":
    main()
