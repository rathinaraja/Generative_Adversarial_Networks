"""train.py — GAN training entry point.
Usage:
  # CycleGAN cross-domain (real WSI ↔ artifact)
  python train.py --config configs/cycle_gan.yaml \
      --options Dataset.domain_a_dir=/data/real Dataset.domain_b_dir=/data/artifacts Training.epochs=300 General.device=0

  # Single-domain (learn style variation from artifact tiles alone)
  python train.py --config configs/single_domain_gan.yaml \
      --options Dataset.domain_a_dir=/data/artifacts Training.epochs=200 General.device=1

  # CUT (faster than CycleGAN, state-of-the-art)
  python train.py --config configs/cut.yaml \
      --options Dataset.domain_a_dir=/data/real Dataset.domain_b_dir=/data/artifacts

  # Limited data? Use DiffAugGAN
  python train.py --config configs/diffaug_gan.yaml \
      --options Dataset.domain_a_dir=/data/real Dataset.domain_b_dir=/data/artifacts
"""
import argparse
import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from utils.config import setup_device, wrap_model
from utils.config    import load_config, print_config
from utils.dataset   import (SingleDomainDataset, UnpairedDataset, PairedDataset,
                              MultiDomainDataset, ImagePool, build_transform)
from utils.losses    import (LSGANLoss, VanillaGANLoss, cycle_consistency_loss,
                              identity_loss, l1_reconstruction_loss, PatchNCELoss,
                              gradient_penalty)
from utils.logger    import get_logger, make_run_dir, CSVLogger, CheckpointManager, save_sample_images
from utils.optimizer import build_optimizer, build_scheduler
from modules         import get_model

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def parse_args():
    p = argparse.ArgumentParser(description="GAN Framework Trainer")
    p.add_argument("--config",  required=True, help="Path to YAML config")
    p.add_argument("--options", nargs="+", default=None,
                   help="Override: Section.key=value ...")
    return p.parse_args()

def build_loader(cfg, augment: bool = True):
    ds   = cfg.Dataset
    tf   = build_transform(cfg, augment)
    name = str(cfg.General.model_name).lower()

    if name == "stargan":
        dataset = MultiDomainDataset(ds.domain_a_dir, tf)
    elif name in ("pix2pix",) or ds.get("paired"):
        dataset = PairedDataset(ds.domain_a_dir, ds.domain_b_dir, tf)
    elif ds.domain_b_dir:
        dataset = UnpairedDataset(ds.domain_a_dir, ds.domain_b_dir, tf)
    else:
        dataset = SingleDomainDataset(ds.domain_a_dir, tf)

    return DataLoader(dataset, batch_size=cfg.Training.batch_size,
                      shuffle=True, num_workers=cfg.General.num_workers,
                      pin_memory=True, drop_last=True)
# ─────────────────────────────────────────────────────────────────────────────
# Per-model training steps
# ─────────────────────────────────────────────────────────────────────────────
def step_cycle_gan(model, batch, opts, scaler, criterion, pools, cfg):
    real_A, real_B = batch[0], batch[1]
    t = cfg.Training
    opt_G, opt_D_A, opt_D_B = opts

    with autocast(enabled=cfg.General.mixed_precision):
        fake_B    = model.G_AB(real_A);   fake_A    = model.G_BA(real_B)
        cycled_A  = model.G_BA(fake_B);   cycled_B  = model.G_AB(fake_A)
        same_A    = model.G_BA(real_A);   same_B    = model.G_AB(real_B)

        # Generator loss
        loss_G  = (criterion(model.D_B(fake_B), True)  + criterion(model.D_A(fake_A), True)
                   + cycle_consistency_loss(real_A, cycled_A, t.lambda_cycle)
                   + cycle_consistency_loss(real_B, cycled_B, t.lambda_cycle)
                   + identity_loss(real_A, same_A, t.lambda_identity)
                   + identity_loss(real_B, same_B, t.lambda_identity))

    opt_G.zero_grad(); scaler.scale(loss_G).backward(); scaler.step(opt_G); scaler.update()

    with autocast(enabled=cfg.General.mixed_precision):
        fake_B_pool = pools[0].query(fake_B.detach())
        fake_A_pool = pools[1].query(fake_A.detach())
        loss_DA = (criterion(model.D_A(real_A), True) + criterion(model.D_A(fake_A_pool), False)) * 0.5
        loss_DB = (criterion(model.D_B(real_B), True) + criterion(model.D_B(fake_B_pool), False)) * 0.5

    opt_D_A.zero_grad(); scaler.scale(loss_DA).backward(); scaler.step(opt_D_A); scaler.update()
    opt_D_B.zero_grad(); scaler.scale(loss_DB).backward(); scaler.step(opt_D_B); scaler.update()

    return {"G": loss_G.item(), "D_A": loss_DA.item(), "D_B": loss_DB.item()}, \
           {"real_A": real_A[:4], "fake_B": fake_B[:4], "cycled_A": cycled_A[:4],
            "real_B": real_B[:4], "fake_A": fake_A[:4], "cycled_B": cycled_B[:4]}

def step_single_domain(model, batch, opts, scaler, criterion, pools, cfg):
    real = batch[0]
    t    = cfg.Training
    opt_G, opt_Dx, opt_Dy = opts

    with autocast(enabled=cfg.General.mixed_precision):
        fake      = model.G(real)
        cycled    = model.F(fake)
        fake_inv  = model.F(real)
        cycled_inv = model.G(fake_inv)

        loss_G = (criterion(model.D_style(fake), True)
                  + cycle_consistency_loss(real, cycled, t.lambda_cycle)
                  + criterion(model.D_real(fake_inv), True)
                  + cycle_consistency_loss(real, cycled_inv, t.lambda_cycle))

    opt_G.zero_grad(); scaler.scale(loss_G).backward(); scaler.step(opt_G); scaler.update()

    with autocast(enabled=cfg.General.mixed_precision):
        fake_pool = pools[0].query(fake.detach())
        loss_D  = ((criterion(model.D_style(fake_pool), False) + criterion(model.D_style(real), True)) * 0.5
                   + (criterion(model.D_real(fake_inv.detach()), False) + criterion(model.D_real(real), True)) * 0.5)

    opt_Dx.zero_grad(); opt_Dy.zero_grad()
    scaler.scale(loss_D).backward(); scaler.step(opt_Dx); scaler.step(opt_Dy); scaler.update()

    return {"G": loss_G.item(), "D": loss_D.item()}, \
           {"real": real[:4], "styled": fake[:4], "cycled": cycled[:4]}

def step_pix2pix(model, batch, opts, scaler, criterion, pools, cfg):
    real_A, real_B = batch[0], batch[1]
    t              = cfg.Training
    opt_G, opt_D   = opts

    with autocast(enabled=cfg.General.mixed_precision):
        fake_B   = model.G(real_A)
        pred_fake = model.D(fake_B, real_A)
        pred_real = model.D(real_B, real_A)
        loss_G   = criterion(pred_fake, True) + l1_reconstruction_loss(fake_B, real_B, t.get("lambda_l1", 100.0))
        loss_D   = (criterion(pred_real, True) + criterion(model.D(fake_B.detach(), real_A), False)) * 0.5

    opt_G.zero_grad(); scaler.scale(loss_G).backward(); scaler.step(opt_G); scaler.update()
    opt_D.zero_grad(); scaler.scale(loss_D).backward(); scaler.step(opt_D); scaler.update()

    return {"G": loss_G.item(), "D": loss_D.item()}, \
           {"real_A": real_A[:4], "fake_B": fake_B[:4], "real_B": real_B[:4]}
# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg  = load_config(args.config, args.options)
    cfg._yaml_path = args.config

    set_seed(cfg.General.seed)
    logger = get_logger()
    logger.info(f"Model: {cfg.General.model_name}")
    print_config(cfg)

    device, use_parallel, gpu_ids = setup_device(cfg.General.device)
    model = wrap_model(get_model(cfg), device, use_parallel)
    logger.info(f"Device: {device}  GPUs: {gpu_ids if gpu_ids else 'CPU'}")

    loader  = build_loader(cfg)
    model   = get_model(cfg).to(device)
    name    = str(cfg.General.model_name).lower()
    t       = cfg.Training
    crit    = LSGANLoss()
    scaler  = GradScaler(enabled=cfg.General.mixed_precision)
    pools   = [ImagePool(t.pool_size), ImagePool(t.pool_size)]

    run_dir  = make_run_dir(cfg)
    csv_log  = CSVLogger(os.path.join(run_dir, "metrics.csv"),
                         ["epoch", "step", "loss_G", "loss_D"])
    ckpt_mgr = CheckpointManager(run_dir, keep_k=cfg.General.keep_best_k)
    logger.info(f"Run dir: {run_dir}")

    # Build optimisers
    def _opt(params): return build_optimizer(params, t.lr_generator, t.beta1, t.beta2)
    def _opt_d(params): return build_optimizer(params, t.lr_discriminator, t.beta1, t.beta2)

    if name in ("cycle_gan", "diffaug_gan"):
        gen_params  = list(model.G_AB.parameters()) + list(model.G_BA.parameters())
        opts        = [_opt(gen_params), _opt_d(model.D_A.parameters()), _opt_d(model.D_B.parameters())]
        schedulers  = [build_scheduler(o, cfg) for o in opts]
        step_fn     = step_cycle_gan
    elif name == "single_domain_gan":
        gen_params  = list(model.G.parameters()) + list(model.F.parameters())
        opts        = [_opt(gen_params), _opt_d(model.D_real.parameters()), _opt_d(model.D_style.parameters())]
        schedulers  = [build_scheduler(o, cfg) for o in opts]
        step_fn     = step_single_domain
    elif name == "pix2pix":
        opts        = [_opt(model.G.parameters()), _opt_d(model.D.parameters())]
        schedulers  = [build_scheduler(o, cfg) for o in opts]
        step_fn     = step_pix2pix
    elif name == "vanilla_gan":
        opts        = [_opt(model.G.parameters()), _opt_d(model.D.parameters())]
        schedulers  = [build_scheduler(o, cfg) for o in opts]
        step_fn     = step_single_domain    # same structure
    else:
        # CUT / StarGAN — use cycle_gan step as default (extend as needed)
        gen_params = list(model.G_AB.parameters()) if hasattr(model, "G_AB") else list(model.G.parameters())
        opts       = [_opt(gen_params)]
        schedulers = [build_scheduler(o, cfg) for o in opts]
        step_fn    = step_cycle_gan

    ga = cfg.General.gradient_accumulation_steps

    for epoch in range(t.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{t.epochs}", ncols=110)
        epoch_losses = {}

        for step, batch in enumerate(pbar):
            if isinstance(batch[0], torch.Tensor):
                batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

            losses, samples = step_fn(model, batch, opts, scaler, crit, pools, cfg)
            epoch_losses = {k: epoch_losses.get(k, 0) + v for k, v in losses.items()}

            if step % cfg.General.log_every_n_steps == 0:
                loss_str = "  ".join(f"{k}={v:.4f}" for k, v in losses.items())
                pbar.set_postfix_str(loss_str)
                csv_log.log({"epoch": epoch+1, "step": step, **{f"loss_{k}": round(v, 4) for k, v in losses.items()}})

        # LR scheduler step
        for sched in schedulers:
            if sched: sched.step()

        # Save samples and checkpoint
        if (epoch + 1) % cfg.General.save_every_n_epochs == 0 or epoch == t.epochs - 1:
            save_sample_images(run_dir, epoch + 1, samples)
            state = {"epoch": epoch + 1, "model": model.state_dict(),
                     "opts":  [o.state_dict() for o in opts], "cfg": dict(cfg)}
            ckpt_mgr.save(state, epoch + 1)
            avg = {k: v / len(loader) for k, v in epoch_losses.items()}
            logger.info(f"Epoch {epoch+1}  " + "  ".join(f"{k}={v:.4f}" for k, v in avg.items()))

    logger.info("✅ Training complete.")

if __name__ == "__main__":
    main()