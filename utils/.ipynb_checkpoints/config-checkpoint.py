"""utils/config.py — YAML config loader with _base_ inheritance and CLI --options override."""
import os
import ast
import yaml
from pathlib import Path
from addict import Dict

import torch
import torch.nn as nn

def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def load_config(yaml_path: str, options: list = None) -> Dict:
    """
    Load YAML, merge with _base_ parent, apply CLI --options overrides.

    Args:
        yaml_path : path to model YAML  (e.g. configs/cycle_gan.yaml)
        options   : list of "Section.key=value" strings from --options

    Returns:
        addict.Dict — dot-accessible config
    """
    cfg_dir = Path(yaml_path).parent
    raw     = _load_yaml(yaml_path)

    base_name = raw.pop("_base_", None)
    if base_name:
        base_raw = _load_yaml(str(cfg_dir / base_name))
        raw = _deep_merge(base_raw, raw)

    if options:
        for opt in options:
            if "=" not in opt:
                continue
            key_path, value = opt.split("=", 1)
            keys = key_path.split(".")
            d = raw
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            target   = keys[-1]
            existing = d.get(target)
            if existing is None:
                d[target] = value
            elif isinstance(existing, bool):
                d[target] = value.lower() in ("true", "1", "yes")
            elif isinstance(existing, int):
                d[target] = int(value)
            elif isinstance(existing, float):
                d[target] = float(value)
            elif isinstance(existing, list):
                d[target] = ast.literal_eval(value)
            else:
                d[target] = value

    return Dict(raw)

def print_config(cfg: Dict, indent: int = 0) -> None:
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            print_config(Dict(v), indent + 2)
        else:
            print(" " * indent + f"{k}: {v}")

def setup_device(device_cfg) -> tuple:
    """
    Parse General.device and return (device, use_parallel, gpu_ids).

    Accepts:
      0            → single GPU cuda:0
      [0, 1, 3]   → DataParallel on GPUs 0,1,3
      "all"        → DataParallel on all available GPUs
      -1 / "cpu"   → CPU
    """
    if device_cfg == "cpu" or device_cfg == -1:
        return torch.device("cpu"), False, []

    if device_cfg == "all":
        n = torch.cuda.device_count()
        if n == 0:
            return torch.device("cpu"), False, []
        gpu_ids = list(range(n))

    elif isinstance(device_cfg, (list, tuple)):
        gpu_ids = [int(g) for g in device_cfg]

    else:
        gpu_ids = [int(device_cfg)]

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    device = torch.device("cuda:0")   # after masking, cuda:0 is always the first listed GPU
    use_parallel = len(gpu_ids) > 1
    return device, use_parallel, gpu_ids

def wrap_model(model: nn.Module, device, use_parallel: bool) -> nn.Module:
    model = model.to(device)
    if use_parallel:
        model = nn.DataParallel(model)
    return model
