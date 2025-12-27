"""
Inference script for HDR Gaussian Splatting.
Uses TOML config file and checkpoint to run evaluation.
"""

import sys
import os
sys.path.append("/mnt/data0/teja/lowlight/LowLightEnhancement/codes/gsplat/examples")

import argparse
import os
from typing import Any, Dict, List

import imageio
import numpy as np
import torch
import tqdm
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from torch.utils.data import DataLoader

import importlib.util
import os

from config_loader import dict_to_config, load_configs
from dataset import Dataset, Parser
from rasterization import rasterize_splats
from splats import create_splats_with_optimizers
from train import Runner

# Import from local utils.py (not gsplat examples utils)
_local_utils_file = os.path.join(os.path.dirname(__file__), "utils.py")
if os.path.exists(_local_utils_file):
    spec = importlib.util.spec_from_file_location("hdr_utils", _local_utils_file)
    hdr_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hdr_utils)
    aces_tonemap = hdr_utils.aces_tonemap
else:
    raise ImportError(f"Local utils.py not found at {_local_utils_file}")


def load_checkpoint(runner: Runner, ckpt_path: str, world_size: int = 1):
    """Load checkpoint into runner.
    
    Args:
        runner: Runner instance
        ckpt_path: Path to checkpoint file or directory
        world_size: World size for distributed training
    """
    if os.path.isdir(ckpt_path):
        # Load all checkpoints from directory
        import glob
        ckpt_files = sorted(glob.glob(os.path.join(ckpt_path, "ckpt_*_rank*.pt")))
        if not ckpt_files:
            raise ValueError(f"No checkpoint files found in {ckpt_path}")
        ckpts = [
            torch.load(f, map_location=runner.device, weights_only=True)
            for f in ckpt_files
        ]
        # Combine splats from all ranks
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
    else:
        # Load single checkpoint file
        ckpt = torch.load(ckpt_path, map_location=runner.device, weights_only=True)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        step = ckpt["step"]
        
        # Load exposure module if present
        if runner.cfg.enable_exposure_opt and "exposure_module" in ckpt:
            if world_size > 1:
                runner.exposure_module.module.load_state_dict(ckpt["exposure_module"])
            else:
                runner.exposure_module.load_state_dict(ckpt["exposure_module"])
    
    return step


def run_inference(cfg_dict: Dict[str, Any], ckpt_path: str):
    """Run inference for a single config.
    
    Args:
        cfg_dict: Configuration dictionary
        ckpt_path: Path to checkpoint file or directory
    """
    # Convert dict to config object
    cfg = dict_to_config(cfg_dict)
    
    # Set strategy if not set
    if not hasattr(cfg, 'strategy') or cfg.strategy is None:
        from gsplat.strategy import MCMCStrategy
        cfg.strategy = MCMCStrategy(verbose=True)
    
    # Setup directories
    os.makedirs(cfg.result_dir, exist_ok=True)
    
    # Create runner (single GPU for inference)
    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    
    # Load checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    step = load_checkpoint(runner, ckpt_path, world_size=1)
    print(f"Loaded checkpoint from step {step}")
    
    # Run evaluation
    runner.eval(step=step)
    
    # Render trajectory
    runner.render_traj(step=step)
    
    # Run compression if enabled
    if hasattr(cfg, 'compression') and cfg.compression is not None:
        runner.run_compression(step=step)
    
    print(f"Inference complete. Results saved to {cfg.result_dir}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run inference for HDR Gaussian Splatting")
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint file or directory",
    )
    parser.add_argument(
        "--experiment_idx",
        type=int,
        default=0,
        help="Experiment index to use (default: 0)",
    )
    args = parser.parse_args()
    
    # Load configs
    configs = load_configs(args.config)
    
    if args.experiment_idx < 0 or args.experiment_idx >= len(configs):
        print(f"Error: experiment_idx {args.experiment_idx} out of range [0, {len(configs)-1}]")
        exit(1)
    
    cfg_dict = configs[args.experiment_idx]
    print(f"Running inference for experiment: {cfg_dict['experiment_name']}")
    print(f"Using checkpoint: {args.ckpt}")
    
    run_inference(cfg_dict, args.ckpt)


if __name__ == "__main__":
    main()

