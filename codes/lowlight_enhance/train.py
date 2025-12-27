"""
Training script for HDR Gaussian Splatting.
Uses TOML config file to generate and run experiments.
"""

import sys
import os
# Add gsplat examples to path BEFORE importing local modules
sys.path.insert(0, "/mnt/data0/teja/lowlight/LowLightEnhancement/codes/gsplat/examples")

import csv
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import imageio
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser
import yaml
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap

# Import utils functions from gsplat examples
# These must be imported before local utils to avoid conflicts
from utils import AppearanceOptModule, CameraOptModule

# Import from local modules
from config_loader import load_configs
from dataset import Dataset, Parser
from losses import LossComputer
from models import ExposureOptModule
from rasterization import rasterize_splats
from splats import create_splats_with_optimizers

# Import local utils functions - use explicit import to avoid conflicts
# Since gsplat/examples is in path first, we need to import local utils explicitly
import importlib.util
local_utils_file = os.path.join(os.path.dirname(__file__), "utils.py")
if os.path.exists(local_utils_file):
    spec = importlib.util.spec_from_file_location("hdr_utils", local_utils_file)
    hdr_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hdr_utils)
    aces_tonemap = hdr_utils.aces_tonemap
    get_curriculum_exposure = hdr_utils.get_curriculum_exposure
    rgb_to_sh = hdr_utils.rgb_to_sh
    set_random_seed = hdr_utils.set_random_seed
else:
    # Fallback (shouldn't happen)
    raise ImportError(f"Local utils.py not found at {local_utils_file}")


# ============================================================================
# Config Class (Converted from dataclass to dict-based)
# ============================================================================

class Config:
    """Configuration class that can be pickled for multiprocessing."""
    
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            elif isinstance(v, list) and len(v) == 1:
                # Unwrap single-item lists
                setattr(self, k, v[0])
            else:
                setattr(self, k, v)
    
    def adjust_steps(self, factor: float):
        """Adjust step-related parameters."""
        if hasattr(self, 'eval_steps') and isinstance(self.eval_steps, list):
            self.eval_steps = [int(i * factor) for i in self.eval_steps]
        if hasattr(self, 'save_steps') and isinstance(self.save_steps, list):
            self.save_steps = [int(i * factor) for i in self.save_steps]
        if hasattr(self, 'ply_steps') and isinstance(self.ply_steps, list):
            self.ply_steps = [int(i * factor) for i in self.ply_steps]
        if hasattr(self, 'max_steps'):
            self.max_steps = int(self.max_steps * factor)
        if hasattr(self, 'sh_degree_interval'):
            self.sh_degree_interval = int(self.sh_degree_interval * factor)
        
        # Adjust strategy steps
        if hasattr(self, 'strategy') and self.strategy is not None:
            strategy = self.strategy
            if isinstance(strategy, DefaultStrategy):
                strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
                strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
                strategy.reset_every = int(strategy.reset_every * factor)
                strategy.refine_every = int(strategy.refine_every * factor)
            elif isinstance(strategy, MCMCStrategy):
                strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
                strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
                strategy.refine_every = int(strategy.refine_every * factor)


def dict_to_config(cfg_dict: Dict[str, Any]) -> Config:
    """Convert dict config to Config object."""
    return Config(cfg_dict)


# ============================================================================
# Runner Class
# ============================================================================

class Runner:
    """Engine for training and testing with HDR capabilities."""
    
    def _get_attr(self, cfg, attr_name, default=None, nested_path=None):
        """Get attribute from config with nested fallback."""
        if hasattr(cfg, attr_name):
            value = getattr(cfg, attr_name)
            if value is not None:
                return value
        if nested_path:
            parts = nested_path.split('.')
            obj = cfg
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    return default
            if isinstance(obj, list) and len(obj) > 0:
                return obj[0]
            return obj
        return default
    
    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg
    ) -> None:
        set_random_seed(42 + local_rank)
        
        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        
        # Where to dump results.
        result_dir = getattr(cfg, 'result_dir', None)
        if result_dir is None:
            raise AttributeError("Config must have result_dir")
        os.makedirs(result_dir, exist_ok=True)
        
        # Setup output directories.
        self.ckpt_dir = f"{result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{result_dir}/tb")
        
        # CSV logging for losses
        self.csv_file = f"{result_dir}/losses.csv"
        self.csv_writer = None
        self.csv_file_handle = None
        # Get CSV columns from config, with default fallback
        default_csv_columns = [
            'step', 'loss', 'loss_nll', 'l1loss', 'ssimloss', 
            'loss_ratio', 'loss_sh', 'loss_albedo_reg', 'loss_illum_sparse',
            'curriculum_exposure', 'depthloss', 'tvloss',
            'exposure_mean', 'exposure_std', 'num_GS', 'mem',
            'min_GS_reached', 'pruning_protection_active'
        ]
        # Try to get csv_columns from config (handles both nested and flattened structures)
        csv_columns = None
        if hasattr(cfg, 'csv_columns'):
            csv_columns = getattr(cfg, 'csv_columns')
        elif hasattr(cfg, 'logging') and hasattr(cfg.logging, 'csv_columns'):
            csv_columns = getattr(cfg.logging, 'csv_columns')
        elif hasattr(cfg, 'logging') and isinstance(getattr(cfg, 'logging'), dict):
            logging_dict = getattr(cfg, 'logging')
            if 'csv_columns' in logging_dict:
                csv_columns = logging_dict['csv_columns']
        
        self.csv_columns = csv_columns if csv_columns is not None else default_csv_columns
        
        if world_rank == 0:
            # Initialize CSV file with headers
            self.csv_file_handle = open(self.csv_file, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file_handle)
            # Write header row from config
            self.csv_writer.writerow(self.csv_columns)
            self.csv_file_handle.flush()
        
        # Load data
        # Handle nested config structure with fallback
        data_dir = getattr(cfg, 'data_dir', None)
        if data_dir is None:
            # Try nested access
            if hasattr(cfg, 'data') and hasattr(cfg.data, 'data_dirs'):
                data_dirs = cfg.data.data_dirs
                if isinstance(data_dirs, list) and len(data_dirs) > 0:
                    data_dir = data_dirs[0]
                else:
                    data_dir = data_dirs
            else:
                raise AttributeError("Config must have data_dir or data.data_dirs")
        
        data_factor = getattr(cfg, 'data_factor', None)
        if data_factor is None:
            if hasattr(cfg, 'data') and hasattr(cfg.data, 'data_factors'):
                data_factors = cfg.data.data_factors
                if isinstance(data_factors, list) and len(data_factors) > 0:
                    data_factor = data_factors[0]
                else:
                    data_factor = data_factors
            else:
                data_factor = 1  # Default
        
        normalize_world_space = getattr(cfg, 'normalize_world_space', True)
        test_every = getattr(cfg, 'test_every', 8)
        
        self.parser = Parser(
            data_dir=data_dir,
            factor=data_factor,
            normalize=normalize_world_space,
            test_every=test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=getattr(cfg, 'patch_size', None),
            load_depths=getattr(cfg, 'depth_loss', False),
        )
        self.valset = Dataset(self.parser, split="val")
        global_scale = getattr(cfg, 'global_scale', 1.0)
        self.scene_scale = self.parser.scene_scale * 1.1 * global_scale
        print("Scene scale:", self.scene_scale)
        
        # Model
        feature_dim = 32 if getattr(cfg, 'app_opt', False) else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=getattr(cfg, 'init_type', 'sfm'),
            init_num_pts=getattr(cfg, 'init_num_pts', 100000),
            init_extent=getattr(cfg, 'init_extent', 3.0),
            init_opacity=getattr(cfg, 'init_opa', 0.1),
            init_scale=getattr(cfg, 'init_scale', 1.0),
            means_lr=getattr(cfg, 'means_lr', 1.6e-4) * 0.5,
            scales_lr=getattr(cfg, 'scales_lr', 5e-3) * 0.5,
            opacities_lr=getattr(cfg, 'opacities_lr', 5e-2) * 0.5,
            quats_lr=getattr(cfg, 'quats_lr', 1e-3) * 0.5,
            sh0_lr=getattr(cfg, 'sh0_lr', 2.5e-3) * 0.5,
            shN_lr=getattr(cfg, 'shN_lr', 1.25e-4) * 0.5,
            scene_scale=self.scene_scale,
            sh_degree=getattr(cfg, 'sh_degree', 3),
            sparse_grad=getattr(cfg, 'sparse_grad', False),
            visible_adam=getattr(cfg, 'visible_adam', False),
            batch_size=getattr(cfg, 'batch_size', 1),
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            enable_intrinsic_decomp=getattr(cfg, 'enable_intrinsic_decomp', False),
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        
        # Densification Strategy
        if not hasattr(cfg, 'strategy') or cfg.strategy is None:
            cfg.strategy = DefaultStrategy(verbose=True)
        cfg.strategy.check_sanity(self.splats, self.optimizers)
        
        if isinstance(cfg.strategy, DefaultStrategy):
            self.strategy_state = cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(cfg.strategy, MCMCStrategy):
            self.strategy_state = cfg.strategy.initialize_state()
        else:
            assert_never(cfg.strategy)
        
        # Compression Strategy
        self.compression_method = None
        if hasattr(cfg, 'compression') and cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")
        
        # Exposure optimization module
        self.exposure_module = None
        self.exposure_optimizers = []
        if getattr(cfg, 'enable_exposure_opt', False):
            exposure_lr = getattr(cfg, 'exposure_lr', 1e-3)
            batch_size = getattr(cfg, 'batch_size', 1)
            self.exposure_module = ExposureOptModule(len(self.trainset)).to(self.device)
            self.exposure_optimizers = [
                torch.optim.Adam(
                    self.exposure_module.parameters(),
                    lr=exposure_lr * math.sqrt(batch_size),
                )
            ]
            if world_size > 1:
                self.exposure_module = DDP(self.exposure_module)
        
        self.pose_optimizers = []
        if getattr(cfg, 'pose_opt', False):
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            pose_opt_lr = getattr(cfg, 'pose_opt_lr', 1e-5)
            pose_opt_reg = getattr(cfg, 'pose_opt_reg', 1e-6)
            batch_size = getattr(cfg, 'batch_size', 1)
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=pose_opt_lr * math.sqrt(batch_size),
                    weight_decay=pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)
        
        if getattr(cfg, 'pose_noise', 0.0) > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)
        
        self.app_optimizers = []
        if getattr(cfg, 'app_opt', False):
            assert feature_dim is not None
            app_embed_dim = getattr(cfg, 'app_embed_dim', 16)
            sh_degree = getattr(cfg, 'sh_degree', 3)
            app_opt_lr = getattr(cfg, 'app_opt_lr', 1e-3)
            app_opt_reg = getattr(cfg, 'app_opt_reg', 1e-6)
            batch_size = getattr(cfg, 'batch_size', 1)
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, app_embed_dim, sh_degree
            ).to(self.device)
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=app_opt_lr * math.sqrt(batch_size) * 10.0,
                    weight_decay=app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=app_opt_lr * math.sqrt(batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)
        
        self.bil_grid_optimizers = []
        if getattr(cfg, 'use_bilateral_grid', False):
            if getattr(cfg, 'use_fused_bilagrid', False):
                from fused_bilagrid import (
                    BilateralGrid,
                    color_correct,
                    slice,
                    total_variation_loss,
                )
            else:
                from lib_bilagrid import (
                    BilateralGrid,
                    color_correct,
                    slice,
                    total_variation_loss,
                )
            bilateral_grid_shape = getattr(cfg, 'bilateral_grid_shape', (16, 16, 8))
            batch_size = getattr(cfg, 'batch_size', 1)
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=bilateral_grid_shape[0],
                grid_Y=bilateral_grid_shape[1],
                grid_W=bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(batch_size),
                    eps=1e-15,
                ),
            ]
        
        # Losses & Metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        
        # Loss computer
        self.loss_computer = LossComputer(cfg, self.device)
        
        # Opacity pruning protection tracking
        self.min_gs_count_reached = float('inf')
        self.pruning_protection_active_until = 0
        
        lpips_net = getattr(cfg, 'lpips_net', 'alex')
        if lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif lpips_net == "vgg":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {lpips_net}")
        
        # Viewer
        if not getattr(cfg, 'disable_viewer', False):
            self.server = viser.ViserServer(port=getattr(cfg, 'port', 8080), verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(result_dir),
                mode="training",
            )
    
    def rasterize_splats_wrapper(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        apply_exposure: bool = True,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Wrapper for rasterize_splats that uses instance variables."""
        return rasterize_splats(
            splats=self.splats,
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            cfg=self.cfg,
            app_module=getattr(self, 'app_module', None),
            exposure_module=self.exposure_module,
            world_size=self.world_size,
            masks=masks,
            rasterize_mode=rasterize_mode,
            camera_model=camera_model,
            apply_exposure=apply_exposure,
            **kwargs,
        )
    
    def _get_csv_value(self, column_name: str, step: int, loss: float, loss_dict: Dict, 
                       target_exposure: Optional[float], num_gs: int, mem: float,
                       curriculum_enabled: bool, world_size: int) -> Optional[float]:
        """Get value for a CSV column, returning None (which will be written as empty/NaN) if not available."""
        cfg = self.cfg
        
        if column_name == 'step':
            return float(step)
        elif column_name == 'loss':
            return loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        elif column_name == 'loss_nll':
            val = loss_dict.get('loss_nll')
            return val.item() if val is not None and isinstance(val, torch.Tensor) else None
        elif column_name == 'l1loss':
            val = loss_dict.get('l1loss')
            return val.item() if val is not None and isinstance(val, torch.Tensor) else None
        elif column_name == 'ssimloss':
            val = loss_dict.get('ssimloss')
            return val.item() if val is not None and isinstance(val, torch.Tensor) else None
        elif column_name == 'loss_ratio':
            val = loss_dict.get('loss_ratio')
            return val.item() if val is not None and isinstance(val, torch.Tensor) else None
        elif column_name == 'loss_sh':
            val = loss_dict.get('loss_sh')
            return val.item() if val is not None and isinstance(val, torch.Tensor) else None
        elif column_name == 'loss_albedo_reg':
            val = loss_dict.get('loss_albedo_reg')
            return val.item() if val is not None and isinstance(val, torch.Tensor) else None
        elif column_name == 'loss_illum_sparse':
            val = loss_dict.get('loss_illum_sparse')
            return val.item() if val is not None and isinstance(val, torch.Tensor) else None
        elif column_name == 'curriculum_exposure':
            return target_exposure if curriculum_enabled and target_exposure is not None else None
        elif column_name == 'depthloss':
            val = loss_dict.get('depthloss')
            return val.item() if val is not None and isinstance(val, torch.Tensor) else None
        elif column_name == 'tvloss':
            val = loss_dict.get('tvloss')
            return val.item() if val is not None and isinstance(val, torch.Tensor) else None
        elif column_name == 'exposure_mean':
            if getattr(cfg, 'enable_exposure_opt', False) and self.exposure_module is not None:
                if world_size > 1:
                    exposures = self.exposure_module.module.exposures.data
                else:
                    exposures = self.exposure_module.exposures.data
                return exposures.mean().item()
            return None
        elif column_name == 'exposure_std':
            if getattr(cfg, 'enable_exposure_opt', False) and self.exposure_module is not None:
                if world_size > 1:
                    exposures = self.exposure_module.module.exposures.data
                else:
                    exposures = self.exposure_module.exposures.data
                return exposures.std().item()
            return None
        elif column_name == 'num_GS':
            return float(num_gs)
        elif column_name == 'mem':
            return mem
        elif column_name == 'min_GS_reached':
            return float(self.min_gs_count_reached) if self.min_gs_count_reached != float('inf') else None
        elif column_name == 'pruning_protection_active':
            min_gs_count = getattr(cfg, 'min_gs_count', 500)
            return 1.0 if num_gs < min_gs_count else 0.0
        else:
            # Unknown column, return None
            return None
    
    def train(self):
        """Main training loop."""
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        
        # Dump cfg
        result_dir = getattr(cfg, 'result_dir', 'results/default')
        if world_rank == 0:
            with open(f"{result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg) if hasattr(cfg, '__dict__') else cfg, f)
        
        max_steps = getattr(cfg, 'max_steps', 30000)
        init_step = 0
        
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if getattr(cfg, 'pose_opt', False):
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if getattr(cfg, 'enable_exposure_opt', False):
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.exposure_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if getattr(cfg, 'use_bilateral_grid', False):
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )
        
        batch_size = getattr(cfg, 'batch_size', 1)
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)
        
        # Training loop
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not getattr(cfg, 'disable_viewer', False):
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()
            
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
            
            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device)
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None
            if getattr(cfg, 'depth_loss', False):
                points = data["points"].to(device)
                depths_gt = data["depths"].to(device)
            
            height, width = pixels.shape[1:3]
            
            if getattr(cfg, 'pose_noise', 0.0) > 0.0:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)
            
            if getattr(cfg, 'pose_opt', False):
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)
            
            # SH schedule
            sh_degree_interval = getattr(cfg, 'sh_degree_interval', 1000)
            sh_degree = getattr(cfg, 'sh_degree', 3)
            sh_degree_to_use = min(step // sh_degree_interval, sh_degree)
            
            # Curriculum learning
            target_exposure = None
            curriculum_enabled = getattr(cfg, 'curriculum_enabled', False)
            if curriculum_enabled:
                curriculum_max_exposure = getattr(cfg, 'curriculum_max_exposure', 10.0)
                target_exposure = get_curriculum_exposure(step, max_steps, curriculum_max_exposure)
            else:
                enable_synthetic_exposures = getattr(cfg, 'enable_synthetic_exposures', False)
                target_exposure = 1.0 if enable_synthetic_exposures else None
            
            # Multi-exposure synthesis
            pixels_target = pixels
            enable_synthetic_exposures = getattr(cfg, 'enable_synthetic_exposures', False)
            if enable_synthetic_exposures and target_exposure is not None:
                dark_base_exposure = 0.1
                hdr_gt = pixels / dark_base_exposure
                curriculum_max_exposure = getattr(cfg, 'curriculum_max_exposure', 10.0)
                curriculum_progress = min(target_exposure / curriculum_max_exposure, 1.0) if curriculum_enabled else 1.0
                pixels_target = pixels * (1.0 - curriculum_progress) + hdr_gt * curriculum_progress
            
            # Verification at step 0
            if step == 0 and world_rank == 0:
                print("=== TRAINING SETUP VERIFICATION ===")
                print(f"GT image brightness: {pixels.mean().item():.6f}")
                print(f"Target exposure: {target_exposure}")
                print(f"Target brightness: {pixels_target.mean().item():.6f}")
                if pixels.mean().item() > 0:
                    print(f"Ratio: {pixels_target.mean().item() / pixels.mean().item():.2f}")
                if target_exposure is not None:
                    print(f"Expected ratio: {target_exposure:.2f}")
                print("===================================")
            
            # Forward pass
            exposure_value_kwarg = {}
            enable_intrinsic_decomp = getattr(cfg, 'enable_intrinsic_decomp', False)
            if enable_intrinsic_decomp and target_exposure is not None:
                exposure_value_kwarg["exposure_value"] = torch.tensor(
                    [target_exposure], device=device, dtype=torch.float32
                )
            
            apply_exposure_flag = not (enable_intrinsic_decomp and target_exposure is not None)
            
            near_plane = getattr(cfg, 'near_plane', 0.01)
            far_plane = getattr(cfg, 'far_plane', 1e10)
            renders, alphas, info = self.rasterize_splats_wrapper(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=near_plane,
                far_plane=far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if getattr(cfg, 'depth_loss', False) else "RGB",
                masks=masks,
                apply_exposure=apply_exposure_flag,
                **exposure_value_kwarg,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None
            
            if getattr(cfg, 'use_bilateral_grid', False):
                from lib_bilagrid import slice
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]
            
            if getattr(cfg, 'random_bkgd', False):
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)
            
            cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )
            
            # Loss computation
            pixels_loss_target = pixels_target if enable_synthetic_exposures else pixels
            
            # Use loss computer
            depths_gt_for_loss = None
            if getattr(cfg, 'depth_loss', False) and depths is not None:
                # Pass points and depths as tuple
                depths_gt_for_loss = (points, depths_gt)
            
            loss, loss_dict = self.loss_computer.compute_total_loss(
                colors=colors,
                pixels_target=pixels_loss_target,
                splats=self.splats,
                trainset=self.trainset,
                rasterize_fn=self.rasterize_splats_wrapper,
                exposure_module=self.exposure_module,
                image_ids=image_ids,
                pixels=pixels,
                depths=depths,
                depths_gt=depths_gt_for_loss,
                width=width,
                height=height,
                scene_scale=self.scene_scale,
                bil_grids=getattr(self, 'bil_grids', None),
                step=step,
                sh_degree_to_use=sh_degree_to_use,
                target_exposure=target_exposure,
                world_size=world_size,
            )
            
            loss.backward()
            
            # Gradient clipping
            use_nll_loss = getattr(cfg, 'use_nll_loss', False)
            if use_nll_loss or getattr(cfg, 'use_log_space_loss', False):
                torch.nn.utils.clip_grad_norm_(self.splats.parameters(), max_norm=5.0)
            
            desc = f"loss={loss.item():.3f}| sh degree={sh_degree_to_use}| "
            if getattr(cfg, 'depth_loss', False) and loss_dict.get('depthloss') is not None:
                desc += f"depth loss={loss_dict['depthloss'].item():.6f}| "
            pbar.set_description(desc)
            
            # Logging
            tb_every = getattr(cfg, 'tb_every', 100)
            if world_rank == 0 and tb_every > 0 and step % tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                num_gs = len(self.splats["means"])
                self.writer.add_scalar("train/loss", loss.item(), step)
                if use_nll_loss:
                    if loss_dict.get('loss_nll') is not None:
                        self.writer.add_scalar("train/nll_loss", loss_dict['loss_nll'].item(), step)
                else:
                    if loss_dict.get('l1loss') is not None:
                        self.writer.add_scalar("train/l1loss", loss_dict['l1loss'].item(), step)
                    if loss_dict.get('ssimloss') is not None:
                        self.writer.add_scalar("train/ssimloss", loss_dict['ssimloss'].item(), step)
                self.writer.add_scalar("train/num_GS", num_gs, step)
                self.writer.add_scalar("train/mem", mem, step)
                
                if cfg.enable_exposure_opt:
                    if world_size > 1:
                        exposures = self.exposure_module.module.exposures.data
                    else:
                        exposures = self.exposure_module.exposures.data
                    self.writer.add_scalar("train/exposure_mean", exposures.mean().item(), step)
                    self.writer.add_scalar("train/exposure_std", exposures.std().item(), step)
                
                if curriculum_enabled and target_exposure is not None:
                    self.writer.add_scalar("train/curriculum_exposure", target_exposure, step)
                
                self.writer.flush()
                
                # Log to CSV
                if self.csv_writer is not None:
                    row = []
                    for col_name in self.csv_columns:
                        value = self._get_csv_value(
                            column_name=col_name,
                            step=step,
                            loss=loss,
                            loss_dict=loss_dict,
                            target_exposure=target_exposure,
                            num_gs=num_gs,
                            mem=mem,
                            curriculum_enabled=curriculum_enabled,
                            world_size=world_size
                        )
                        # Write NaN string for None values (CSV standard)
                        row.append('nan' if value is None else value)
                    self.csv_writer.writerow(row)
                    self.csv_file_handle.flush()
            
            # Save checkpoint
            save_steps = getattr(cfg, 'save_steps', [7000, 30000])
            if step in [i - 1 for i in save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if getattr(cfg, 'enable_exposure_opt', False):
                    if world_size > 1:
                        data["exposure_module"] = self.exposure_module.module.state_dict()
                    else:
                        data["exposure_module"] = self.exposure_module.state_dict()
                if getattr(cfg, 'pose_opt', False):
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if getattr(cfg, 'app_opt', False):
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            
            # Save PLY
            if (
                step in [i - 1 for i in getattr(cfg, 'ply_steps', [])] or step == max_steps - 1
            ) and getattr(cfg, 'save_ply', False):
                if getattr(cfg, 'app_opt', False):
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                elif enable_intrinsic_decomp:
                    albedo_sh0 = self.splats["albedo_sh0"]
                    albedo_shN = self.splats["albedo_shN"]
                    illum_sh0 = self.splats["illum_sh0"]
                    illum_shN = self.splats["illum_shN"]
                    sh0 = albedo_sh0 + illum_sh0 * 1.0
                    shN = albedo_shN + illum_shN * 1.0
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]
                
                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )
            
            # Sparse gradients
            if getattr(cfg, 'sparse_grad', False):
                assert getattr(cfg, 'packed', False), "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],
                        values=grad[gaussian_ids],
                        size=self.splats[k].size(),
                        is_coalesced=len(Ks) == 1,
                    )
            
            if getattr(cfg, 'visible_adam', False):
                gaussian_cnt = self.splats.means.shape[0]
                if getattr(cfg, 'packed', False):
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)
            
            # Optimize
            for optimizer in self.optimizers.values():
                if getattr(cfg, 'visible_adam', False):
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.exposure_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()
            
            # Opacity clamping
            if getattr(cfg, 'enable_opacity_clamping', False) and getattr(cfg, 'min_opacity_threshold', 0) > 0:
                with torch.no_grad():
                    opacities_sigmoid = torch.sigmoid(self.splats["opacities"])
                    opacities_sigmoid = torch.clamp(opacities_sigmoid, min=cfg.min_opacity_threshold)
                    self.splats["opacities"].data = torch.logit(opacities_sigmoid)
            
            # Track GS count
            num_gs_before = len(self.splats["means"])
            if num_gs_before == 0:
                if world_rank == 0:
                    print(f"ERROR: GS count reached 0 at step {step}. Stopping training.")
                raise RuntimeError(f"GS count reached 0 at step {step}.")
            
            if num_gs_before < self.min_gs_count_reached:
                self.min_gs_count_reached = num_gs_before
                if world_rank == 0:
                    print(f"Warning: GS count dropped to {num_gs_before} at step {step}")
            
            if num_gs_before < cfg.min_gs_count:
                if self.pruning_protection_active_until < step:
                    self.pruning_protection_active_until = step + cfg.pruning_protection_steps
                    if world_rank == 0:
                        print(f"Pruning protection activated: GS count {num_gs_before} < {cfg.min_gs_count}.")
            
            # Post-backward steps
            should_prune = (num_gs_before >= cfg.min_gs_count or step >= self.pruning_protection_active_until)
            skip_densification = (num_gs_before < 50)
            
            if not skip_densification and should_prune:
                if isinstance(cfg.strategy, DefaultStrategy):
                    cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        packed=getattr(cfg, 'packed', False),
                    )
                elif isinstance(cfg.strategy, MCMCStrategy):
                    cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        lr=schedulers[0].get_last_lr()[0],
                    )
            
            num_gs_after = len(self.splats["means"])
            if num_gs_after == 0:
                if world_rank == 0:
                    print(f"ERROR: GS count reached 0 after pruning at step {step}.")
                raise RuntimeError(f"GS count reached 0 after pruning at step {step}.")
            
            # Evaluation
            eval_steps = getattr(cfg, 'eval_steps', [7000, 30000])
            if step in [i - 1 for i in eval_steps]:
                self.eval(step)
                self.render_traj(step)
            
            # Compression
            if getattr(cfg, 'compression', None) is not None and step in [i - 1 for i in eval_steps]:
                self.run_compression(step=step)
            
            if not getattr(cfg, 'disable_viewer', False):
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                self.viewer.update(step, num_train_rays_per_step)
        
        # Close CSV and plot losses
        if world_rank == 0 and self.csv_file_handle is not None:
            self.csv_file_handle.close()
            self.plot_losses()
    
    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Evaluation entry point."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]
            image_ids = data["image_id"].to(device)
            
            torch.cuda.synchronize()
            tic = time.time()
            sh_degree = getattr(cfg, 'sh_degree', 3)
            near_plane = getattr(cfg, 'near_plane', 0.01)
            far_plane = getattr(cfg, 'far_plane', 1e10)
            colors, _, _ = self.rasterize_splats_wrapper(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree,
                near_plane=near_plane,
                far_plane=far_plane,
                masks=masks,
                image_ids=image_ids,
                apply_exposure=True,
            )
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)
            
            colors = torch.clamp(colors, 0.0, 1.0)
            
            virtual_gain = getattr(cfg, 'virtual_gain', 50.0)
            if virtual_gain > 1.0:
                hdr_colors, _, _ = self.rasterize_splats_wrapper(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    masks=masks,
                    image_ids=image_ids,
                    apply_exposure=False,
                )
                hdr_colors = hdr_colors * virtual_gain
                tonemapped = aces_tonemap(hdr_colors)
                canvas_list = [pixels, colors, tonemapped]
            else:
                canvas_list = [pixels, colors]
            
            if world_rank == 0:
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )
                
                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors.permute(0, 3, 1, 2)
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
        
        if world_rank == 0:
            ellipse_time /= len(valloader)
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update({
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
            })
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()
    
    @torch.no_grad()
    def render_traj(self, step: int):
        """Render trajectory video."""
        if getattr(self.cfg, 'disable_video', False):
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device
        
        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if getattr(cfg, 'render_traj_path', 'interp') == "interp":
            camtoworlds_all = generate_interpolated_path(camtoworlds_all, 1)
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(camtoworlds_all, height=height)
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        
        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )
        
        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]
        
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]
            
            sh_degree = getattr(cfg, 'sh_degree', 3)
            near_plane = getattr(cfg, 'near_plane', 0.01)
            far_plane = getattr(cfg, 'far_plane', 1e10)
            virtual_gain = getattr(cfg, 'virtual_gain', 50.0)
            renders, _, _ = self.rasterize_splats_wrapper(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree,
                near_plane=near_plane,
                far_plane=far_plane,
                render_mode="RGB+ED",
                apply_exposure=False,
            )
            
            hdr_colors = renders[..., 0:3] * virtual_gain
            colors = aces_tonemap(hdr_colors)
            colors = torch.clamp(colors, 0.0, 1.0)
            
            depths = renders[..., 3:4]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]
            
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")
    
    @torch.no_grad()
    def run_compression(self, step: int):
        """Run compression."""
        print("Running compression...")
        world_rank = self.world_rank
        cfg = self.cfg
        
        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)
        
        self.compression_method.compress(compress_dir, self.splats)
        
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")
    
    def plot_losses(self):
        """Plot losses from CSV."""
        if self.world_rank != 0 or not os.path.exists(self.csv_file):
            return
        
        print("Plotting losses...")
        steps = []
        losses = defaultdict(list)
        
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Convert step from float string to int (handles '0.0' -> 0)
                    step = int(float(row['step']))
                    steps.append(step)
                except (ValueError, TypeError, KeyError):
                    # Skip invalid rows
                    continue
                
                for key in row.keys():
                    if key != 'step' and row[key] and row[key] not in ['None', 'nan', '']:
                        try:
                            losses[key].append(float(row[key]))
                        except (ValueError, TypeError):
                            losses[key].append(None)
                    else:
                        losses[key].append(None)
        
        if not steps:
            return
        
        # Create plots
        num_losses = len([k for k in losses.keys() if k not in ['num_GS', 'mem']])
        num_cols = 3
        num_rows = (num_losses + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        plot_idx = 0
        loss_names = {
            'loss': 'Total Loss',
            'loss_nll': 'NLL Loss',
            'l1loss': 'L1 Loss',
            'ssimloss': 'SSIM Loss',
            'loss_ratio': 'Ratio Loss',
            'loss_sh': 'SH Regularization',
        }
        
        for key in ['loss', 'loss_nll', 'l1loss', 'ssimloss', 'loss_ratio', 'loss_sh']:
            if key in losses and any(v is not None for v in losses[key]):
                ax = axes[plot_idx]
                values = [v if v is not None else float('nan') for v in losses[key]]
                ax.plot(steps, values, linewidth=1.5)
                ax.set_xlabel('Step')
                ax.set_ylabel(loss_names.get(key, key))
                ax.set_title(loss_names.get(key, key))
                ax.grid(True, alpha=0.3)
                plot_idx += 1
        
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        result_dir = getattr(self.cfg, 'result_dir', 'results/default')
        plot_path = f"{result_dir}/losses_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved to {plot_path}")
    
    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        """Viewer render function."""
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)
        
        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }
        
        sh_degree = getattr(self.cfg, 'sh_degree', 3)
        render_colors, render_alphas, info = self.rasterize_splats_wrapper(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device) / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            apply_exposure=True,
        )
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()
        
        if render_tab_state.render_mode == "rgb":
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


# ============================================================================
# Main Functions
# ============================================================================

def setup_directories(cfg):
    """Setup result directories."""
    result_dir = getattr(cfg, 'result_dir', 'results/default')
    os.makedirs(result_dir, exist_ok=True)


def run_training(cfg_dict: Dict[str, Any]):
    """Run training for a single config."""
    # Convert dict to config object
    cfg = dict_to_config(cfg_dict)
    
    # Set strategy if not set - use MCMCStrategy by default for HDR training
    if not hasattr(cfg, 'strategy') or cfg.strategy is None:
        from gsplat.strategy import MCMCStrategy
        cfg.strategy = MCMCStrategy(verbose=True)
    
    # Set earlier densification stop for training stability
    if hasattr(cfg, 'strategy') and cfg.strategy is not None:
        cfg.strategy.refine_stop_iter = 15000
    
    # Adjust steps if needed
    if hasattr(cfg, 'steps_scaler') and cfg.steps_scaler != 1.0:
        cfg.adjust_steps(cfg.steps_scaler)
    
    # Setup directories
    setup_directories(cfg)
    
    # Run training using distributed CLI
    from gsplat.distributed import cli
    cli(main, cfg, verbose=True)


def main(local_rank: int, world_rank, world_size: int, cfg):
    """Main function called by distributed CLI."""
    if world_size > 1 and not getattr(cfg, 'disable_viewer', False):
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")
    
    runner = Runner(local_rank, world_rank, world_size, cfg)
    runner.train()
    
    # Clean up distributed training if needed
    if world_size > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    
    # Only keep viewer running if explicitly enabled (not in distributed mode)
    if not getattr(cfg, 'disable_viewer', False) and world_size == 1:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        try:
            time.sleep(1000000)
        except KeyboardInterrupt:
            print("\nShutting down viewer...")
    else:
        if world_rank == 0:
            print("Training completed successfully.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train HDR Gaussian Splatting")
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--experiment_idx",
        type=int,
        default=None,
        help="Run specific experiment index (default: run all)",
    )
    args = parser.parse_args()
    
    # Load configs
    configs = load_configs(args.config)
    
    print(f"Loaded {len(configs)} experiment configurations")
    
    if args.experiment_idx is not None:
        # Run single experiment
        if args.experiment_idx < 0 or args.experiment_idx >= len(configs):
            print(f"Error: experiment_idx {args.experiment_idx} out of range [0, {len(configs)-1}]")
            exit(1)
        print(f"Running experiment {args.experiment_idx}: {configs[args.experiment_idx]['experiment_name']}")
        run_training(configs[args.experiment_idx])
    else:
        # Run all experiments
        for idx, cfg_dict in enumerate(configs):
            print(f"\n{'='*80}")
            print(f"Running experiment {idx+1}/{len(configs)}: {cfg_dict['experiment_name']}")
            print(f"{'='*80}\n")
            run_training(cfg_dict)

