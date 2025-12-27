"""
Loss computation class for HDR Gaussian Splatting training.
"""

import importlib.util
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from fused_ssim import fused_ssim

# Import from local utils.py (not gsplat examples utils)
_local_utils_file = os.path.join(os.path.dirname(__file__), "utils.py")
if os.path.exists(_local_utils_file):
    spec = importlib.util.spec_from_file_location("hdr_utils", _local_utils_file)
    hdr_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hdr_utils)
    get_curriculum_exposure = hdr_utils.get_curriculum_exposure
else:
    raise ImportError(f"Local utils.py not found at {_local_utils_file}")


def poisson_gaussian_nll_loss(rendered: Tensor, gt: Tensor, eps: float = 1e-3) -> Tensor:
    """Poisson-Gaussian NLL loss with stop-gradient for stability.
    
    Args:
        rendered: Rendered image tensor
        gt: Ground truth image tensor
        eps: Small epsilon for numerical stability (noise floor)
        
    Returns:
        NLL loss value
    """
    # variance = detaching the rendered values to stop the 'cheat'
    # we add a noise floor (eps) to prevent log(0)
    # Clamp variance to prevent extreme values that could destabilize training
    variance = rendered.detach().clamp(min=1e-2, max=10.0) + eps  # detach() acts as stop_gradient
    
    # 1. The residual term (MSE weighted by signal)
    residual = ((rendered - gt) ** 2) / (2 * variance)
    
    # 2. The log term (This is what makes it go negative)
    # Weight log term less to reduce its impact
    log_term = torch.log(variance)
    
    return torch.mean(residual + 0.1 * log_term)


class LossComputer:
    """Loss computation class for HDR Gaussian Splatting."""
    
    def __init__(self, cfg, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
    
    def compute_nll_loss(self, rendered: Tensor, gt: Tensor, eps: float = 1e-3) -> Tensor:
        """Compute Poisson-Gaussian NLL loss."""
        return poisson_gaussian_nll_loss(rendered, gt, eps)
    
    def compute_l1_loss(self, rendered: Tensor, gt: Tensor) -> Tensor:
        """Compute L1 loss."""
        return F.l1_loss(rendered, gt)
    
    def compute_ssim_loss(self, rendered: Tensor, gt: Tensor) -> Tensor:
        """Compute SSIM loss."""
        return 1.0 - fused_ssim(
            rendered.permute(0, 3, 1, 2), 
            gt.permute(0, 3, 1, 2), 
            padding="valid"
        )
    
    def compute_ratio_loss(
        self,
        splats: torch.nn.ParameterDict,
        trainset,
        rasterize_fn,
        device: str,
        step: int,
        sh_degree_to_use: int,
        target_exposure: Optional[float] = None,
    ) -> Optional[Tensor]:
        """Compute exposure ratio consistency loss.
        
        Args:
            splats: Gaussian splat parameters
            trainset: Training dataset
            rasterize_fn: Function to rasterize splats
            device: Device to compute on
            step: Current training step
            sh_degree_to_use: SH degree to use
            target_exposure: Target exposure value
            
        Returns:
            Ratio loss tensor or None if skipped
        """
        enable_ratio_loss = getattr(self.cfg, 'enable_ratio_loss', False)
        if not enable_ratio_loss:
            return None
        
        ratio_loss_freq = getattr(self.cfg, 'ratio_loss_freq', 10)
        if step % ratio_loss_freq != 0 or len(trainset) < 2:
            return None
        
        try:
            # Sample 2 random cameras from the dataset
            indices = np.random.choice(len(trainset), size=2, replace=False)
            ratio_data = [trainset[int(idx)] for idx in indices]
            
            # Get GT brightness ratio (this is ground truth)
            img_i_gt = ratio_data[0]["image"].to(device)
            img_j_gt = ratio_data[1]["image"].to(device)
            brightness_i = img_i_gt.mean()
            brightness_j = img_j_gt.mean()
            gt_brightness_ratio = brightness_i / (brightness_j + 1e-6)
            
            # Render both at the SAME target exposure
            ratio_target_exposure = target_exposure if target_exposure is not None else 1.0
            
            radiances = []
            enable_intrinsic_decomp = getattr(self.cfg, 'enable_intrinsic_decomp', False)
            near_plane = getattr(self.cfg, 'near_plane', 0.01)
            far_plane = getattr(self.cfg, 'far_plane', 1e10)
            for rd in ratio_data:
                c2w = rd["camtoworld"].unsqueeze(0).to(device)
                K = rd["K"].unsqueeze(0).to(device)
                h, w = rd["image"].shape[:2]
                
                # Render at fixed exposure
                exposure_kwarg = {}
                if enable_intrinsic_decomp:
                    exposure_kwarg["exposure_value"] = torch.tensor(
                        [ratio_target_exposure], device=device, dtype=torch.float32
                    )
                
                rad, _, _ = rasterize_fn(
                    camtoworlds=c2w,
                    Ks=K,
                    width=w,
                    height=h,
                    sh_degree=sh_degree_to_use,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    apply_exposure=True,  # Use exposure when rendering both images for ratio loss
                    **exposure_kwarg,
                )
                radiances.append(rad[..., 0:3])
            
            # HDR radiances should have the SAME brightness ratio as GT
            R_i, R_j = radiances[0], radiances[1]
            rendered_brightness_i = R_i.mean()
            rendered_brightness_j = R_j.mean()
            rendered_ratio = rendered_brightness_i / (rendered_brightness_j + 1e-6)
            
            # Ratio loss: rendered ratio should match GT ratio
            loss_ratio = F.l1_loss(rendered_ratio, gt_brightness_ratio)
            return loss_ratio
        except Exception as e:
            # Skip ratio loss if there's an error
            return None
    
    def compute_sh_regularization(self, splats: torch.nn.ParameterDict) -> Optional[Tensor]:
        """Compute SH regularization loss."""
        sh_reg_lambda = getattr(self.cfg, 'sh_reg_lambda', 0.01)
        if sh_reg_lambda <= 0:
            return None
        
        enable_intrinsic_decomp = getattr(self.cfg, 'enable_intrinsic_decomp', False)
        if enable_intrinsic_decomp:
            # For intrinsic decomposition, regularize illumination SH
            loss_sh = (splats["illum_shN"] ** 2).mean()
        else:
            loss_sh = (splats["shN"] ** 2).mean()
        
        return loss_sh
    
    def compute_albedo_regularization(self, splats: torch.nn.ParameterDict) -> Optional[Tensor]:
        """Compute albedo smoothness regularization."""
        enable_intrinsic_decomp = getattr(self.cfg, 'enable_intrinsic_decomp', False)
        albedo_reg_lambda = getattr(self.cfg, 'albedo_reg_lambda', 0.01)
        if not enable_intrinsic_decomp or albedo_reg_lambda <= 0:
            return None
        
        # Albedo smoothness: Encourage exposure-invariant albedo
        loss_albedo_reg = splats["albedo_sh0"].var() * albedo_reg_lambda
        return loss_albedo_reg
    
    def compute_illumination_sparsity(self, splats: torch.nn.ParameterDict) -> Optional[Tensor]:
        """Compute illumination sparsity regularization."""
        enable_intrinsic_decomp = getattr(self.cfg, 'enable_intrinsic_decomp', False)
        illum_reg_lambda = getattr(self.cfg, 'illum_reg_lambda', 0.01)
        if not enable_intrinsic_decomp or illum_reg_lambda <= 0:
            return None
        
        # Illumination sparsity: Encourage sparse illumination
        loss_illum_sparse = splats["illum_shN"].abs().mean() * illum_reg_lambda
        return loss_illum_sparse
    
    def compute_exposure_diversity(
        self, 
        exposure_module, 
        image_ids: Tensor, 
        pixels: Tensor,
        world_size: int = 1
    ) -> Optional[Tensor]:
        """Compute exposure diversity loss."""
        enable_exposure_opt = getattr(self.cfg, 'enable_exposure_opt', False)
        if not enable_exposure_opt or exposure_module is None:
            return None
        
        if world_size > 1:
            learned_exposures = exposure_module.module.exposures
        else:
            learned_exposures = exposure_module.exposures
        
        # Encourage diversity: exposures shouldn't all be 1.0
        exp_std = learned_exposures.std()
        loss_exp_diversity = -0.1 * exp_std  # Negative because we want HIGH std
        
        # Also encourage learned exposures to match image brightness
        batch_brightness = pixels.mean()
        learned_exp_batch = learned_exposures[image_ids].mean()
        target_exp = 1.0 / (batch_brightness + 1e-3)  # Bright images need low exp
        loss_exp_align = F.l1_loss(learned_exp_batch, target_exp)
        
        return loss_exp_diversity + 0.01 * loss_exp_align
    
    def compute_total_loss(
        self,
        colors: Tensor,
        pixels_target: Tensor,
        splats: torch.nn.ParameterDict,
        trainset=None,
        rasterize_fn=None,
        exposure_module=None,
        image_ids: Optional[Tensor] = None,
        pixels: Optional[Tensor] = None,
        depths: Optional[Tensor] = None,
        depths_gt: Optional[Tensor] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scene_scale: float = 1.0,
        bil_grids=None,
        step: int = 0,
        sh_degree_to_use: int = 3,
        target_exposure: Optional[float] = None,
        world_size: int = 1,
    ) -> Tuple[Tensor, Dict[str, Optional[Tensor]]]:
        """Compute total loss combining all components.
        
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual loss components
        """
        loss_dict = {
            "loss_nll": None,
            "l1loss": None,
            "ssimloss": None,
            "loss_ratio": None,
            "loss_sh": None,
            "loss_albedo_reg": None,
            "loss_illum_sparse": None,
            "loss_exp_diversity": None,
            "depthloss": None,
            "tvloss": None,
        }
        
        # Main reconstruction loss
        use_log_space_loss = getattr(self.cfg, 'use_log_space_loss', False)
        log_space_eps = getattr(self.cfg, 'log_space_eps', 1e-3)
        ssim_lambda = getattr(self.cfg, 'ssim_lambda', 0.2)
        use_nll_loss = getattr(self.cfg, 'use_nll_loss', False)
        
        if use_log_space_loss:
            loss = F.l1_loss(
                torch.log(colors + log_space_eps),
                torch.log(pixels_target + log_space_eps)
            )
            if ssim_lambda > 0:
                loss_dict["ssimloss"] = self.compute_ssim_loss(colors, pixels_target)
                loss = loss * (1.0 - ssim_lambda) + loss_dict["ssimloss"] * ssim_lambda
        elif use_nll_loss:
            loss_dict["loss_nll"] = self.compute_nll_loss(colors, pixels_target)
            loss = loss_dict["loss_nll"]
            if ssim_lambda > 0:
                loss_dict["ssimloss"] = self.compute_ssim_loss(colors, pixels_target)
                loss = loss_dict["loss_nll"] * (1.0 - ssim_lambda) + loss_dict["ssimloss"] * ssim_lambda
        else:
            loss_dict["l1loss"] = self.compute_l1_loss(colors, pixels_target)
            loss_dict["ssimloss"] = self.compute_ssim_loss(colors, pixels_target)
            loss = loss_dict["l1loss"] * (1.0 - ssim_lambda) + loss_dict["ssimloss"] * ssim_lambda
        
        # Ratio loss (HDR consistency)
        ratio_lambda = getattr(self.cfg, 'ratio_lambda', 0.1)
        if rasterize_fn is not None and trainset is not None:
            loss_dict["loss_ratio"] = self.compute_ratio_loss(
                splats, trainset, rasterize_fn, self.device, step, sh_degree_to_use, target_exposure
            )
            if loss_dict["loss_ratio"] is not None:
                loss += ratio_lambda * loss_dict["loss_ratio"]
        
        # SH regularization
        sh_reg_lambda = getattr(self.cfg, 'sh_reg_lambda', 0.01)
        loss_dict["loss_sh"] = self.compute_sh_regularization(splats)
        if loss_dict["loss_sh"] is not None:
            loss += sh_reg_lambda * loss_dict["loss_sh"]
        
        # Intrinsic decomposition regularizations
        loss_dict["loss_albedo_reg"] = self.compute_albedo_regularization(splats)
        if loss_dict["loss_albedo_reg"] is not None:
            loss += loss_dict["loss_albedo_reg"]
        
        loss_dict["loss_illum_sparse"] = self.compute_illumination_sparsity(splats)
        if loss_dict["loss_illum_sparse"] is not None:
            loss += loss_dict["loss_illum_sparse"]
        
        # Exposure diversity
        if exposure_module is not None and image_ids is not None and pixels is not None:
            loss_dict["loss_exp_diversity"] = self.compute_exposure_diversity(
                exposure_module, image_ids, pixels, world_size
            )
            if loss_dict["loss_exp_diversity"] is not None:
                loss += loss_dict["loss_exp_diversity"]
        
        # Depth loss
        depth_loss = getattr(self.cfg, 'depth_loss', False)
        depth_lambda = getattr(self.cfg, 'depth_lambda', 1e-2)
        if depth_loss and depths is not None and depths_gt is not None and width is not None and height is not None:
            # depths_gt should be a tuple of (points, depths) where points is [B, M, 2] and depths is [B, M]
            if isinstance(depths_gt, tuple):
                points_gt, depths_gt_values = depths_gt
            else:
                # Fallback: assume depths_gt is just depths values
                points_gt = None
                depths_gt_values = depths_gt
            
            if points_gt is not None:
                # Normalize points to [-1, 1]
                points = torch.stack(
                    [
                        points_gt[:, :, 0] / (width - 1) * 2 - 1,
                        points_gt[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths_sampled = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths_sampled = depths_sampled.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths_sampled > 0.0, 1.0 / depths_sampled, torch.zeros_like(depths_sampled))
                disp_gt = 1.0 / depths_gt_values  # [1, M]
                loss_dict["depthloss"] = F.l1_loss(disp, disp_gt) * scene_scale
                loss += loss_dict["depthloss"] * depth_lambda
        
        # Total variation loss (bilateral grid)
        use_bilateral_grid = getattr(self.cfg, 'use_bilateral_grid', False)
        if use_bilateral_grid and bil_grids is not None:
            from lib_bilagrid import total_variation_loss
            loss_dict["tvloss"] = 10 * total_variation_loss(bil_grids.grids)
            loss += loss_dict["tvloss"]
        
        # Regularizations
        opacity_reg = getattr(self.cfg, 'opacity_reg', 0.0)
        scale_reg = getattr(self.cfg, 'scale_reg', 0.0)
        if opacity_reg > 0.0:
            loss += opacity_reg * torch.sigmoid(splats["opacities"]).mean()
        if scale_reg > 0.0:
            loss += scale_reg * torch.exp(splats["scales"]).mean()
        
        return loss, loss_dict

