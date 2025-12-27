"""
HDR-aware rasterization module for Gaussian Splatting.
"""

from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import assert_never

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy


def rasterize_splats(
    splats: torch.nn.ParameterDict,
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    cfg,
    app_module=None,
    exposure_module=None,
    world_size: int = 1,
    masks: Optional[Tensor] = None,
    rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
    camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
    apply_exposure: bool = True,
    **kwargs,
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize splats with optional exposure application and HDR support.
    
    Args:
        splats: ParameterDict containing Gaussian splat parameters
        camtoworlds: Camera-to-world matrices [B, 4, 4]
        Ks: Camera intrinsics [B, 3, 3]
        width: Image width
        height: Image height
        cfg: Configuration object
        app_module: Appearance optimization module (optional)
        exposure_module: Exposure optimization module (optional)
        world_size: Number of distributed processes
        masks: Optional mask tensor [B, H, W]
        rasterize_mode: Rasterization mode
        camera_model: Camera model type
        apply_exposure: If True, multiply rendered radiance by exposure.
                       If False, return raw HDR radiance.
        **kwargs: Additional arguments passed to rasterization
        
    Returns:
        Tuple of (render_colors, render_alphas, info)
    """
    means = splats["means"]  # [N, 3]
    quats = splats["quats"]  # [N, 4]
    scales = torch.exp(splats["scales"])  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"])  # [N,]

    image_ids = kwargs.pop("image_ids", None)
    device = means.device
    
    # Get config attributes with defaults
    app_opt = getattr(cfg, 'app_opt', False)
    sh_degree = getattr(cfg, 'sh_degree', 3)
    enable_intrinsic_decomp = getattr(cfg, 'enable_intrinsic_decomp', False)
    enable_exposure_opt = getattr(cfg, 'enable_exposure_opt', False)
    antialiased = getattr(cfg, 'antialiased', False)
    camera_model_val = getattr(cfg, 'camera_model', 'pinhole')
    packed = getattr(cfg, 'packed', False)
    sparse_grad = getattr(cfg, 'sparse_grad', False)
    with_ut = getattr(cfg, 'with_ut', False)
    with_eval3d = getattr(cfg, 'with_eval3d', False)
    
    if app_opt and app_module is not None:
        colors = app_module(
            features=splats["features"],
            embed_ids=image_ids,
            dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
            sh_degree=kwargs.pop("sh_degree", sh_degree),
        )
        colors = colors + splats["colors"]
        colors = torch.sigmoid(colors)
    elif enable_intrinsic_decomp:
        # Intrinsic decomposition: colors = albedo + illumination * exposure
        # Get exposure value (default to 1.0 if not provided)
        exposure_value = kwargs.pop("exposure_value", None)
        if exposure_value is None:
            if apply_exposure and image_ids is not None and enable_exposure_opt and exposure_module is not None:
                if world_size > 1:
                    exposure_value = exposure_module.module(image_ids)  # [B,]
                else:
                    exposure_value = exposure_module(image_ids)  # [B,]
            else:
                # Default to 1.0 if no exposure module or not applying exposure
                exposure_value = torch.ones(camtoworlds.shape[0], device=device)
        
        # Combine albedo and illumination
        albedo = torch.cat([splats["albedo_sh0"], splats["albedo_shN"]], 1)  # [N, K, 3]
        illumination = torch.cat([splats["illum_sh0"], splats["illum_shN"]], 1)  # [N, K, 3]
        
        # Final color = albedo + illumination * exposure
        # For per-image exposure: Since colors are per-Gaussian but exposure is per-image,
        # we need to handle this. For now, use mean exposure across batch (typical batch_size=1)
        if exposure_value.numel() > 0:
            exposure_scalar = exposure_value.mean().item()  # Use mean for simplicity
        else:
            exposure_scalar = 1.0
        
        # Combine: colors = albedo + illumination * exposure
        colors = albedo + illumination * exposure_scalar
    else:
        colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]

    if rasterize_mode is None:
        rasterize_mode = "antialiased" if antialiased else "classic"
    if camera_model is None:
        camera_model = camera_model_val
    
    # Get strategy absgrad with fallback
    absgrad = False
    if hasattr(cfg, 'strategy') and cfg.strategy is not None:
        if isinstance(cfg.strategy, DefaultStrategy):
            absgrad = getattr(cfg.strategy, 'absgrad', False)
    
    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
        Ks=Ks,  # [C, 3, 3]
        width=width,
        height=height,
        packed=packed,
        absgrad=absgrad,
        sparse_grad=sparse_grad,
        rasterize_mode=rasterize_mode,
        distributed=world_size > 1,
        camera_model=camera_model_val,
        with_ut=with_ut,
        with_eval3d=with_eval3d,
        **kwargs,
    )
    
    # Apply exposure if enabled and requested
    # Note: For intrinsic decomposition, exposure is already applied in color computation
    if (apply_exposure and enable_exposure_opt and image_ids is not None 
        and not enable_intrinsic_decomp and exposure_module is not None):
        if world_size > 1:
            exposures = exposure_module.module(image_ids)  # [B,]
        else:
            exposures = exposure_module(image_ids)  # [B,]
        render_colors = render_colors * exposures.view(-1, 1, 1, 1)
    
    if masks is not None:
        render_colors[~masks] = 0
    return render_colors, render_alphas, info

