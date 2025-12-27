"""
Splat creation and optimization setup.
"""

import importlib.util
import math
import os
from typing import Dict, Optional, Tuple

import torch

from dataset import Parser
from gsplat.optimizers import SelectiveAdam

# Import from local utils.py (not gsplat examples utils)
_local_utils_file = os.path.join(os.path.dirname(__file__), "utils.py")
if os.path.exists(_local_utils_file):
    spec = importlib.util.spec_from_file_location("hdr_utils", _local_utils_file)
    hdr_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hdr_utils)
    knn = hdr_utils.knn
    rgb_to_sh = hdr_utils.rgb_to_sh
else:
    raise ImportError(f"Local utils.py not found at {_local_utils_file}")


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    enable_intrinsic_decomp: bool = False,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """Create Gaussian splats and their optimizers.
    
    Args:
        parser: Parser object with scene data
        init_type: Initialization type ("sfm" or "random")
        init_num_pts: Initial number of points (for random init)
        init_extent: Initial extent (for random init)
        init_opacity: Initial opacity value
        init_scale: Initial scale factor
        means_lr: Learning rate for positions
        scales_lr: Learning rate for scales
        opacities_lr: Learning rate for opacities
        quats_lr: Learning rate for quaternions
        sh0_lr: Learning rate for SH band 0
        shN_lr: Learning rate for higher-order SH
        scene_scale: Scene scale factor
        sh_degree: Spherical harmonics degree
        sparse_grad: Use sparse gradients
        visible_adam: Use visible Adam optimizer
        batch_size: Batch size
        feature_dim: Feature dimension (for appearance optimization)
        device: Device to use
        world_rank: World rank for distributed training
        world_size: World size for distributed training
        enable_intrinsic_decomp: Enable intrinsic decomposition
        
    Returns:
        Tuple of (splats ParameterDict, optimizers dict)
    """
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        if enable_intrinsic_decomp:
            # Intrinsic decomposition: Split into albedo (invariant) + illumination (variant)
            # Albedo: Exposure-invariant reflectance (what color is the surface?)
            albedo_sh0_raw = rgb_to_sh(rgbs)  # Returns [N, 3]
            albedo_sh0 = albedo_sh0_raw.unsqueeze(1)  # Reshape to [N, 1, 3]
            # Higher-order SH: (sh_degree + 1)^2 - 1 coefficients (excluding sh0)
            num_sh_coeffs = (sh_degree + 1) ** 2 - 1
            albedo_shN = torch.zeros((N, num_sh_coeffs, 3))  # Higher-order SH for albedo
            
            # Illumination: Exposure-dependent lighting (how much light hits it?)
            illum_sh0 = torch.zeros((N, 1, 3))  # Initialize to small values
            illum_shN = torch.zeros((N, num_sh_coeffs, 3))  # Higher-order SH for illumination
            
            params.append(("albedo_sh0", torch.nn.Parameter(albedo_sh0), sh0_lr))
            params.append(("albedo_shN", torch.nn.Parameter(albedo_shN), shN_lr))
            params.append(("illum_sh0", torch.nn.Parameter(illum_sh0), sh0_lr))
            params.append(("illum_shN", torch.nn.Parameter(illum_shN), shN_lr))
        else:
            # Standard: color is SH coefficients.
            colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
            colors[:, 0, :] = rgb_to_sh(rgbs)
            params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
            params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )
        for name, _, lr in params
    }
    return splats, optimizers

