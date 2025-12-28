"""
Utility functions for Multi-Exposure Gaussian Splatting.
"""

import math
import random
from typing import List

import numpy as np
import torch


def compute_view_directions(
    positions: torch.Tensor,
    camtoworlds: torch.Tensor,
) -> torch.Tensor:
    """
    Convert 3D positions to spherical coordinates (theta, phi) relative to camera.
    
    Args:
        positions: [N_G, 3] - Gaussian positions in world space
        camtoworlds: [B, 4, 4] or [1, 4, 4] - Camera-to-world transformation matrices
    
    Returns:
        view_dirs: [N_G, 2] - Spherical coordinates (theta, phi) for each Gaussian
    """
    # Get camera position (translation component)
    if camtoworlds.dim() == 3:
        camera_pos = camtoworlds[0, :3, 3]  # [3]
    else:
        camera_pos = camtoworlds[:3, 3]  # [3]
    
    # Compute direction from camera to Gaussians
    directions = positions - camera_pos.unsqueeze(0)  # [N_G, 3]
    
    # Normalize
    norms = torch.norm(directions, dim=-1, keepdim=True) + 1e-8
    directions = directions / norms
    
    # Convert to spherical coordinates
    # theta: azimuthal angle (0 to 2π)
    # phi: polar angle (0 to π)
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    
    theta = torch.atan2(y, x)  # [-π, π]
    phi = torch.acos(torch.clamp(z, -1, 1))  # [0, π]
    
    # Normalize theta to [0, 1] and phi to [0, 1] for better MLP input
    theta_norm = (theta + math.pi) / (2 * math.pi)  # [0, 1]
    phi_norm = phi / math.pi  # [0, 1]
    
    return torch.stack([theta_norm, phi_norm], dim=-1)  # [N_G, 2]


def normalize_exposure(exposure_ev: float) -> float:
    """
    Convert exposure value (EV) to normalized range [-1, 1].
    
    Args:
        exposure_ev: Exposure value in EV
    
    Returns:
        Normalized exposure in [-1, 1]
    """
    return max(-1.0, min(1.0, exposure_ev / 2.0))


def denormalize_exposure(exposure_norm: float) -> float:
    """
    Convert normalized exposure back to EV.
    
    Args:
        exposure_norm: Normalized exposure in [-1, 1]
    
    Returns:
        Exposure value in EV
    """
    return exposure_norm * 2.0


def get_curriculum_exposure(
    step: int,
    max_steps: int,
    phases: List[int],
    ranges: List[float],
) -> float:
    """
    Get exposure value based on curriculum learning schedule.
    
    Args:
        step: Current training step
        max_steps: Maximum training steps
        phases: List of phase boundaries (e.g., [300, 600, 1000])
        ranges: List of exposure ranges per phase (e.g., [1.0, 2.0, 3.0])
    
    Returns:
        Exposure value in EV (sampled uniformly from current phase range)
    """
    # Determine current phase
    phase_idx = 0
    for i, phase_boundary in enumerate(phases):
        if step < phase_boundary:
            phase_idx = i
            break
    else:
        phase_idx = len(ranges) - 1
    
    # Get exposure range for current phase
    exposure_range = ranges[phase_idx]
    
    # Sample uniformly from [-range, +range]
    exposure_ev = random.uniform(-exposure_range, exposure_range)
    
    return exposure_ev

