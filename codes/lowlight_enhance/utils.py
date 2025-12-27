"""
Utility functions for HDR Gaussian Splatting training.
"""

import os
import random
from typing import List, Tuple

import imageio.v2 as imageio_v2
import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import tqdm


def srgb_to_linear(img: Tensor) -> Tensor:
    """Convert sRGB images to linear space.
    
    Args:
        img: Image tensor in [0, 1] sRGB space
        
    Returns:
        Image tensor in linear space
    """
    return torch.where(
        img <= 0.04045,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4
    )


def aces_tonemap(x: Tensor) -> Tensor:
    """ACES tone mapping for HDR to LDR conversion.
    
    Args:
        x: HDR radiance values
        
    Returns:
        Tone-mapped values in [0, 1]
    """
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def get_curriculum_exposure(step: int, max_steps: int, max_exposure: float = 10.0) -> float:
    """Get curriculum exposure based on training progress.
    
    Phase 1 (0-30%): Reconstruct dark images (exposure ~1.0)
    Phase 2 (30-70%): Gradually increase (exposure 1.0 → 5.0)
    Phase 3 (70-100%): Target bright images (exposure 5.0 → max_exposure)
    
    Args:
        step: Current training step
        max_steps: Total number of training steps
        max_exposure: Maximum exposure value (default: 10.0)
        
    Returns:
        Target exposure value for current step
    """
    progress = step / max_steps if max_steps > 0 else 0.0
    
    if progress < 0.3:
        # Phase 1: Reconstruct dark (exposure = 1.0)
        return 1.0
    elif progress < 0.7:
        # Phase 2: Gradual increase from 1.0 to 5.0
        phase_progress = (progress - 0.3) / 0.4  # 0.0 to 1.0 within phase 2
        return 1.0 + phase_progress * 4.0  # 1.0 → 5.0
    else:
        # Phase 3: Target bright from 5.0 to max_exposure
        phase_progress = (progress - 0.7) / 0.3  # 0.0 to 1.0 within phase 3
        return 5.0 + phase_progress * (max_exposure - 5.0)  # 5.0 → max_exposure


def generate_synthetic_exposures(
    dark_image: Tensor, 
    num_exposures: int = 5,
    min_exposure: float = 0.5,
    max_exposure: float = 5.0
) -> Tuple[List[Tensor], List[float]]:
    """Generate synthetic multi-exposure training data from single dark image.
    
    Creates training triplets: (dark, mid, bright) from single dark image.
    Key insight: Even if GT is dark, we can create synthetic bright versions
    and enforce consistency in learned radiance space.
    
    Args:
        dark_image: Dark input image tensor [H, W, 3] or [1, H, W, 3]
        num_exposures: Number of synthetic exposures to generate
        min_exposure: Minimum exposure factor
        max_exposure: Maximum exposure factor
        
    Returns:
        Tuple of (synthetic_images, exposures) where:
        - synthetic_images: List of synthetic image tensors
        - exposures: List of corresponding exposure values
    """
    # Ensure image is 4D [1, H, W, 3]
    if dark_image.dim() == 3:
        dark_image = dark_image.unsqueeze(0)
    
    exposures = torch.linspace(min_exposure, max_exposure, num_exposures).tolist()
    synthetic_images = []
    
    for exp in exposures:
        # Simple: Just brighten the image
        synthetic = dark_image * exp
        
        # Better: Add noise model (bright images have less relative noise)
        noise_level = 0.01 / exp  # Less noise in bright images
        noise = torch.randn_like(synthetic) * noise_level
        synthetic = synthetic + noise
        
        # Clamp to valid range
        synthetic = torch.clamp(synthetic, 0.0, 1.0)
        synthetic_images.append(synthetic)
    
    return synthetic_images, exposures


def knn(x: Tensor, K: int = 4) -> Tensor:
    """K-nearest neighbors distance computation."""
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    """Convert RGB to spherical harmonics coefficient 0."""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm.tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio_v2.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio_v2.imwrite(resized_path, resized_image)
    return resized_dir

