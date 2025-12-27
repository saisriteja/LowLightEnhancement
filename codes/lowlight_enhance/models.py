"""
Model classes for HDR Gaussian Splatting training.
"""

import torch
from torch import Tensor


class ExposureOptModule(torch.nn.Module):
    """Per-image exposure parameter optimization module."""
    
    def __init__(self, n_images: int):
        super().__init__()
        # Initialize exposures to 1.0 (neutral exposure)
        self.exposures = torch.nn.Parameter(torch.ones(n_images))
    
    def forward(self, image_ids: Tensor) -> Tensor:
        """Get exposure values for given image IDs.
        
        Args:
            image_ids: Tensor of image indices [B,]
            
        Returns:
            Exposure values [B,]
        """
        return self.exposures[image_ids]

