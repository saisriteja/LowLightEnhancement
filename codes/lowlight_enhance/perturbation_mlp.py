"""
Perturbation MLP for Multi-Exposure Gaussian Splatting.

This module implements MLP_Δ that outputs exposure-dependent perturbations
for Gaussian parameters (colors, opacities, scales).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerturbationMLP(nn.Module):
    """
    Small MLP that outputs perturbations for Gaussian parameters based on exposure.
    
    Input: [position (3D), view_dir (2D spherical), exposure_normalized (1D)] = 6D
    Output: [Δc (3), Δα (1), Δσ (1)] = 5D perturbations
    
    Args:
        hidden_dim: Hidden dimension for MLP layers (default: 32)
        max_delta_c: Maximum color perturbation (default: 0.5)
        max_delta_alpha: Maximum opacity perturbation (default: 0.25)
        max_delta_sigma: Maximum scale perturbation (default: 0.1)
    """
    
    def __init__(
        self,
        hidden_dim: int = 32,
        max_delta_c: float = 0.1,
        max_delta_alpha: float = 0.05,
        max_delta_sigma: float = 0.05,
    ):
        super().__init__()
        
        self.max_delta_c = max_delta_c
        self.max_delta_alpha = max_delta_alpha
        self.max_delta_sigma = max_delta_sigma
        
        # Input: [position (3), view_dir (2), exposure (1)] = 6D
        self.fc1 = nn.Linear(6, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 7)  # Output 7 values (3 for c, 1 for alpha, 1 for sigma, 2 reserved)
        
        # CRITICAL: Initialize final layer to output zeros (no perturbation at start)
        # This ensures stable training - MLP starts with identity (no change to Gaussians)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
        
    def forward(
        self,
        position: torch.Tensor,
        view_dir: torch.Tensor,
        exposure_normalized: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            position: [N_G, 3] or [1, 3] - Gaussian positions
            view_dir: [N_G, 2] or [1, 2] - View directions in spherical coords (theta, phi)
            exposure_normalized: [N_G, 1] or [1, 1] - Normalized exposure in [-1, 1]
        
        Returns:
            perturbations: [N_G, 5] = [Δc_R, Δc_G, Δc_B, Δα, Δσ]
        """
        # Concatenate inputs
        x = torch.cat([position, view_dir, exposure_normalized], dim=-1)  # [N, 6]
        
        # Forward through MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc_out(x)  # [N, 7]
        
        # Bound outputs
        delta_c = torch.tanh(out[..., :3]) * self.max_delta_c  # [N, 3]
        delta_alpha = torch.tanh(out[..., 3:4]) * self.max_delta_alpha  # [N, 1]
        delta_sigma = torch.tanh(out[..., 4:5]) * self.max_delta_sigma  # [N, 1]
        
        return torch.cat([delta_c, delta_alpha, delta_sigma], dim=-1)  # [N, 5]


class IlluminationMLP(nn.Module):
    """
    MLP that outputs illumination multiplier L(e) as a function of exposure.
    
    Input: exposure_normalized (1D) 
    Output: L(e) ∈ ℝ³ (RGB illumination multiplier)
    
    Property: L(exposure=0) = [1, 1, 1] to fix scale ambiguity
    Smooth function: d²(log L)/de² ≈ 0
    
    Args:
        hidden_dim: Hidden dimension for MLP layers (default: 32)
    """
    
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        
        # Input: exposure_normalized (1D)
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 3)  # Output RGB illumination
        
        # Initialize to output [1, 1, 1] at exposure=0
        # This fixes the scale ambiguity: L(0) = [1, 1, 1]
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
        
    def forward(self, exposure_normalized: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            exposure_normalized: [N_G, 1] or [1, 1] - Normalized exposure in [-1, 1]
        
        Returns:
            illumination: [N_G, 3] or [1, 3] - RGB illumination multiplier L(e)
        """
        # Forward through MLP
        x = F.relu(self.fc1(exposure_normalized))
        x = F.relu(self.fc2(x))
        out = self.fc_out(x)  # [N, 3]
        
        # Ensure positive output: use sigmoid-based activation
        # L(e) = 1 + tanh(out) * scale_factor
        # Constrain to [0.5, 1.5] range as per update.md
        # L(0) = 1.0 (hard constraint)
        illumination = 1.0 + torch.tanh(out) * 0.5  # L ∈ [0.5, 1.5], L(0) = 1
        
        return illumination


class VisibilityMLP(nn.Module):
    """
    MLP that outputs visibility multiplier V(view) as a function of view direction.
    
    Input: view_dir (2D spherical coordinates)
    Output: V(view) ∈ ℝ³ (RGB visibility multiplier)
    
    Property: Smooth function of view direction
    Initialize to output [1, 1, 1] (no occlusion initially)
    
    Args:
        hidden_dim: Hidden dimension for MLP layers (default: 32)
    """
    
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        
        # Input: view_dir (2D spherical coordinates)
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 3)  # Output RGB visibility
        
        # Initialize to output [1, 1, 1] (no occlusion initially)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
        
    def forward(self, view_dir: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            view_dir: [N_G, 2] or [1, 2] - View directions in spherical coords (theta, phi)
        
        Returns:
            visibility: [N_G, 3] or [1, 3] - RGB visibility multiplier V(view)
        """
        # Forward through MLP
        x = F.relu(self.fc1(view_dir))
        x = F.relu(self.fc2(x))
        out = self.fc_out(x)  # [N, 3]
        
        # Ensure positive output: use sigmoid-based activation
        # V(view) = 1 + tanh(out) ensures V ∈ [0, 2] and V(0) ≈ 1
        visibility = 1.0 + torch.tanh(out) * 0.5  # V ∈ [0.5, 1.5], V(0) ≈ 1
        
        return visibility

