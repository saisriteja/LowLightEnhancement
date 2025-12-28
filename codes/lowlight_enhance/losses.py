"""
Loss functions for Multi-Exposure Gaussian Splatting.

Implements exposure consistency, smoothness, and regularization losses.
"""

import torch
import torch.nn.functional as F


def compute_exposure_consistency_loss(
    gaussians: torch.nn.ParameterDict,
    perturbation_mlp: torch.nn.Module,
    exposure_pairs: list,
    lambda_weight: float = 0.5,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Enforce that geometry is exposure-invariant.
    
    Computes L1 distance between normalized perturbed Gaussians at different exposures.
    
    Args:
        gaussians: ParameterDict containing Gaussian parameters (means, opacities, etc.)
        perturbation_mlp: PerturbationMLP network
        exposure_pairs: List of (e1, e2) tuples to compare
        lambda_weight: Loss weight
        epsilon: Small epsilon for numerical stability
    
    Returns:
        Scalar loss tensor
    """
    loss = 0.0
    means = gaussians["means"]  # [N_G, 3]
    
    # Compute view directions (dummy - we'll use zero view dir for consistency)
    # In practice, view direction shouldn't affect consistency, so we use zeros
    N_G = means.shape[0]
    device = means.device
    dummy_view_dir = torch.zeros((N_G, 2), device=device)
    
    for e1, e2 in exposure_pairs:
        # Normalize exposures
        e1_norm = torch.clamp(torch.tensor(e1 / 2.0, device=device), -1, 1)
        e2_norm = torch.clamp(torch.tensor(e2 / 2.0, device=device), -1, 1)
        
        # Expand to match number of Gaussians
        e1_norm_expanded = e1_norm.unsqueeze(0).repeat(N_G, 1)
        e2_norm_expanded = e2_norm.unsqueeze(0).repeat(N_G, 1)
        
        # Get perturbations at two exposures
        pert1 = perturbation_mlp(means, dummy_view_dir, e1_norm_expanded)  # [N_G, 5]
        pert2 = perturbation_mlp(means, dummy_view_dir, e2_norm_expanded)  # [N_G, 5]
        
        # Normalize perturbations (L2 normalize)
        pert1_norm = pert1 / (torch.norm(pert1, dim=-1, keepdim=True) + epsilon)
        pert2_norm = pert2 / (torch.norm(pert2, dim=-1, keepdim=True) + epsilon)
        
        # L1 distance between normalized perturbations
        diff = torch.abs(pert1_norm - pert2_norm)
        loss += torch.mean(diff)
    
    return lambda_weight * loss / len(exposure_pairs) if len(exposure_pairs) > 0 else torch.tensor(0.0, device=means.device)


def compute_smoothness_loss(
    perturbation_mlp: torch.nn.Module,
    positions: torch.Tensor,
    view_dirs: torch.Tensor,
    exposures: list,
    delta_e: float = 0.2,
    lambda_weight: float = 0.1,
) -> torch.Tensor:
    """
    Regularize that exposure transitions are smooth.
    
    Uses finite differences to compute gradients w.r.t. exposure.
    
    Args:
        perturbation_mlp: PerturbationMLP network
        positions: [N_G, 3] - Gaussian positions
        view_dirs: [N_G, 2] - View directions
        exposures: List of exposure values to sample
        delta_e: Step size for finite difference (in EV)
        lambda_weight: Loss weight
    
    Returns:
        Scalar loss tensor
    """
    loss = 0.0
    device = positions.device
    
    for e_center in exposures:
        e_plus = (e_center + delta_e) / 2.0  # Normalize
        e_minus = (e_center - delta_e) / 2.0  # Normalize
        
        # Expand to match number of Gaussians
        N_G = positions.shape[0]
        e_plus_expanded = torch.ones((N_G, 1), device=device) * e_plus
        e_minus_expanded = torch.ones((N_G, 1), device=device) * e_minus
        
        # Get perturbations at e+δe and e-δe
        delta_plus = perturbation_mlp(positions, view_dirs, e_plus_expanded)  # [N_G, 5]
        delta_minus = perturbation_mlp(positions, view_dirs, e_minus_expanded)  # [N_G, 5]
        
        # Gradient w.r.t. exposure (finite difference)
        grad_e = (delta_plus - delta_minus) / (2 * delta_e)  # [N_G, 5]
        
        # Smooth (low gradient) - penalize squared gradient
        loss += torch.mean(grad_e ** 2)
    
    return lambda_weight * loss / len(exposures) if len(exposures) > 0 else torch.tensor(0.0, device=device)


def compute_regularization_loss(
    perturbation_mlp: torch.nn.Module,
    positions: torch.Tensor,
    view_dirs: torch.Tensor,
    exposure_normalized: torch.Tensor,
    lambda_weight: float = 1e-4,
) -> torch.Tensor:
    """
    Regularize perturbation magnitudes (L2 penalty).
    
    Args:
        perturbation_mlp: PerturbationMLP network
        positions: [N_G, 3] - Gaussian positions
        view_dirs: [N_G, 2] - View directions
        exposure_normalized: [N_G, 1] - Normalized exposure
        lambda_weight: Loss weight
    
    Returns:
        Scalar loss tensor
    """
    perturbations = perturbation_mlp(positions, view_dirs, exposure_normalized)  # [N_G, 5]
    
    # L2 norm of perturbations
    reg = torch.mean(torch.norm(perturbations, p=2, dim=-1))
    
    return lambda_weight * reg

