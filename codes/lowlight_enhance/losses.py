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


def compute_reflectance_consistency_loss(
    illumination_mlp: torch.nn.Module,
    visibility_mlp: torch.nn.Module,
    observations: list,
    lambda_weight: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Enforce that reflectance R is consistent across multiple observations.
    
    For each Gaussian, compute R̂_k = I_k / (L(e_k) × V(C_k)) from multiple observations.
    Loss: Σ_{i,j} ||R̂_i - R̂_j||² where i,j are different views/exposures.
    
    Args:
        illumination_mlp: IlluminationMLP network
        visibility_mlp: VisibilityMLP network
        observations: List of dicts with keys: 'exposure_ev', 'view_dir', 'intensity'
            Each dict represents one observation: {'exposure_ev': float, 'view_dir': [N_G, 2], 'intensity': [N_G, 3]}
        lambda_weight: Loss weight
        epsilon: Small epsilon for numerical stability
    
    Returns:
        Scalar loss tensor
    """
    if len(observations) < 2:
        device = next(illumination_mlp.parameters()).device
        return torch.tensor(0.0, device=device)
    
    device = next(illumination_mlp.parameters()).device
    N_G = observations[0]['intensity'].shape[0]
    
    # Compute R̂_k for each observation
    R_hats = []
    for obs in observations:
        exposure_ev = obs['exposure_ev']
        view_dir = obs['view_dir']  # [N_G, 2]
        intensity = obs['intensity']  # [N_G, 3] - observed intensity I_k
        
        # Normalize exposure
        exposure_norm = torch.clamp(torch.tensor(exposure_ev / 2.0, device=device), -1, 1)
        exposure_norm_expanded = exposure_norm.unsqueeze(0).repeat(N_G, 1)  # [N_G, 1]
        
        # Get L(e_k) and V(C_k)
        L_e = illumination_mlp(exposure_norm_expanded)  # [N_G, 3]
        V_view = visibility_mlp(view_dir)  # [N_G, 3]
        
        # Compute R̂_k = I_k / (L(e_k) × V(C_k))
        denominator = L_e * V_view + epsilon  # [N_G, 3]
        R_hat = intensity / denominator  # [N_G, 3]
        R_hats.append(R_hat)
    
    # Compute pairwise differences
    loss = 0.0
    num_pairs = 0
    for i in range(len(R_hats)):
        for j in range(i + 1, len(R_hats)):
            diff = R_hats[i] - R_hats[j]  # [N_G, 3]
            loss += torch.mean(diff ** 2)
            num_pairs += 1
    
    return lambda_weight * loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=device)


def compute_illumination_smoothness_loss(
    illumination_mlp: torch.nn.Module,
    exposures: list,
    delta_e: float = 0.1,
    lambda_weight: float = 0.1,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Regularize that illumination L(e) is smooth in exposure space.
    
    Computes second derivative: d²(log L)/de² using finite differences.
    Loss: Σ_e ||d²(log L)/de²||²
    
    Args:
        illumination_mlp: IlluminationMLP network
        exposures: List of exposure values (in EV) to sample
        delta_e: Step size for finite difference (in EV)
        lambda_weight: Loss weight
        epsilon: Small epsilon for numerical stability
    
    Returns:
        Scalar loss tensor
    """
    if len(exposures) == 0:
        return torch.tensor(0.0)
    
    device = next(illumination_mlp.parameters()).device
    loss = 0.0
    
    for e_center in exposures:
        # Sample at e_center + delta_e, e_center, e_center - delta_e
        e_plus = e_center + delta_e
        e_minus = e_center - delta_e
        
        # Normalize exposures
        e_center_norm = torch.clamp(torch.tensor(e_center / 2.0, device=device), -1, 1)
        e_plus_norm = torch.clamp(torch.tensor(e_plus / 2.0, device=device), -1, 1)
        e_minus_norm = torch.clamp(torch.tensor(e_minus / 2.0, device=device), -1, 1)
        
        # Expand to batch (we'll use single sample for smoothness)
        e_center_expanded = e_center_norm.unsqueeze(0).unsqueeze(0)  # [1, 1]
        e_plus_expanded = e_plus_norm.unsqueeze(0).unsqueeze(0)  # [1, 1]
        e_minus_expanded = e_minus_norm.unsqueeze(0).unsqueeze(0)  # [1, 1]
        
        # Get L(e) at three points
        L_center = illumination_mlp(e_center_expanded)  # [1, 3]
        L_plus = illumination_mlp(e_plus_expanded)  # [1, 3]
        L_minus = illumination_mlp(e_minus_expanded)  # [1, 3]
        
        # Compute log L
        log_L_center = torch.log(L_center + epsilon)
        log_L_plus = torch.log(L_plus + epsilon)
        log_L_minus = torch.log(L_minus + epsilon)
        
        # Second derivative: d²(log L)/de² ≈ (log_L_plus - 2*log_L_center + log_L_minus) / delta_e²
        second_deriv = (log_L_plus - 2 * log_L_center + log_L_minus) / (delta_e ** 2)  # [1, 3]
        
        # Penalize squared second derivative
        loss += torch.mean(second_deriv ** 2)
    
    return lambda_weight * loss / len(exposures) if len(exposures) > 0 else torch.tensor(0.0, device=device)


def compute_visibility_smoothness_loss(
    visibility_mlp: torch.nn.Module,
    view_pairs: list,
    lambda_weight: float = 0.1,
) -> torch.Tensor:
    """
    Regularize that visibility V(view) changes smoothly across nearby views.
    
    For nearby camera views, compute: ||V(view_a) - V(view_b)||²
    Weight by distance: exp(-distance) × ||V(A) - V(B)||²
    
    Args:
        visibility_mlp: VisibilityMLP network
        view_pairs: List of dicts with keys: 'view_dir_a', 'view_dir_b', 'distance'
            Each dict: {'view_dir_a': [N_G, 2], 'view_dir_b': [N_G, 2], 'distance': float}
        lambda_weight: Loss weight
    
    Returns:
        Scalar loss tensor
    """
    if len(view_pairs) == 0:
        return torch.tensor(0.0)
    
    device = next(visibility_mlp.parameters()).device
    loss = 0.0
    
    for pair in view_pairs:
        view_dir_a = pair['view_dir_a']  # [N_G, 2]
        view_dir_b = pair['view_dir_b']  # [N_G, 2]
        distance = pair['distance']  # scalar
        
        # Get V(view_a) and V(view_b)
        V_a = visibility_mlp(view_dir_a)  # [N_G, 3]
        V_b = visibility_mlp(view_dir_b)  # [N_G, 3]
        
        # Compute difference
        diff = V_a - V_b  # [N_G, 3]
        
        # Weight by distance: exp(-distance) encourages smoothness for nearby views
        weight = torch.exp(-torch.tensor(distance, device=device))
        loss += weight * torch.mean(diff ** 2)
    
    return lambda_weight * loss / len(view_pairs) if len(view_pairs) > 0 else torch.tensor(0.0, device=device)

