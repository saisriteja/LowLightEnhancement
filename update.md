# Multi-Exposure Gaussian Splatting: Complete Implementation Guide

## Part 1: Core Mathematical Definitions

### Exposure Representation
```
EV (Exposure Value) = log_2(scene_radiance / reference)
Typical range: e ∈ [-2, +2] EV (4× darker to 4× brighter)

Normalized: ê = e / 2.0  →  ê ∈ [-1, 1]  (better for neural networks)

Photon count relationship:
  Photons(e) = Photons(0) * 2^e
```

### Perturbation Field Definition
For Gaussian i with base parameters (μ_i, Σ_i, α_i, c_i) at exposure e:

```
Δ_i(e) = MLP_Δ(μ_i, ω, ê) → [Δc_i, Δα_i, Δσ_i]

Applied Gaussian at exposure e:
  μ_i(e) = μ_i                          [geometry FIXED]
  α_i(e) = clamp(α_i + Δα_i(e), 0, 1)
  c_i(e) = c_i · (1 + Δc_i(e))         [multiplicative better for tone]
  Σ_i(e) = Σ_i · (1 + Δσ_i(e))²       [scale perturbation]
```

### Loss Function Definitions

**Primary Photometric Loss** (L1 + SSIM):
```
L_photo = (1/N_pixels) * Σ_p [
    (1 - λ_SSIM) * |C_p(e_k) - I_p^GT| 
    + λ_SSIM * (1 - SSIM(C_p(e_k), I_p^GT))
]
```

**Exposure Consistency Loss** (forces geometry to be exposure-invariant):
```
Given two exposures e_1, e_2 from same scene:

L_consist = (1/N_G) * Σ_i ||
    normalize(G_i + Δ_i(e_1)) - normalize(G_i + Δ_i(e_2))
||_L1

where normalize(x) = x / (||x|| + ε)
```

**Smoothness Loss** (exposure transitions should be smooth):
```
L_smooth = λ_smooth * (1/N_G) * Σ_i [
    ||∇_e Δc_i||² + ||∇_e Δα_i||²
]

Numerically (finite differences):
  ∇_e Δc_i ≈ (Δc_i(e+δe) - Δc_i(e-δe)) / (2*δe)
  where δe = 0.2 EV
```

**Magnitude Regularization**:
```
L_reg = λ_reg * (1/N_G) * Σ_i ||[Δc_i, Δα_i, Δσ_i]||²_L2
```

**Total Loss**:
```
L_total = L_photo 
        + λ_consist * L_consist 
        + λ_smooth * L_smooth 
        + λ_reg * L_reg
```

---

## Part 2: Pseudocode Implementation

### 2.1 MLP_Δ Architecture

```python
class PerturbationMLP(nn.Module):
    def __init__(self, hidden_dim=32, max_perturbations=[0.5, 0.25, 0.1]):
        super().__init__()
        # Input: [pos (3), view_dir (2), exposure (1)] = 6D
        # Optional: positional encoding (8D)
        
        self.fc1 = nn.Linear(6, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 7)  # [Δc (3), Δα (1), Δσ (1), etc.]
        
        self.max_delta_c = max_perturbations[0]    # 0.5
        self.max_delta_alpha = max_perturbations[1] # 0.25
        self.max_delta_sigma = max_perturbations[2] # 0.1
        
    def forward(self, position, view_dir, exposure_normalized):
        """
        Args:
            position: [N_G, 3] or [1, 3]
            view_dir: [N_G, 2] or [1, 2] (spherical coords: theta, phi)
            exposure_normalized: [N_G, 1] or [1, 1] (in [-1, 1])
        
        Returns:
            deltas: [N_G, 7] = [Δc_R, Δc_G, Δc_B, Δα, Δσ, reserved1, reserved2]
        """
        x = torch.cat([position, view_dir, exposure_normalized], dim=-1)  # [N, 6]
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc_out(x)  # [N, 7]
        
        # Bound outputs
        delta_c = torch.tanh(out[..., :3]) * self.max_delta_c
        delta_alpha = torch.sigmoid(out[..., 3:4]) * self.max_delta_alpha - 0.125
        delta_sigma = torch.tanh(out[..., 4:5]) * self.max_delta_sigma
        
        return torch.cat([delta_c, delta_alpha, delta_sigma], dim=-1)


def render_with_perturbations(gaussians, perturbation_mlp, exposure, 
                              viewpoint_camera, intrinsics):
    """
    Modified Gaussian splatting render with exposure-dependent perturbations.
    
    Args:
        gaussians: GaussianModel with attributes
                  .positions [N_G, 3]
                  .opacities [N_G, 1]
                  .colors_precomp [N_G, 3] (SH or RGB)
                  .covariance_matrix [N_G, 3, 3]
        
        perturbation_mlp: PerturbationMLP network
        exposure: float or [N_G] (in EV, normalized to [-1, 1])
        viewpoint_camera: camera pose and intrinsics
        intrinsics: K matrix
    
    Returns:
        rendered_image [H, W, 3]
        rendered_depth [H, W, 1]
    """
    
    # Normalize exposure to [-1, 1]
    exposure_norm = torch.clamp(exposure / 2.0, -1, 1)
    
    # Compute perturbations
    positions = gaussians.positions  # [N_G, 3]
    view_dirs = compute_view_directions(positions, viewpoint_camera)  # [N_G, 2]
    
    if isinstance(exposure_norm, torch.Tensor):
        if exposure_norm.dim() == 0:
            exposure_norm = exposure_norm.unsqueeze(0).repeat(len(positions), 1)
    else:
        exposure_norm = torch.tensor([exposure_norm]).repeat(len(positions), 1)
    
    perturbations = perturbation_mlp(positions, view_dirs, exposure_norm)  # [N_G, 7]
    
    # Extract deltas
    delta_c = perturbations[:, :3]         # [N_G, 3]
    delta_alpha = perturbations[:, 3:4]    # [N_G, 1]
    delta_sigma = perturbations[:, 4:5]    # [N_G, 1]
    
    # Apply perturbations to Gaussians
    perturbed_colors = gaussians.colors_precomp * (1.0 + delta_c)
    perturbed_opacities = torch.clamp(gaussians.opacities + delta_alpha, 0, 1)
    perturbed_covars = gaussians.covariance_matrix * ((1.0 + delta_sigma) ** 2)
    
    # Standard Gaussian splatting rendering with perturbed parameters
    rendered = gsplat.render_gaussians(
        positions=positions,
        covariances=perturbed_covars,
        opacities=perturbed_opacities,
        colors=perturbed_colors,
        viewpoint_camera=viewpoint_camera,
        intrinsics=intrinsics
    )
    
    return rendered


def compute_exposure_consistency_loss(gaussians, perturbation_mlp, 
                                     exposure_pairs, lambda_weight=0.5):
    """
    Enforce that geometry is exposure-invariant.
    
    Args:
        gaussians: GaussianModel
        perturbation_mlp: PerturbationMLP
        exposure_pairs: list of (e1, e2) tuples to compare
        lambda_weight: loss weight
    
    Returns:
        scalar loss
    """
    loss = 0.0
    
    for e1, e2 in exposure_pairs:
        # Get perturbed Gaussians at two exposures
        g1 = gaussians.positions + perturbation_mlp(
            gaussians.positions, 
            compute_view_directions(gaussians.positions, None),
            torch.tensor([e1 / 2.0])
        )[:, :3]
        
        g2 = gaussians.positions + perturbation_mlp(
            gaussians.positions, 
            compute_view_directions(gaussians.positions, None),
            torch.tensor([e2 / 2.0])
        )[:, :3]
        
        # Normalize to make rotation/scale invariant
        g1_norm = g1 / (torch.norm(g1, dim=-1, keepdim=True) + 1e-8)
        g2_norm = g2 / (torch.norm(g2, dim=-1, keepdim=True) + 1e-8)
        
        # L1 distance
        loss += torch.mean(torch.abs(g1_norm - g2_norm))
    
    return lambda_weight * loss


def compute_smoothness_loss(perturbation_mlp, positions, view_dirs, 
                           exposures, delta_e=0.2, lambda_weight=0.1):
    """
    Regularize that exposure transitions are smooth.
    
    Args:
        perturbation_mlp: MLP
        positions: [N_G, 3]
        view_dirs: [N_G, 2]
        exposures: list of exposure values to sample
        delta_e: step for finite difference
        lambda_weight: loss weight
    
    Returns:
        scalar loss
    """
    loss = 0.0
    
    for e_center in exposures:
        e_plus = (e_center + delta_e) / 2.0
        e_minus = (e_center - delta_e) / 2.0
        
        delta_plus = perturbation_mlp(positions, view_dirs, 
                                      torch.ones_like(positions[:, :1]) * e_plus)
        delta_minus = perturbation_mlp(positions, view_dirs, 
                                       torch.ones_like(positions[:, :1]) * e_minus)
        
        # Gradient w.r.t. exposure
        grad_e = (delta_plus - delta_minus) / (2 * delta_e)
        
        # Smooth (low gradient)
        loss += torch.mean(grad_e ** 2)
    
    return lambda_weight * loss


# Training loop pseudocode
def train_epoch(gaussians, perturbation_mlp, train_dataset, 
                optimizer_g, optimizer_mlp, epoch, total_epochs):
    """
    Single epoch of training.
    """
    
    # Determine exposure curriculum phase
    phase = min(3, epoch // 300 + 1)
    exposure_range = [1.0, 2.0, 3.0][phase - 1]  # EV
    
    # Exposure curriculum annealing
    lambda_curriculum = 10.0 * math.exp(-epoch / 300.0)
    
    for batch_idx, batch in enumerate(train_dataset.get_batches_stratified(
        exposure_range, phase, batch_size=32)):
        
        losses = {}
        total_loss = 0.0
        
        for image, exposure_ev, camera_pose, intrinsics in batch:
            # Forward: render at exposure
            rendered = render_with_perturbations(
                gaussians, perturbation_mlp, exposure_ev, 
                camera_pose, intrinsics
            )
            
            # Loss: photometric
            loss_photo = F.l1_loss(rendered, image)
            loss_ssim = 1.0 - compute_ssim(rendered, image)
            loss_photo = 0.8 * loss_photo + 0.2 * loss_ssim
            
            # Loss: consistency (sample two exposures from batch)
            if batch_idx % 3 == 0:  # Periodic
                e1, e2 = exposure_ev, exposure_ev + 0.5
                loss_consist = compute_exposure_consistency_loss(
                    gaussians, perturbation_mlp, [(e1, e2)], 
                    lambda_weight=0.5
                )
            else:
                loss_consist = 0.0
            
            # Loss: smoothness
            exposures = [exposure_ev - 0.5, exposure_ev, exposure_ev + 0.5]
            loss_smooth = compute_smoothness_loss(
                perturbation_mlp, gaussians.positions, 
                None, exposures, delta_e=0.2, lambda_weight=0.1
            )
            
            # Loss: regularization
            loss_reg = torch.mean(torch.norm(
                perturbation_mlp(gaussians.positions, None, 
                               torch.zeros_like(gaussians.positions[:, :1])), 
                p=2, dim=-1
            )) * 1e-4
            
            # Loss: curriculum
            loss_curr = lambda_curriculum * (exposure_ev - 0.0) ** 2
            
            # Total
            loss = loss_photo + loss_consist + loss_smooth + loss_reg + loss_curr
            
            losses['photo'] = loss_photo.item()
            losses['consist'] = loss_consist.item() if isinstance(loss_consist, torch.Tensor) else 0.0
            losses['smooth'] = loss_smooth.item()
            losses['reg'] = loss_reg.item()
            losses['total'] = loss.item()
            
            total_loss = total_loss + loss
        
        # Backward
        optimizer_g.zero_grad()
        optimizer_mlp.zero_grad()
        total_loss.backward()
        optimizer_g.step()
        optimizer_mlp.step()
        
        # 3DGS pruning/splitting (on base Gaussians only)
        if epoch % 100 == 0:
            prune_and_split(gaussians)
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: "
                  f"Photo={losses['photo']:.4f}, "
                  f"Consist={losses['consist']:.4f}, "
                  f"Smooth={losses['smooth']:.4f}, "
                  f"Reg={losses['reg']:.6f}")
```

---

## Part 3: Inference Code

### 3.1 Rendering at Target Exposure

```python
def render_at_exposure(gaussians, perturbation_mlp, target_exposure, 
                       viewpoint_camera, intrinsics, tone_mapping='aces'):
    """
    Render Gaussian splatting at a target exposure.
    
    Args:
        target_exposure: float in EV (can be beyond training range)
        tone_mapping: 'aces', 'linear', or None
    
    Returns:
        image [H, W, 3] in sRGB (if tone mapped) or linear (if not)
    """
    
    # Render
    rendered_linear = render_with_perturbations(
        gaussians, perturbation_mlp, target_exposure, 
        viewpoint_camera, intrinsics
    )  # Shape: [H, W, 3], linear space
    
    # Tone map if requested
    if tone_mapping == 'aces':
        # ACES tone mapping curve
        rendered_display = aces_tonemap(rendered_linear * 50.0)  # 50.0 is virtual gain
    elif tone_mapping == 'linear':
        rendered_display = torch.clamp(rendered_linear, 0, 1)
    else:
        rendered_display = rendered_linear
    
    return rendered_display


def interpolate_exposures(gaussians, perturbation_mlp, e_start, e_end, 
                         num_steps, viewpoint_camera, intrinsics):
    """
    Generate an exposure interpolation sequence.
    """
    exposures = torch.linspace(e_start, e_end, num_steps)
    images = []
    
    for e in exposures:
        img = render_at_exposure(gaussians, perturbation_mlp, e.item(),
                                viewpoint_camera, intrinsics)
        images.append(img.cpu().numpy())
    
    return images


def aces_tonemap(x):
    """
    ACES tone mapping curve.
    """
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x = (x * (a * x + b)) / (x * (c * x + d) + e)
    return torch.clamp(x, 0, 1)
```

---

## Part 4: Dataset Preparation

### 4.1 Loading and Preprocessing

```python
def load_lom_dataset_with_exposures(dataset_root, scene_name):
    """
    Load LOM dataset and estimate/infer exposures.
    """
    
    low_light_dir = os.path.join(dataset_root, scene_name, 'low_light')
    normal_light_dir = os.path.join(dataset_root, scene_name, 'normal_light')
    
    images_low = []
    images_normal = []
    cameras_low = []
    cameras_normal = []
    exposures_low = []
    exposures_normal = []
    
    # Load low-light images
    for img_file in sorted(os.listdir(low_light_dir)):
        img = cv2.imread(os.path.join(low_light_dir, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        images_low.append(img)
        
        # Load corresponding camera pose
        cam_file = img_file.replace('.jpg', '.json')
        with open(os.path.join(low_light_dir, cam_file)) as f:
            cam_data = json.load(f)
        cameras_low.append(cam_data)
    
    # Load normal-light images (same poses)
    for img_file in sorted(os.listdir(normal_light_dir)):
        img = cv2.imread(os.path.join(normal_light_dir, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        images_normal.append(img)
    
    # Estimate exposures from image brightness
    for img_low in images_low:
        # Mean luminance
        lum = 0.299 * img_low[..., 0] + 0.587 * img_low[..., 1] + 0.114 * img_low[..., 2]
        mean_lum = np.mean(lum)
        
        # Estimate exposure relative to normal-light (assume normal is e=0)
        # e_low ≈ log_2(mean_lum_low / mean_lum_normal)
        # For simplicity, assume normal-light has lum ~0.5
        e_ev = np.log2(mean_lum / 0.5)
        exposures_low.append(e_ev)
    
    # Normal-light exposures
    for img_normal in images_normal:
        lum = 0.299 * img_normal[..., 0] + 0.587 * img_normal[..., 1] + 0.114 * img_normal[..., 2]
        mean_lum = np.mean(lum)
        e_ev = np.log2(mean_lum / 0.5)
        exposures_normal.append(e_ev)
    
    return {
        'images_low': images_low,
        'images_normal': images_normal,
        'cameras': cameras_low,  # Same for both
        'exposures': exposures_low + exposures_normal,
        'all_images': images_low + images_normal
    }
```

---

## Part 5: Validation and Metrics

```python
def evaluate_exposure_consistency(gaussians, perturbation_mlp, 
                                 test_camera, e_samples=[-1, 0, 1]):
    """
    Evaluate that rendering is exposure-consistent.
    """
    
    renderings = []
    for e in e_samples:
        img = render_at_exposure(gaussians, perturbation_mlp, e, 
                                test_camera, None)
        renderings.append(img)
    
    # Compute normalized renderings
    renderings_norm = [r / (torch.norm(r) + 1e-8) for r in renderings]
    
    # Consistency metric: how different are normalized versions?
    consistency = 0.0
    for i in range(len(renderings_norm) - 1):
        diff = torch.mean(torch.abs(renderings_norm[i] - renderings_norm[i+1]))
        consistency += diff
    
    return consistency / (len(renderings_norm) - 1)


def evaluate_psnr_ssim(gaussians, perturbation_mlp, test_images, 
                      test_exposures, test_cameras):
    """
    Compute PSNR and SSIM on test set.
    """
    
    psnrs = []
    ssims = []
    
    for img_gt, exp, cam in zip(test_images, test_exposures, test_cameras):
        rendered = render_at_exposure(gaussians, perturbation_mlp, exp, cam, None)
        
        psnr = compute_psnr(rendered.cpu().numpy(), img_gt)
        ssim = compute_ssim(rendered, torch.tensor(img_gt))
        
        psnrs.append(psnr)
        ssims.append(ssim)
    
    return np.mean(psnrs), np.mean(ssims)
```

---

## Part 6: Hyperparameter Summary

| Hyperparameter | Recommended Value | Notes |
|----------------|-------------------|-------|
| MLP hidden dim | 32 | Small, fast |
| MLP layers | 2 | Minimal |
| max_delta_c | 0.5 | Max color perturbation |
| max_delta_alpha | 0.25 | Max opacity change |
| max_delta_sigma | 0.1 | Max scale change (±10%) |
| λ_photo | 1.0 | Primary loss weight |
| λ_consist | 0.5 | Consistency weight (critical) |
| λ_smooth | 0.1 | Smoothness weight |
| λ_reg | 1e-4 | Magnitude regularization |
| λ_curriculum_init | 10.0 | Initial curriculum weight |
| curriculum_decay_tau | 300 | Epoch decay constant |
| lr_gaussian | 1.6e-4 | Same as standard 3DGS |
| lr_mlp | 1e-4 | Slightly lower |
| Phase 1 exposure range | ±1 EV | Epochs 1-300 |
| Phase 2 exposure range | ±2 EV | Epochs 300-600 |
| Phase 3 exposure range | ±3 EV | Epochs 600-1000 |
| Batch size | 32 | Standard |

---

## Part 7: Checklist for Publication

- [ ] Implement MLP_Δ architecture
- [ ] Modify render() to apply perturbations
- [ ] Implement L_consist loss with exposure pairs
- [ ] Implement L_smooth with finite differences
- [ ] Add exposure curriculum scheduling
- [ ] Test on LOM low-light subset
- [ ] Generate ablations (each loss term on/off)
- [ ] Write section on exposure interpolation/extrapolation
- [ ] Create qualitative comparison images (low → normal light)
