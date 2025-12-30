# Mathematical Differences: HDR Gaussian Splatting vs Lowlight Enhancement Gaussian Splatting

This document outlines the core mathematical differences between theoretical HDR Gaussian Splatting approaches and the actual implementation in `lowlight_enhance/simple_trainer.py` (Lowlight Enhancement Gaussian Splatting).

## 1. Color Space Conversion

### Theoretical HDR Approach
Images are converted from sRGB to linear space before training:

$$
I_{\text{linear}} = \begin{cases}
\frac{I_{\text{sRGB}}}{12.92} & \text{if } I_{\text{sRGB}} \leq 0.04045 \\
\left(\frac{I_{\text{sRGB}} + 0.055}{1.055}\right)^{2.4} & \text{otherwise}
\end{cases}
$$

**Purpose**: Enables proper HDR radiance representation. sRGB is a non-linear encoding optimized for display, while linear space preserves physical radiance relationships.

### Lowlight Enhancement Trainer (`lowlight_enhance/simple_trainer.py`)
Images are used directly (assumed sRGB):

$$
I = \frac{I_{\text{uint8}}}{255.0}
$$

**No conversion**: Works in sRGB space directly, suitable for LDR (Low Dynamic Range) images. The multi-exposure approach handles varying brightness through exposure-aware perturbations rather than color space conversion.

---

## 2. Rendering Equation

### Theoretical HDR Approach
Rendering includes per-image exposure scaling:

$$
C_{\text{rendered}} = R \cdot E_i
$$

where:
- $R$ = Raw HDR radiance from Gaussian splatting
- $E_i$ = Per-image exposure parameter (learned)
- $C_{\text{rendered}}$ = Final rendered color

The exposure $E_i$ is optimized per training image to account for varying camera exposures.

### Lowlight Enhancement Trainer (`lowlight_enhance/simple_trainer.py`)
Rendering uses exposure-aware perturbations:

$$
C_{\text{rendered}} = \text{Render}(\mathbf{G} + \Delta(\mathbf{G}, e, \mathbf{v}))
$$

where:
- $\mathbf{G}$ = Base Gaussian parameters (means, scales, opacities, colors)
- $\Delta(\mathbf{G}, e, \mathbf{v})$ = Perturbation from PerturbationMLP
- $e$ = Current exposure value (in EV, normalized to $[-1, 1]$)
- $\mathbf{v}$ = View direction vector

The perturbation $\Delta$ modifies Gaussian parameters based on exposure:
$$
\Delta = [\Delta_{\text{means}}, \Delta_{\text{scales}}, \Delta_{\text{opacities}}, \Delta_{\text{colors}}]
$$

**Key Difference**: Instead of scaling rendered radiance, the lowlight enhancement approach perturbs the 3D Gaussian parameters themselves, allowing geometry and appearance to adapt to exposure conditions.

---

## 3. Loss Functions

### Theoretical HDR Approach

#### Primary Loss: Poisson-Gaussian NLL Loss
$$
\mathcal{L}_{\text{NLL}} = \frac{1}{N} \sum_{p} \left[ \frac{(C_p - \hat{C}_p)^2}{\hat{C}_p + \epsilon} + \log(\hat{C}_p + \epsilon) \right]
$$

where:
- $C_p$ = Ground truth pixel value (linear space)
- $\hat{C}_p$ = Rendered pixel value (detached for stability)
- $\epsilon = 10^{-3}$ = Small epsilon for numerical stability
- $N$ = Number of pixels

**Rationale**: Models photon noise (Poisson) and sensor noise (Gaussian) inherent in HDR imaging.

#### Exposure Ratio Loss (HDR Consistency):
$$
\mathcal{L}_{\text{ratio}} = \frac{1}{N} \sum_{p} \left| \frac{R_{i,p}}{E_i} - \frac{R_{j,p}}{E_j} \right|
$$

where:
- $R_{i,p}, R_{j,p}$ = Raw radiance from two different camera views
- $E_i, E_j$ = Corresponding exposure parameters

**Purpose**: Enforces that normalized radiance (radiance divided by exposure) should be consistent across different camera views.

### Lowlight Enhancement Trainer (`lowlight_enhance/simple_trainer.py`)

#### Primary Loss: L1 + SSIM
$$
\mathcal{L}_{\text{L1}} = \frac{1}{N} \sum_{p} |C_p - \hat{C}_p|
$$

$$
\mathcal{L}_{\text{SSIM}} = 1 - \text{SSIM}(C, \hat{C})
$$

$$
\mathcal{L}_{\text{photo}} = (1 - \lambda_{\text{SSIM}}) \cdot \mathcal{L}_{\text{L1}} + \lambda_{\text{SSIM}} \cdot \mathcal{L}_{\text{SSIM}}
$$

where:
- $C_p$ = Ground truth pixel value (sRGB space)
- $\hat{C}_p$ = Rendered pixel value
- $\lambda_{\text{SSIM}} = 0.2$ = Weight for SSIM loss (default)
- Uses `fused_ssim` with `padding="valid"` for efficiency

**Note**: The implementation uses L1 + SSIM instead of Poisson-Gaussian NLL loss, which is more suitable for LDR images and provides better perceptual quality.

#### Structure-Preserving Loss (Gradient/Edge Loss)
When `cfg.enable_multi_exposure = True` and `cfg.lambda_structure > 0`:

$$
\mathcal{L}_{\text{structure}} = \lambda_{\text{structure}} \cdot \frac{1}{N} \sum_{p} |\nabla C_p - \nabla \hat{C}_p|
$$

where:
- $\nabla C_p$ = Image gradient (computed via Sobel operator) at pixel $p$
- $\lambda_{\text{structure}} = 2.0$ = Loss weight (default)

**Purpose**: Preserves image structure (edges, gradients) across different exposures, ensuring that brightness changes don't destroy image details.

#### Exposure Consistency Loss
When `cfg.enable_multi_exposure = True`:

$$
\mathcal{L}_{\text{consist}} = \lambda_{\text{consist}} \cdot \frac{1}{N_G} \sum_{n} \left\| \frac{\Delta(\mathbf{g}_n, e_1, \mathbf{0})}{\|\Delta(\mathbf{g}_n, e_1, \mathbf{0})\|} - \frac{\Delta(\mathbf{g}_n, e_2, \mathbf{0})}{\|\Delta(\mathbf{g}_n, e_2, \mathbf{0})\|} \right\|_1
$$

where:
- $\mathbf{g}_n$ = Gaussian $n$ position
- $e_1, e_2$ = Two different exposure values
- $\Delta(\cdot)$ = Perturbation from PerturbationMLP
- $\lambda_{\text{consist}} = 5.0$ = Loss weight (default)
- $N_G$ = Number of sampled Gaussians (max 1000 for efficiency)

**Purpose**: Enforces that geometry perturbations are exposure-invariant (normalized perturbations should be similar across exposures). This ensures the underlying 3D structure remains consistent regardless of exposure.

#### Smoothness Loss (Exposure Transitions)
$$
\mathcal{L}_{\text{smooth}} = \lambda_{\text{smooth}} \cdot \frac{1}{N_G} \sum_{n} \left\| \frac{\partial \Delta(\mathbf{g}_n, e, \mathbf{v}_n)}{\partial e} \right\|^2
$$

where the gradient is approximated via finite differences:
$$
\frac{\partial \Delta}{\partial e} \approx \frac{\Delta(e + \delta_e) - \Delta(e - \delta_e)}{2\delta_e}
$$

with:
- $\delta_e = 0.2$ EV = Step size for finite difference
- $\lambda_{\text{smooth}} = 10.0$ = Loss weight (default)

**Purpose**: Ensures that exposure transitions are smooth, preventing abrupt changes in appearance when exposure varies.

#### Regularization Loss (Perturbation Magnitude)
$$
\mathcal{L}_{\text{reg}} = \lambda_{\text{reg}} \cdot \frac{1}{N_G} \sum_{n} \|\Delta(\mathbf{g}_n, e, \mathbf{v}_n)\|_2
$$

where:
- $\lambda_{\text{reg}} = 0.01$ = Loss weight (default)

**Purpose**: Prevents perturbations from becoming too large, maintaining stability during training.

#### Curriculum Loss (Exposure Scheduling)
$$
\mathcal{L}_{\text{curriculum}} = \lambda_{\text{curriculum}}(t) \cdot (e_{\text{current}} - e_{\text{target}})^2
$$

where:
- $\lambda_{\text{curriculum}}(t) = \lambda_{\text{init}} \cdot \exp(-t / \tau)$ = Decaying weight
- $\lambda_{\text{init}} = 10.0$ = Initial weight
- $\tau = 300$ = Decay time constant
- $e_{\text{target}} = 0.0$ = Target exposure (neutral)

**Purpose**: Gradually guides the model toward neutral exposure during training, implementing curriculum learning.

#### Reflectance Consistency Loss (Intrinsic Factorization)
When `cfg.enable_intrinsic_factorization = True`:

$$
\mathcal{L}_{\text{reflect}} = \lambda_{\text{reflect}} \cdot \frac{1}{N_G} \sum_{n} \sum_{i<j} \left\| \frac{I_{n,i}}{L(e_i) \odot V(\mathbf{v}_n)} - \frac{I_{n,j}}{L(e_j) \odot V(\mathbf{v}_n)} \right\|^2
$$

where:
- $I_{n,k}$ = Observed intensity for Gaussian $n$ at exposure $e_k$
- $L(e_k)$ = Illumination at exposure $e_k$ (from IlluminationMLP)
- $V(\mathbf{v}_n)$ = Visibility for view direction $\mathbf{v}_n$ (from VisibilityMLP)
- $\lambda_{\text{reflect}} = 10.0$ = Loss weight (default)

**Purpose**: Enforces that reflectance $R$ is consistent across multiple exposures, ensuring physically plausible intrinsic decomposition: $I = R \odot L(e) \odot V(\mathbf{v})$.

#### Illumination Smoothness Loss
$$
\mathcal{L}_{\text{illum-smooth}} = \lambda_{\text{illum-smooth}} \cdot \left\| \frac{\partial^2 \log L(e)}{\partial e^2} \right\|^2
$$

where the second derivative is approximated via finite differences:
$$
\frac{\partial^2 \log L}{\partial e^2} \approx \frac{\log L(e+\delta) - 2\log L(e) + \log L(e-\delta)}{\delta^2}
$$

with:
- $\lambda_{\text{illum-smooth}} = 1.0$ = Loss weight (default)

**Purpose**: Ensures illumination changes smoothly with exposure.

#### Visibility Smoothness Loss
$$
\mathcal{L}_{\text{vis-smooth}} = \lambda_{\text{vis-smooth}} \cdot \frac{1}{N_{\text{pairs}}} \sum_{(\mathbf{v}_a, \mathbf{v}_b)} \exp(-d_{ab}) \cdot \|V(\mathbf{v}_a) - V(\mathbf{v}_b)\|^2
$$

where:
- $d_{ab}$ = Distance between view directions $\mathbf{v}_a$ and $\mathbf{v}_b$
- $\lambda_{\text{vis-smooth}} = 0.1$ = Loss weight (default)

**Purpose**: Encourages smooth visibility changes for nearby views.

#### Reflectance Spatial Smoothness Loss
$$
\mathcal{L}_{\text{reflect-spatial}} = \lambda_{\text{reflect-spatial}} \cdot \frac{1}{N_G} \sum_{n} \frac{1}{K} \sum_{k \in \mathcal{N}_n} \|R_n - R_k\|^2
$$

where:
- $\mathcal{N}_n$ = Set of $K$ nearest neighbors of Gaussian $n$
- $R_n$ = Reflectance of Gaussian $n$
- $\lambda_{\text{reflect-spatial}} = 1.0$ = Loss weight (default)

**Purpose**: Enforces spatial smoothness on reflectance, ensuring neighboring Gaussians have similar material properties.

#### Total Loss (Lowlight Enhancement Trainer)
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{photo}} + \mathcal{L}_{\text{structure}} + \mathcal{L}_{\text{consist}} + \mathcal{L}_{\text{smooth}} + \mathcal{L}_{\text{reg}} + \mathcal{L}_{\text{curriculum}} + \mathcal{L}_{\text{reflect}} + \mathcal{L}_{\text{illum-smooth}} + \mathcal{L}_{\text{vis-smooth}} + \mathcal{L}_{\text{reflect-spatial}}
$$

where each term is only included if its corresponding flag is enabled and weight > 0.

---

## 4. Multi-Exposure Training

### Lowlight Enhancement Trainer (`lowlight_enhance/simple_trainer.py`)

#### Exposure Curriculum Learning
The training uses a curriculum learning approach with multiple phases:

$$
e_{\text{current}} \in [0, E_{\text{phase}}]
$$

where $E_{\text{phase}}$ increases over training phases:
- Phase 1 (steps 0-3000): $E_{\text{phase}} = 1.0$ EV
- Phase 2 (steps 3000-10000): $E_{\text{phase}} = 2.0$ EV  
- Phase 3 (steps 10000-20000): $E_{\text{phase}} = 3.0$ EV
- Phase 4 (steps 20000+): $E_{\text{phase}} = 3.0$ EV (fixed)

At each step, exposure is sampled uniformly:
$$
e \sim \mathcal{U}(0, E_{\text{phase}})
$$

**Purpose**: Gradually increases exposure range during training, allowing the model to learn exposure adaptation progressively.

#### PerturbationMLP Architecture
The PerturbationMLP takes as input:
- Gaussian positions: $\mathbf{g} \in \mathbb{R}^3$
- View directions: $\mathbf{v} \in \mathbb{R}^2$ (normalized)
- Exposure: $e \in [-1, 1]$ (normalized from EV)

Outputs perturbations:
$$
\Delta = \text{MLP}(\mathbf{g}, \mathbf{v}, e) \in \mathbb{R}^5
$$

where $\Delta$ is clamped to:
- $\Delta_{\text{means}} \in [-0.1, 0.1]$ (position perturbation)
- $\Delta_{\text{scales}} \in [-0.05, 0.05]$ (scale perturbation)
- $\Delta_{\text{opacities}} \in [-0.05, 0.05]$ (opacity perturbation)
- $\Delta_{\text{colors}} \in [-0.1, 0.1]$ (color perturbation)

**Architecture**: MLP with hidden dimension 32 (default).

---

## 5. Intrinsic Factorization

### Lowlight Enhancement Trainer (`lowlight_enhance/simple_trainer.py`)

When `cfg.enable_intrinsic_factorization = True`, the model decomposes appearance into:

$$
I = R \odot L(e) \odot V(\mathbf{v})
$$

where:
- $R \in \mathbb{R}^3$ = Reflectance (albedo) - stored as Gaussian parameter
- $L(e) \in \mathbb{R}^3$ = Illumination at exposure $e$ - from IlluminationMLP
- $V(\mathbf{v}) \in \mathbb{R}^3$ = Visibility for view direction $\mathbf{v}$ - from VisibilityMLP

#### IlluminationMLP
$$
L(e) = \text{MLP}_{\text{illum}}(e_{\text{norm}})
$$

where:
- Input: Normalized exposure $e_{\text{norm}} = \text{clamp}(e / 2.0, -1, 1)$
- Output: RGB illumination values
- Hidden dimension: 32 (default)

#### VisibilityMLP
$$
V(\mathbf{v}) = \text{MLP}_{\text{vis}}(\mathbf{v})
$$

where:
- Input: View direction $\mathbf{v} \in \mathbb{R}^2$ (normalized)
- Output: RGB visibility values
- Hidden dimension: 32 (default)

**Purpose**: Separates material properties (reflectance) from lighting conditions (illumination) and view-dependent effects (visibility), enabling better generalization across exposures and viewpoints.

---

## 6. Dataset Preprocessing

### Theoretical HDR Approach
1. Load image: $I_{\text{uint8}} \in [0, 255]$
2. Normalize: $I_{\text{sRGB}} = I_{\text{uint8}} / 255.0$
3. **Convert to linear**: $I_{\text{linear}} = \text{sRGB-to-linear}(I_{\text{sRGB}})$
4. Use $I_{\text{linear}}$ for training

### Lowlight Enhancement Trainer (`lowlight_enhance/simple_trainer.py`)
1. Load image: $I_{\text{uint8}} \in [0, 255]$
2. Normalize: $I = I_{\text{uint8}} / 255.0$
3. Use $I$ directly (assumed sRGB)
4. **Exposure is sampled during training** (not from image metadata)

**Key Difference**: Exposure values are not learned per-image but are sampled dynamically during training as part of the curriculum learning strategy.

---

## 7. Summary Table

| Aspect | Theoretical HDR Approach | Lowlight Enhancement Trainer (`lowlight_enhance/simple_trainer.py`) |
|--------|-------------------------|----------------------------------------------------------------------|
| **Color Space** | Linear (sRGB → linear conversion) | sRGB (direct) |
| **Rendering** | $C = R \cdot E_i$ (exposure scaling) | $C = \text{Render}(\mathbf{G} + \Delta(\mathbf{G}, e, \mathbf{v}))$ (perturbation-based) |
| **Primary Loss** | Poisson-Gaussian NLL | L1 + SSIM |
| **Exposure Handling** | Per-image learned parameters | Dynamic sampling with curriculum learning |
| **Exposure Loss** | Ratio loss (radiance consistency) | Consistency loss (geometry invariance) |
| **Structure Preservation** | Not explicitly modeled | Gradient/edge loss |
| **Multi-Exposure Training** | Not implemented | Curriculum learning with phases |
| **Perturbation MLP** | Not implemented | Exposure-aware Gaussian perturbations |
| **Intrinsic Factorization** | Not implemented | Reflectance × Illumination × Visibility |
| **Smoothness Regularization** | Not implemented | Exposure smoothness, illumination smoothness, visibility smoothness |
| **Spatial Regularization** | Not implemented | Reflectance spatial smoothness |
| **Use Case** | HDR scenes, varying exposures | Lowlight scenes, multi-exposure enhancement |

---

## 8. Key Mathematical Insights

### Why Perturbation-Based Approach?
Instead of scaling rendered radiance, the lowlight enhancement approach perturbs Gaussian parameters:

**Advantages**:
1. **Geometry Adaptation**: Allows 3D structure to adapt to exposure (e.g., brighter regions may need more Gaussians)
2. **Appearance Control**: Fine-grained control over opacity and color per Gaussian
3. **Exposure Invariance**: Consistency loss ensures underlying geometry remains stable
4. **Smooth Transitions**: Smoothness loss prevents abrupt changes

### Why Intrinsic Factorization?
Decomposing appearance into $I = R \odot L(e) \odot V(\mathbf{v})$ provides:

1. **Material Consistency**: Reflectance $R$ is exposure-invariant
2. **Lighting Modeling**: Illumination $L(e)$ captures exposure-dependent lighting
3. **View Effects**: Visibility $V(\mathbf{v})$ models view-dependent effects (specularity, etc.)
4. **Generalization**: Better generalization to novel exposures and viewpoints

### Why Curriculum Learning?
Gradually increasing exposure range:

$$
E_{\text{phase}}(t) = \begin{cases}
1.0 & \text{if } t < 3000 \\
2.0 & \text{if } 3000 \leq t < 10000 \\
3.0 & \text{if } t \geq 10000
\end{cases}
$$

**Benefits**:
- Starts with easier examples (small exposure range)
- Progressively increases difficulty
- Prevents early training instability
- Allows model to learn exposure adaptation gradually

### Why Structure-Preserving Loss?
Gradient-based loss preserves image structure:

$$
\mathcal{L}_{\text{structure}} = \|\nabla I - \nabla \hat{I}\|
$$

**Purpose**: Ensures that brightness enhancement doesn't destroy edges, textures, or fine details. Critical for lowlight enhancement where detail preservation is essential.

---

## 9. Additional Mathematical Details

### Exposure Normalization
Exposure values (in EV) are normalized to $[-1, 1]$:

$$
e_{\text{norm}} = \text{clamp}\left(\frac{e_{\text{EV}}}{2.0}, -1, 1\right)
$$

This ensures the MLP inputs are in a standard range, improving training stability.

### View Direction Computation
View directions are computed from Gaussian positions and camera pose:

$$
\mathbf{v}_n = \text{normalize}\left(\frac{\mathbf{g}_n - \mathbf{c}}{\|\mathbf{g}_n - \mathbf{c}\|}\right)
$$

where $\mathbf{c}$ is the camera center. The direction is then projected to 2D for MLP input.

### Gaussian Sampling for Losses
To reduce computational cost, losses are computed on a subset of Gaussians:

$$
N_{\text{sample}} = \min(1000, N_G)
$$

Gaussians are randomly sampled each iteration, ensuring all Gaussians contribute over time while keeping training efficient.

### Loss Computation Frequency
Some losses are computed periodically to reduce overhead:
- Consistency loss: Every `consist_loss_freq` steps (default: 1)
- Reflectance consistency loss: Every `reflect_loss_freq` steps (default: 5)
- Illumination smoothness: Every 2 steps
- Reflectance spatial smoothness: Every 10 steps

---

## References

- 3D Gaussian Splatting: Original paper on Gaussian splatting
- Multi-Exposure Image Enhancement: Techniques for handling varying exposures
- Intrinsic Image Decomposition: Reflectance and illumination separation
- Curriculum Learning: Progressive training strategies
- Structure-Preserving Image Enhancement: Gradient-based preservation methods
