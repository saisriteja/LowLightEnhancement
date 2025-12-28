# Mathematical Differences: HDR Gaussian Splatting vs Standard Gaussian Splatting

This document outlines the core mathematical differences between `train.py` (HDR Gaussian Splatting) and `simple_trainer.py` (Standard Gaussian Splatting).

## 1. Color Space Conversion

### HDR Trainer (`train.py`)
Images are converted from sRGB to linear space before training:

\[
I_{\text{linear}} = \begin{cases}
\frac{I_{\text{sRGB}}}{12.92} & \text{if } I_{\text{sRGB}} \leq 0.04045 \\
\left(\frac{I_{\text{sRGB}} + 0.055}{1.055}\right)^{2.4} & \text{otherwise}
\end{cases}
\]

**Purpose**: Enables proper HDR radiance representation. sRGB is a non-linear encoding optimized for display, while linear space preserves physical radiance relationships.

### Standard Trainer (`simple_trainer.py`)
Images are used directly (assumed sRGB):

\[
I = \frac{I_{\text{uint8}}}{255.0}
\]

**No conversion**: Works in sRGB space directly, suitable for LDR (Low Dynamic Range) images.

---

## 2. Rendering Equation

### HDR Trainer (`train.py`)
Rendering includes per-image exposure scaling:

\[
C_{\text{rendered}} = R \cdot E_i
\]

where:
- \(R\) = Raw HDR radiance from Gaussian splatting
- \(E_i\) = Per-image exposure parameter (learned)
- \(C_{\text{rendered}}\) = Final rendered color

The exposure \(E_i\) is optimized per training image to account for varying camera exposures.

### Standard Trainer (`simple_trainer.py`)
Direct rendering without exposure:

\[
C_{\text{rendered}} = R
\]

No exposure compensation - assumes uniform exposure across all images.

---

## 3. Loss Functions

### HDR Trainer (`train.py`)

#### Primary Loss: Poisson-Gaussian NLL Loss
\[
\mathcal{L}_{\text{NLL}} = \frac{1}{N} \sum_{p} \left[ \frac{(C_p - \hat{C}_p)^2}{\hat{C}_p + \epsilon} + \log(\hat{C}_p + \epsilon) \right]
\]

where:
- \(C_p\) = Ground truth pixel value (linear space)
- \(\hat{C}_p\) = Rendered pixel value (detached for stability)
- \(\epsilon = 10^{-3}\) = Small epsilon for numerical stability
- \(N\) = Number of pixels

**Rationale**: Models photon noise (Poisson) and sensor noise (Gaussian) inherent in HDR imaging. The denominator \(\hat{C}_p + \epsilon\) provides adaptive weighting based on signal strength.

#### Combined Loss (with SSIM):
\[
\mathcal{L} = (1 - \lambda_{\text{SSIM}}) \cdot \mathcal{L}_{\text{NLL}} + \lambda_{\text{SSIM}} \cdot \mathcal{L}_{\text{SSIM}}
\]

#### Exposure Ratio Loss (HDR Consistency):
\[
\mathcal{L}_{\text{ratio}} = \frac{1}{N} \sum_{p} \left| \frac{R_{i,p}}{E_i} - \frac{R_{j,p}}{E_j} \right|
\]

where:
- \(R_{i,p}, R_{j,p}\) = Raw radiance from two different camera views
- \(E_i, E_j\) = Corresponding exposure parameters

**Purpose**: Enforces that normalized radiance (radiance divided by exposure) should be consistent across different camera views, ensuring physically plausible HDR reconstruction.

#### SH Regularization Loss:
\[
\mathcal{L}_{\text{SH}} = \frac{1}{N} \sum_{n} \| \text{SH}_N^{(n)} \|^2
\]

where \(\text{SH}_N^{(n)}\) are the higher-order spherical harmonics coefficients (excluding band 0).

**Purpose**: Separates global illumination (SH band 0) from view-dependent effects (higher bands), encouraging the model to learn physically meaningful lighting.

#### Total Loss:
\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{primary}} + \lambda_{\text{ratio}} \cdot \mathcal{L}_{\text{ratio}} + \lambda_{\text{SH}} \cdot \mathcal{L}_{\text{SH}} + \text{other regularizations}
\]

### Standard Trainer (`simple_trainer.py`)

#### Primary Loss: L1 + SSIM
\[
\mathcal{L}_{\text{L1}} = \frac{1}{N} \sum_{p} |C_p - \hat{C}_p|
\]

\[
\mathcal{L}_{\text{SSIM}} = 1 - \text{SSIM}(C, \hat{C})
\]

\[
\mathcal{L} = (1 - \lambda_{\text{SSIM}}) \cdot \mathcal{L}_{\text{L1}} + \lambda_{\text{SSIM}} \cdot \mathcal{L}_{\text{SSIM}}
\]

**No exposure ratio loss or SH regularization**: Standard LDR reconstruction doesn't require these HDR-specific constraints.

---

## 4. Tone Mapping

### HDR Trainer (`train.py`)
Uses ACES (Academy Color Encoding System) tone mapping for visualization:

\[
T(x) = \text{clamp}\left( \frac{x \cdot (a \cdot x + b)}{x \cdot (c \cdot x + d) + e}, 0, 1 \right)
\]

where:
- \(a = 2.51\), \(b = 0.03\), \(c = 2.43\), \(d = 0.59\), \(e = 0.14\)

Applied during rendering/video generation:
\[
C_{\text{display}} = T(C_{\text{HDR}} \cdot G_{\text{virtual}})
\]

where \(G_{\text{virtual}}\) is a virtual gain factor (default: 50.0) to boost HDR radiance before tone mapping.

### Standard Trainer (`simple_trainer.py`)
Simple clamping:

\[
C_{\text{display}} = \text{clamp}(C, 0, 1)
\]

No tone mapping needed for LDR images.

---

## 5. Exposure Optimization

### HDR Trainer (`train.py`)
Per-image exposure parameters are learned:

\[
E_i = \text{Parameter}(i), \quad i \in \{1, 2, \ldots, N_{\text{images}}\}
\]

Optimized with Adam optimizer:
\[
E_i^{(t+1)} = E_i^{(t)} - \alpha_E \cdot \nabla_{E_i} \mathcal{L}
\]

where \(\alpha_E\) is the exposure learning rate (default: \(10^{-3}\)).

**Initialization**: \(E_i^{(0)} = 1.0\) (neutral exposure)

### Standard Trainer (`simple_trainer.py`)
No exposure optimization - assumes uniform exposure.

---

## 6. Dataset Preprocessing

### HDR Trainer (`train.py`)
1. Load image: \(I_{\text{uint8}} \in [0, 255]\)
2. Normalize: \(I_{\text{sRGB}} = I_{\text{uint8}} / 255.0\)
3. **Convert to linear**: \(I_{\text{linear}} = \text{sRGB-to-linear}(I_{\text{sRGB}})\)
4. Use \(I_{\text{linear}}\) for training

### Standard Trainer (`simple_trainer.py`)
1. Load image: \(I_{\text{uint8}} \in [0, 255]\)
2. Normalize: \(I = I_{\text{uint8}} / 255.0\)
3. Use \(I\) directly (assumed sRGB)

---

## 7. Summary Table

| Aspect | HDR Trainer (`train.py`) | Standard Trainer (`simple_trainer.py`) |
|--------|-------------------------|----------------------------------------|
| **Color Space** | Linear (sRGB → linear conversion) | sRGB (direct) |
| **Rendering** | \(C = R \cdot E_i\) (with exposure) | \(C = R\) (no exposure) |
| **Primary Loss** | Poisson-Gaussian NLL | L1 + SSIM |
| **Exposure Opt.** | Per-image learned parameters | None |
| **Ratio Loss** | Enforces HDR consistency | None |
| **SH Regularization** | Separates global/local lighting | None |
| **Tone Mapping** | ACES tone mapping | Simple clamp |
| **Use Case** | HDR scenes, varying exposures | LDR scenes, uniform exposure |

---

## 8. Key Mathematical Insights

### Why Poisson-Gaussian NLL Loss?
HDR imaging captures photon-limited scenes where noise follows:
\[
\text{Noise} \sim \mathcal{N}(0, \sigma^2) + \text{Poisson}(\lambda = \text{signal})
\]

The NLL loss naturally accounts for this noise model:
\[
\mathcal{L}_{\text{NLL}} \propto \frac{\text{error}^2}{\text{signal}} + \log(\text{signal})
\]

This provides:
- **Adaptive weighting**: Dark regions (low signal) have higher uncertainty
- **Noise-aware**: Matches the physical noise characteristics
- **HDR-appropriate**: Works well with high dynamic range values

### Why Exposure Ratio Loss?
For physically consistent HDR reconstruction:
\[
\frac{R_i}{E_i} = \frac{R_j}{E_j} = L_{\text{scene}}
\]

where \(L_{\text{scene}}\) is the true scene radiance. This ensures the model learns a consistent 3D radiance field regardless of camera exposure settings.

### Why SH Regularization?
Spherical harmonics decomposition:
\[
C(\theta, \phi) = \sum_{l=0}^{L} \sum_{m=-l}^{l} \text{SH}_l^m \cdot Y_l^m(\theta, \phi)
\]

- Band 0 (\(l=0\)): Global illumination (albedo)
- Bands 1+: View-dependent effects (specularity, reflections)

Regularizing higher bands encourages the model to:
- Store global lighting in band 0
- Use higher bands only for view-dependent effects
- Produce more physically plausible results

---

## References

- ACES Tone Mapping: Academy Color Encoding System
- Poisson-Gaussian Noise Model: Common in photon-limited imaging
- Spherical Harmonics: Standard representation for view-dependent appearance

