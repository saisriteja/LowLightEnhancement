## Core Insight: Learn Scene Properties that Transfer Across Exposures

The key is to learn **exposure-invariant scene properties** (geometry, albedo, reflectance) from dark images, then **render at arbitrary exposures**.

---

## 🔥 Advanced Ideas for Low-Light → Well-Lit Transfer

### **1. Intrinsic Decomposition in GS Space**

Decompose the scene into exposure-invariant and exposure-dependent components:

```python
class IntrinsicGaussianSplats(torch.nn.Module):
    """Split Gaussians into albedo (invariant) + illumination (variant)"""
    
    def __init__(self, N, sh_degree):
        super().__init__()
        # Albedo: Exposure-invariant reflectance (what color is the surface?)
        self.albedo_sh0 = torch.nn.Parameter(torch.zeros(N, 1, 3))  # Base color
        self.albedo_shN = torch.nn.Parameter(torch.zeros(N, sh_degree**2-1, 3))  # Reflectance detail
        
        # Illumination: Exposure-dependent lighting (how much light hits it?)
        self.illum_sh0 = torch.nn.Parameter(torch.zeros(N, 1, 3))  # Global light
        self.illum_shN = torch.nn.Parameter(torch.zeros(N, sh_degree**2-1, 3))  # Local light
        
        # Geometry (standard)
        self.means = torch.nn.Parameter(torch.zeros(N, 3))
        self.scales = torch.nn.Parameter(torch.zeros(N, 3))
        self.quats = torch.nn.Parameter(torch.zeros(N, 4))
        self.opacities = torch.nn.Parameter(torch.zeros(N))
    
    def forward(self, exposure: float):
        # Final color = albedo * illumination * exposure
        # In SH space: addition (since we're in log space conceptually)
        colors_sh0 = self.albedo_sh0 + self.illum_sh0 * exposure
        colors_shN = self.albedo_shN + self.illum_shN * exposure
        return colors_sh0, colors_shN
```

**Training:**
```python
# During training on low-light images
colors_sh0, colors_shN = self.splats(exposure=exposure_low)
rendered = rasterize(colors_sh0, colors_shN, ...)
loss = nll_loss(rendered, gt_dark)

# Add regularization: albedo should be exposure-invariant
loss_albedo_reg = self.splats.albedo_sh0.var() * 0.01  # Encourage smooth albedo
loss_illum_reg = (self.splats.illum_sh0.mean() - torch.log(exposure_low))**2 * 0.1

# At test time: render with high exposure
colors_sh0, colors_shN = self.splats(exposure=10.0)  # Bright!
```

**Why this works:** The model learns that albedo is constant, illumination scales with exposure. Dark images provide albedo information, and exposure control provides brightness.

---

### **2. Self-Supervised Multi-Exposure Synthesis**

Generate synthetic multi-exposure training data from single low-light images:

```python
def generate_synthetic_exposures(dark_image, num_exposures=5):
    """
    Create training triplets: (dark, mid, bright) from single dark image
    Key insight: Even if GT is dark, we can create synthetic bright versions
    and enforce consistency in learned radiance space
    """
    exposures = torch.linspace(0.5, 5.0, num_exposures)
    synthetic_images = []
    
    for exp in exposures:
        # Simple: Just brighten the image
        synthetic = dark_image * exp
        # Better: Add noise model (bright images have less relative noise)
        noise_level = 0.01 / exp  # Less noise in bright images
        synthetic = synthetic + torch.randn_like(synthetic) * noise_level
        synthetic_images.append(synthetic)
    
    return synthetic_images, exposures

# Training loop
dark_image = load_image(...)
synthetic_images, exposures = generate_synthetic_exposures(dark_image)

for syn_img, exp in zip(synthetic_images, exposures):
    # Render with this exposure
    rendered = rasterize_splats(..., exposure=exp)
    # Train to match synthetic
    loss += nll_loss(rendered, syn_img)

# Critical: Add cross-exposure consistency
# If I render the same point at 2 exposures, radiance should scale linearly
radiance_1 = rasterize_splats(..., exposure=1.0, apply_exposure=False)
radiance_2 = rasterize_splats(..., exposure=2.0, apply_exposure=False)
loss_consistency = F.l1_loss(radiance_1 * 2.0, radiance_2)
```

**Insight:** You're teaching the model "this is what the scene looks like at different exposures" even though you only have one dark capture.

---

### **3. Physics-Based Radiance Clues**

Even in dark images, there are physics clues about true radiance:

```python
def extract_radiance_priors(dark_image):
    """
    Extract high-confidence radiance estimates from dark images
    """
    # 1. Specular highlights (even in dark images, these are clipped)
    # These tell us about maximum radiance
    specular_mask = (dark_image > 0.9).any(dim=-1)  # Saturated pixels
    
    # 2. Shadow/non-shadow ratios (geometry is exposure-invariant)
    # If region A is 2x brighter than region B in dark image,
    # this ratio should hold in bright image
    
    # 3. Color constancy (ratios between RGB channels)
    rgb_ratios = dark_image / (dark_image.mean(dim=-1, keepdim=True) + 1e-6)
    
    return {
        'specular_mask': specular_mask,
        'rgb_ratios': rgb_ratios,
    }

# Use in loss
priors = extract_radiance_priors(gt_dark)

# Loss: RGB ratios should match even at different exposures
rendered_ratios = rendered / (rendered.mean(dim=-1, keepdim=True) + 1e-6)
loss_color_const = F.l1_loss(rendered_ratios, priors['rgb_ratios'])

# Loss: Specular regions should be bright in radiance space
radiance = rendered / exposure
loss_specular = (1.0 - radiance[priors['specular_mask']]).mean()
```

---

### **4. Ratio-Based Supervision (Upgraded)**

Your ratio loss is close, but make it smarter:

```python
def compute_smart_ratio_loss(splats, trainset):
    """
    Use relative brightness between images to supervise absolute radiance
    """
    # Sample 2 images with different brightness
    img1_data = trainset[random.randint(0, len(trainset)-1)]
    img2_data = trainset[random.randint(0, len(trainset)-1)]
    
    # Compute mean brightness ratio (this is ground truth)
    brightness_1 = img1_data['image'].mean()
    brightness_2 = img2_data['image'].mean()
    gt_ratio = brightness_1 / (brightness_2 + 1e-6)
    
    # Render both
    R1 = rasterize_splats(..., apply_exposure=False)  # Raw radiance
    R2 = rasterize_splats(..., apply_exposure=False)
    
    # Get learned exposures
    exp1 = exposure_module(img1_data['image_id'])
    exp2 = exposure_module(img2_data['image_id'])
    
    # The ratio of exposures should match brightness ratio
    learned_ratio = exp1 / (exp2 + 1e-6)
    
    loss_ratio = F.l1_loss(learned_ratio, gt_ratio)
    
    # Additionally: HDR radiance should be similar after exposure correction
    I1 = R1 / exp1  # HDR reconstruction
    I2 = R2 / exp2
    loss_hdr_consistency = F.l1_loss(I1, I2.detach())  # One-way
    
    return loss_ratio + 0.1 * loss_hdr_consistency
```

**Key insight:** The brightness ratios between images tell you about relative exposures!

---

### **5. Adversarial Exposure Training**

Train a discriminator to distinguish "real bright images" from "rendered bright images":

```python
class BrightnessDiscriminator(torch.nn.Module):
    """Discriminator that judges if an image looks naturally well-lit"""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 4, 2, 1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, 4, 2, 1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 1, 4, 1, 0),
        )
    
    def forward(self, x):
        return self.net(x.permute(0, 3, 1, 2)).mean()

# Training
discriminator = BrightnessDiscriminator()

# Render at high exposure
rendered_bright = rasterize(..., exposure=10.0)

# Get "fake" bright images from synthetic data or other dataset
real_bright = load_welllit_images(...)  # From different dataset!

# GAN loss
D_real = discriminator(real_bright)
D_fake = discriminator(rendered_bright)

loss_gan = -torch.log(D_fake + 1e-8)  # Fool the discriminator
loss_disc = -torch.log(D_real + 1e-8) - torch.log(1 - D_fake + 1e-8)

# Combined
loss_total = loss_recon + 0.1 * loss_gan
```

**Insight:** You're learning "what do well-lit images look like" from external data, even though your 3D data is dark.

---

### **6. Curriculum Learning: Dark → Bright Gradually**

Don't jump from dark to bright immediately. Gradually increase target exposure:

```python
def get_curriculum_exposure(step, max_steps):
    """Start by reconstructing dark, gradually target brighter"""
    progress = step / max_steps
    
    # Phase 1 (0-30%): Reconstruct dark images (exposure ~1.0)
    # Phase 2 (30-70%): Gradually increase (exposure 1.0 → 5.0)
    # Phase 3 (70-100%): Target bright images (exposure 5.0 → 10.0)
    
    if progress < 0.3:
        return 1.0
    elif progress < 0.7:
        return 1.0 + (progress - 0.3) / 0.4 * 4.0  # 1.0 → 5.0
    else:
        return 5.0 + (progress - 0.7) / 0.3 * 5.0  # 5.0 → 10.0

# Training
target_exposure = get_curriculum_exposure(step, max_steps)
rendered = rasterize(..., exposure=target_exposure)

# Generate synthetic target at this exposure
synthetic_target = gt_dark * target_exposure
loss = nll_loss(rendered, synthetic_target)
```

**Why:** The model learns gradual brightness changes, not a sudden jump.

---

### **7. Albedo Smoothness + Illumination Sparsity**

Encourage the right decomposition through priors:

```python
# Albedo should be spatially smooth (surfaces have coherent colors)
def spatial_smoothness_loss(albedo_map):
    """Encourage nearby pixels to have similar albedo"""
    dy = albedo_map[:, 1:, :, :] - albedo_map[:, :-1, :, :]
    dx = albedo_map[:, :, 1:, :] - albedo_map[:, :, :-1, :]
    return (dy.abs().mean() + dx.abs().mean())

# Illumination should be sparse (most light comes from few directions)
def sparsity_loss(illumination_sh):
    """Encourage illumination to be sparse in SH basis"""
    return torch.abs(illumination_sh).mean()

# In training
albedo_sh0, albedo_shN = self.splats.albedo_sh0, self.splats.albedo_shN
illum_sh0, illum_shN = self.splats.illum_sh0, self.splats.illum_shN

# Render albedo map (for visualization and regularization)
albedo_map = rasterize(albedo_sh0, albedo_shN, ...)

loss_smooth = spatial_smoothness_loss(albedo_map) * 0.01
loss_sparse = sparsity_loss(illum_shN) * 0.01
loss += loss_smooth + loss_sparse
```

---

### **8. Test-Time Refinement with Bright Images**

If you get ANY bright images (even 1-2), use them for test-time adaptation:

```python
# During test time, if you have a bright image
def test_time_refinement(splats, bright_image, num_steps=100):
    """Fine-tune on a single bright image to calibrate exposure"""
    
    # Clone current model
    splats_test = copy.deepcopy(splats)
    optimizer = torch.optim.Adam(splats_test.parameters(), lr=1e-4)
    
    for _ in range(num_steps):
        rendered = rasterize(splats_test, exposure=10.0)
        loss = F.l1_loss(rendered, bright_image)
        loss.backward()
        optimizer.step()
    
    return splats_test

# Use this refined model for final rendering
```

---

### **9. Pretrain on Synthetic Data**

Use synthetic bright+dark pairs to pretrain:

```python
# Generate synthetic low-light data from well-lit datasets
def create_synthetic_lowlight(bright_image):
    """Simulate low-light capture"""
    # Reduce exposure
    dark_image = bright_image / 10.0
    # Add realistic noise
    noise = torch.randn_like(dark_image) * 0.05
    dark_image = dark_image + noise
    return dark_image

# Pretrain on synthetic pairs
bright_gt = load_welllit_dataset()
dark_synthetic = create_synthetic_lowlight(bright_gt)

# Train with paired supervision
rendered_dark = rasterize(..., exposure=1.0)
rendered_bright = rasterize(..., exposure=10.0)

loss = nll_loss(rendered_dark, dark_synthetic) + nll_loss(rendered_bright, bright_gt)

# Then fine-tune on real LOM dark images
```

---

### **10. Exposure as a Latent Variable (VAE-Style)**

Treat exposure as a latent variable you want to infer:

```python
class ExposureVAE(torch.nn.Module):
    """Learn a distribution over exposures"""
    def __init__(self, n_images):
        super().__init__()
        self.exposure_mean = torch.nn.Parameter(torch.ones(n_images))
        self.exposure_logvar = torch.nn.Parameter(torch.zeros(n_images))
    
    def forward(self, image_ids, sample=True):
        mean = self.exposure_mean[image_ids]
        logvar = self.exposure_logvar[image_ids]
        
        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean
    
    def kl_divergence(self, image_ids):
        """KL divergence with prior p(exposure) = LogNormal(0, 1)"""
        mean = self.exposure_mean[image_ids]
        logvar = self.exposure_logvar[image_ids]
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

# Training
exposure = exposure_vae(image_ids, sample=True)
rendered = rasterize(..., exposure=exposure)
loss_recon = nll_loss(rendered, gt_dark)
loss_kl = exposure_vae.kl_divergence(image_ids) * 0.01
loss = loss_recon + loss_kl
```

**Why:** Uncertainty in exposure is explicitly modeled. At test time, you sample or use the mean.

---

### **11. Depth-Guided Lighting Separation**

Use depth to separate ambient vs direct lighting:

```python
# Objects further away receive more ambient light, less direct
def depth_aware_lighting(depth_map, ambient_coeff, direct_coeff):
    """
    Ambient: Uniform, exposure-invariant background light
    Direct: Distance-dependent, exposure-dependent spotlight
    """
    ambient = ambient_coeff * torch.ones_like(depth_map)
    direct = direct_coeff * torch.exp(-depth_map / depth_scale)
    
    return ambient + direct

# In rasterization, compute per-pixel depth
depth = render_depth(...)
lighting = depth_aware_lighting(depth, self.ambient, self.direct)

# Final color = albedo * lighting * exposure
colors = albedo * lighting * exposure
```

---

### **12. Cross-Dataset Transfer Learning**

Use well-lit datasets to learn "what brightness looks like":

```python
# Step 1: Train a "brightness prior" on well-lit datasets (e.g., Mip-NeRF360)
brightness_encoder = train_on_bright_dataset()  # Learns brightness statistics

# Step 2: Use this prior when training on LOM
def brightness_prior_loss(rendered_bright, brightness_encoder):
    """Ensure rendered bright image matches statistics of real bright images"""
    rendered_features = brightness_encoder(rendered_bright)
    # Match mean and variance of features
    loss = (rendered_features.mean() - bright_mean)**2
    loss += (rendered_features.std() - bright_std)**2
    return loss

# Training on LOM
rendered = rasterize(..., exposure=10.0)
loss_prior = brightness_prior_loss(rendered, brightness_encoder)
loss += 0.1 * loss_prior
```

---

## 🎯 Recommended Strategy: Combine Ideas #1, #2, #4, #6

Here's my recommended pipeline:

```python
# 1. Intrinsic Decomposition (Idea #1)
splats = IntrinsicGaussianSplats(...)

# 2. Multi-Exposure Synthesis (Idea #2)
synthetic_exposures = generate_synthetic_exposures(dark_image, num=5)

# 3. Curriculum Learning (Idea #6)
target_exposure = get_curriculum_exposure(step, max_steps)

# 4. Ratio Supervision (Idea #4)
loss_ratio = compute_smart_ratio_loss(splats, trainset)

# Combined Training Loop
for step in range(max_steps):
    # Get dark GT
    dark_gt = load_dark_image(...)
    
    # Generate synthetic bright target
    target_exp = get_curriculum_exposure(step, max_steps)
    bright_target = dark_gt * target_exp
    
    # Render with intrinsic decomposition
    colors_sh0, colors_shN = splats(exposure=target_exp)
    rendered = rasterize(colors_sh0, colors_shN, ...)
    
    # Multi-task loss
    loss_recon = nll_loss(rendered, bright_target)
    loss_ratio = compute_smart_ratio_loss(splats, trainset)
    loss_albedo_reg = splats.albedo_sh0.var() * 0.01
    loss_illum_sparse = splats.illum_shN.abs().mean() * 0.01
    
    loss = loss_recon + 0.1*loss_ratio + loss_albedo_reg + loss_illum_sparse
    loss.backward()
```

---

## Key Takeaway

**The fundamental trick:** You need to give the model **inductive biases** about what "brightness" means, since you don't have bright GT. These biases come from:
1. **Physics:** Albedo is exposure-invariant, illumination scales linearly
2. **Synthetic data:** Brighten dark images yourself as pseudo-GT
3. **Cross-image consistency:** Brightness ratios provide relative exposure info
4. **External priors:** Use statistics from well-lit datasets
