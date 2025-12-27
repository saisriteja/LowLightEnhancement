# ============================================================================
# KEY FIXES FOR YOUR TRAINING LOOP
# ============================================================================

# 1. ENABLE INTRINSIC DECOMPOSITION (you have this disabled!)
cfg.enable_intrinsic_decomp = True
cfg.albedo_reg_lambda = 0.01
cfg.illum_reg_lambda = 0.01

# 2. ENABLE SYNTHETIC EXPOSURE GENERATION
cfg.enable_synthetic_exposures = True
cfg.num_synthetic_exposures = 5

# 3. FIX THE TRAINING TARGET CREATION
# In your training loop, BEFORE rendering:

# Get curriculum target exposure
if cfg.curriculum_enabled:
    target_exposure = get_curriculum_exposure(step, max_steps, cfg.curriculum_max_exposure)
else:
    target_exposure = 1.0

# Generate synthetic bright target
if cfg.enable_synthetic_exposures and target_exposure is not None:
    # CRITICAL: Don't just multiply by exposure, that's still dark!
    # Instead, use the intrinsic decomposition properly
    
    # Option A: Train in HDR space (RECOMMENDED)
    # Convert dark GT to HDR by dividing by a base exposure estimate
    # Assume dark images were captured at ~1/10 of proper exposure
    dark_base_exposure = 0.1  # Dark images are 10x underexposed
    hdr_gt = pixels / dark_base_exposure  # Convert to HDR radiance
    
    # Now create synthetic target at target_exposure
    # The model renders at target_exposure, and we train against HDR GT
    pixels_target = hdr_gt  # This is the TRUE bright radiance we want
    
    # For rendering, we'll use target_exposure in intrinsic decomp
    # Final render = albedo + illumination * target_exposure
    # We want: (albedo + illum * target_exposure) ≈ hdr_gt
    
    # This encourages:
    # - albedo to capture exposure-invariant color
    # - illumination to scale with exposure

else:
    pixels_target = pixels  # Fallback to dark image


# 4. FIX THE RENDERING CALL
# Pass target_exposure explicitly for intrinsic decomposition
exposure_value_kwarg = {}
if cfg.enable_intrinsic_decomp and target_exposure is not None:
    exposure_value_kwarg["exposure_value"] = torch.tensor(
        [target_exposure], device=device, dtype=torch.float32
    )

renders, alphas, info = self.rasterize_splats(
    camtoworlds=camtoworlds,
    Ks=Ks,
    width=width,
    height=height,
    sh_degree=sh_degree_to_use,
    near_plane=cfg.near_plane,
    far_plane=cfg.far_plane,
    image_ids=image_ids,
    render_mode="RGB+ED" if cfg.depth_loss else "RGB",
    masks=masks,
    apply_exposure=False,  # DON'T apply learned exposure here!
    **exposure_value_kwarg,  # Use curriculum exposure instead
)


# 5. FIX THE LOSS COMPUTATION
# Compare rendered (at target_exposure) against HDR GT
if cfg.use_nll_loss:
    loss_nll = poisson_gaussian_nll_loss(colors, pixels_target)
    loss = loss_nll
else:
    l1loss = F.l1_loss(colors, pixels_target)
    ssimloss = 1.0 - fused_ssim(
        colors.permute(0, 3, 1, 2), 
        pixels_target.permute(0, 3, 1, 2), 
        padding="valid"
    )
    loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda


# 6. ENHANCED RATIO LOSS (Use brightness ratios from GT)
if cfg.enable_ratio_loss and step % cfg.ratio_loss_freq == 0:
    # Sample 2 images
    indices = np.random.choice(len(self.trainset), size=2, replace=False)
    ratio_data = [self.trainset[int(idx)] for idx in indices]
    
    # Get GT brightness ratio
    img_i_gt = ratio_data[0]["image"].to(device)
    img_j_gt = ratio_data[1]["image"].to(device)
    brightness_i = img_i_gt.mean()
    brightness_j = img_j_gt.mean()
    gt_brightness_ratio = brightness_i / (brightness_j + 1e-6)
    
    # Render both at the SAME target exposure (e.g., target_exposure)
    radiances = []
    for rd in ratio_data:
        c2w = rd["camtoworld"].unsqueeze(0).to(device)
        K = rd["K"].unsqueeze(0).to(device)
        h, w = rd["image"].shape[:2]
        
        # Render at fixed exposure
        rad, _, _ = self.rasterize_splats(
            camtoworlds=c2w,
            Ks=K,
            width=w,
            height=h,
            sh_degree=sh_degree_to_use,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            apply_exposure=False,
            exposure_value=torch.tensor([target_exposure], device=device),
        )
        radiances.append(rad[..., 0:3])
    
    # HDR radiances should have the SAME brightness ratio as GT
    R_i, R_j = radiances[0], radiances[1]
    rendered_brightness_i = R_i.mean()
    rendered_brightness_j = R_j.mean()
    rendered_ratio = rendered_brightness_i / (rendered_brightness_j + 1e-6)
    
    # Ratio loss: rendered ratio should match GT ratio
    loss_ratio = F.l1_loss(rendered_ratio, gt_brightness_ratio)
    loss += cfg.ratio_lambda * loss_ratio


# 7. ADDITIONAL LOSS: Exposure Diversity (force learned exposures to diverge)
if cfg.enable_exposure_opt:
    # Even though we're not using learned exposures for training,
    # we still optimize them to match GT brightness
    # This helps at test time when we don't have curriculum
    
    if world_size > 1:
        learned_exposures = self.exposure_module.module.exposures
    else:
        learned_exposures = self.exposure_module.exposures
    
    # Encourage diversity: exposures shouldn't all be 1.0
    exp_std = learned_exposures.std()
    loss_exp_diversity = -0.1 * exp_std  # Negative because we want HIGH std
    loss += loss_exp_diversity
    
    # Also encourage learned exposures to match image brightness
    # For current batch, learned exposure should be ~1/brightness
    batch_brightness = pixels.mean()
    learned_exp_batch = learned_exposures[image_ids].mean()
    target_exp = 1.0 / (batch_brightness + 1e-3)  # Bright images need low exp
    loss_exp_align = F.l1_loss(learned_exp_batch, target_exp)
    loss += 0.01 * loss_exp_align


# ============================================================================
# ALTERNATIVE APPROACH: Multi-Exposure Supervision (Simpler)
# ============================================================================

# If intrinsic decomposition is too complex, try this simpler approach:

if cfg.enable_synthetic_exposures:
    # Generate multiple exposures from the dark image
    synthetic_images, exposure_levels = generate_synthetic_exposures(
        pixels, 
        num_exposures=3,  # dark, mid, bright
        min_exposure=1.0,
        max_exposure=target_exposure
    )
    
    # Render at multiple exposures and supervise each
    total_loss = 0.0
    for syn_img, exp_level in zip(synthetic_images, exposure_levels):
        # Render at this exposure level
        renders, _, _ = self.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            apply_exposure=False,
            exposure_value=torch.tensor([exp_level], device=device),
        )
        
        # Loss for this exposure level
        if cfg.use_nll_loss:
            total_loss += poisson_gaussian_nll_loss(renders, syn_img)
        else:
            total_loss += F.l1_loss(renders, syn_img)
    
    loss = total_loss / len(synthetic_images)


# ============================================================================
# CRITICAL DEBUG LOGGING
# ============================================================================

# Add this to your tensorboard logging to understand what's happening:
if world_rank == 0 and step % cfg.tb_every == 0:
    # Log actual image brightness
    self.writer.add_scalar("debug/gt_brightness_mean", pixels.mean().item(), step)
    self.writer.add_scalar("debug/gt_brightness_max", pixels.max().item(), step)
    
    # Log rendered brightness
    self.writer.add_scalar("debug/rendered_brightness_mean", colors.mean().item(), step)
    self.writer.add_scalar("debug/rendered_brightness_max", colors.max().item(), step)
    
    # Log target brightness
    self.writer.add_scalar("debug/target_brightness_mean", pixels_target.mean().item(), step)
    
    # Log curriculum vs learned exposure
    if cfg.curriculum_enabled:
        self.writer.add_scalar("debug/curriculum_exposure", target_exposure, step)
    if cfg.enable_exposure_opt:
        learned_exp = (self.exposure_module.module if world_size > 1 
                      else self.exposure_module).exposures[image_ids].mean()
        self.writer.add_scalar("debug/learned_exposure", learned_exp.item(), step)
    
    # Log intrinsic components (if enabled)
    if cfg.enable_intrinsic_decomp:
        albedo_mean = self.splats["albedo_sh0"].abs().mean().item()
        illum_mean = self.splats["illum_sh0"].abs().mean().item()
        self.writer.add_scalar("debug/albedo_magnitude", albedo_mean, step)
        self.writer.add_scalar("debug/illumination_magnitude", illum_mean, step)


# ============================================================================
# UPDATED CONFIG RECOMMENDATIONS
# ============================================================================

# Use these settings for best results:
Config(
    # Core HDR settings
    enable_intrinsic_decomp=True,  # ENABLE THIS!
    albedo_reg_lambda=0.01,
    illum_reg_lambda=0.01,
    
    # Multi-exposure synthesis
    enable_synthetic_exposures=True,
    num_synthetic_exposures=3,
    
    # Curriculum learning
    curriculum_enabled=True,
    curriculum_max_exposure=10.0,
    
    # Exposure optimization (keep enabled for test time)
    enable_exposure_opt=True,
    exposure_lr=1e-3,
    
    # Loss configuration
    use_nll_loss=True,
    ratio_lambda=0.1,
    sh_reg_lambda=0.01,
    
    # Prevent GS collapse
    strategy=MCMCStrategy(verbose=True),  # Use MCMC, not DefaultStrategy
    init_opa=0.5,
    init_scale=0.1,
    opacity_reg=0.01,
    scale_reg=0.01,
    min_gs_count=100,  # Lower threshold
    min_opacity_threshold=0.005,  # Higher threshold
    enable_opacity_clamping=True,
)


# ============================================================================
# WHY THIS WORKS
# ============================================================================

"""
The key insight: You need to decouple THREE things that you're currently conflating:

1. **Curriculum target exposure** (what exposure we render at during training)
   - Increases from 1.0 → 10.0 over training
   - Used in intrinsic decomposition: colors = albedo + illum * target_exposure

2. **Learned per-image exposures** (what the model thinks each image's exposure was)
   - Optimized to match GT brightness ratios
   - Used at test time when we don't have curriculum

3. **GT HDR radiance** (what we supervise against)
   - Derived from dark images by estimating their underexposure
   - Represents the "true" scene radiance

Current problem:
- You render at target_exposure=10.0
- But supervise against dark GT (brightness ~0.05)
- Model learns: "to match dark GT at exposure=10, I need to output ~0.05/10 = 0.005 radiance"
- This means learned exposures stay ~1.0 and model stores dark radiance

Solution:
- Render at target_exposure=10.0
- Supervise against hdr_gt = dark_gt / 0.1 (assuming 10x underexposure)
- Model learns: "to match bright GT at exposure=10, I need to output bright radiance"
- Now illumination scales correctly with exposure
- At test time, render at any exposure and get corresponding brightness
"""


# ============================================================================
# QUICK TEST: Verify Your Setup
# ============================================================================

# Add this to your training loop to verify the setup is correct:
if step == 0 and world_rank == 0:
    print("=== TRAINING SETUP VERIFICATION ===")
    print(f"GT image brightness: {pixels.mean().item():.6f}")
    print(f"Target exposure: {target_exposure}")
    print(f"Target brightness: {pixels_target.mean().item():.6f}")
    print(f"Ratio: {pixels_target.mean().item() / pixels.mean().item():.2f}")
    print(f"Expected ratio: {target_exposure:.2f}")
    
    if cfg.enable_intrinsic_decomp:
        print("✓ Intrinsic decomposition ENABLED")
    else:
        print("✗ Intrinsic decomposition DISABLED - THIS IS THE PROBLEM!")
    
    if cfg.enable_synthetic_exposures:
        print("✓ Synthetic exposures ENABLED")
    else:
        print("✗ Synthetic exposures DISABLED")
    
    print("===================================")