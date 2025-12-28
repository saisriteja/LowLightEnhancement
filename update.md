1. Structure-Preserving Losses

Add gradient-based loss (Sobel/edge detection) to preserve image structure
Add perceptual loss (VGG features) to maintain high-level features
Add SSIM loss with higher weight to preserve local structure
Consider total variation loss on the perturbations themselves

2. Better Illumination Constraints

Limit illumination range: Instead of [0, 2], use tighter bounds like [0.5, 1.5]
Smooth illumination curve: Add stronger 2nd derivative penalty (increase lambda_illum_smooth from 0.1 to 1.0)
Physical constraint: Enforce that L(e) follows actual exposure response (exponential or gamma curve)
Reference anchor: Force L(0) = 1.0 as hard constraint, not just initialization

3. Perturbation MLP Improvements

Reduce perturbation magnitudes: Lower max_delta_c, max_delta_alpha, max_delta_sigma (try 0.1, 0.05, 0.05)
Add sparsity: L1 regularization on MLP weights to encourage minimal perturbations
Frequency regularization: Penalize high-frequency changes in perturbations across space
Scale-dependent perturbations: Allow smaller Gaussians to have smaller perturbations

4. Loss Weight Rebalancing
Your current issues:

loss_smooth, loss_reg are too small (e-9 range) → increase by 1000x
loss_reflect, loss_illum_smooth, loss_vis_smooth barely active → increase by 100x
Photometric loss dominates everything → add structure losses with similar magnitude

Suggested weights:
lambda_consist: 5.0 (up from 0.5)
lambda_smooth: 10.0 (up from 0.1)
lambda_reg: 0.01 (up from 1e-4)
lambda_reflect: 10.0 (up from 1.0)
lambda_illum_smooth: 1.0 (up from 0.1)
lambda_structure: 2.0 (NEW - gradient/edge loss)


6. Training Strategy Changes

Two-stage training:

Stage 1 (0-10K): Train geometry only (no exposure variation)
Stage 2 (10K-30K): Add exposure perturbations gradually


Freeze base Gaussians: After 15K steps, freeze means/scales/quats, only train MLPs
Curriculum should be slower: Your phases [300, 600, 1000] are too fast - try [3000, 10000, 20000]
Increase consist_loss_freq: From 3 to 1 (compute every step)

7. Intrinsic Decomposition Fixes

Reflectance should be slowly varying: Add spatial smoothness loss on reflectance
Visibility should be view-dependent only: Remove exposure dependence completely
Bidirectional constraint: R should reconstruct I at ALL exposures, not just pairs
Reflectance normalization: Keep reflectance in [0, 1] range with sigmoid

8. Debugging/Monitoring

Visualize intermediate outputs: Save R, L(e), V(view) separately
Track perturbation statistics: Mean/std of delta_c, delta_alpha, delta_sigma
Compare with/without MLPs: Render same view with MLPs disabled
Check gradient flows: Are MLP gradients much smaller than Gaussian gradients?

9. Architectural Changes

Add skip connections in MLPs to preserve identity
Use residual blocks instead of plain MLPs
Separate MLPs per component: One for color, one for opacity, one for scale
Add positional encoding to capture high-frequency details

10. Quick Fixes to Try First

Reduce illumination MLP output range to [0.8, 1.2]
Increase lambda_structure to 2.0 (add gradient loss)
Increase lambda_smooth to 10.0
Reduce max_delta_c to 0.1
Slow down curriculum: multiply phases by 10x