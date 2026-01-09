import numpy as np
from scipy import fft
from skimage import io, exposure, img_as_ubyte, img_as_float


def firstOrderDerivative(n, k=1):
    """Compute first-order derivative matrix."""
    return np.eye(n) * (-1) + np.eye(n, k=k)


def toeplitizMatrix(n, row):
    """Construct Toeplitz matrix for ADMM optimization."""
    vecDD = np.zeros(n)
    vecDD[0] = 4
    vecDD[1] = -1
    vecDD[row] = -1
    # vecDD[-1] = -1
    # vecDD[-row] = -1
    vecDD[-1] = 0
    vecDD[-row] = 0
    return vecDD


def vectorize(matrix):
    """Vectorize matrix in column-major order."""
    return matrix.T.ravel()


def reshape(vector, row, col):
    """Reshape vector to matrix in column-major order."""
    return vector.reshape((row, col), order='F')


class PBD:
    """
    Photon Budget Distribution (PBD) algorithm for low-light image enhancement.
    
    Based on LIME but with explicit photon budget estimation and confidence mapping.
    """
    
    def __init__(self, iterations=10, alpha=2, rho=2, epsilon=1e-6, beta=0.7, 
                 use_iteration=False, image_type='sRGB', strategy=2,
                 use_priors=True,
                 fidelity_weight=0.0, saturation_max=1.0,
                 color_weight=0.02, detail_weight=0.15, structure_weight=0.0,
                 target_brightness=0.5, sigma_read=0.01,
                 poisson_weight=0.0, smoothness_weight=0.15, 
                 exposure_weight=0.0, spatial_weight=0.5):
        """
        Initialize PBD algorithm with physics-based photon count estimation.
        
        λ represents absolute photon count proxy (unbounded), not normalized illumination.
        Physical model: L(x) ∝ λ(x) where λ(x) = E[N(x)] (expected photon count)
        
        Parameters:
        -----------
        iterations : int
            Number of ADMM iterations for photon budget estimation
        alpha : float
            Regularization parameter for structure-aware refinement
        rho : float
            ADMM penalty parameter multiplier
        epsilon : float
            Small value to prevent division by zero (only lower bound for λ)
        beta : float
            Weight for iterative refinement (anchors to observations)
        use_iteration : bool
            Whether to use iterative refinement (2-3 iterations)
        image_type : str
            Image type: 'sRGB' or 'RAW'
        strategy : int
            Weighting strategy (1 or 2)
        use_priors : bool
            Whether to apply photon-budget-aligned priors (default: True)
        fidelity_weight : float
            Weight for confidence-weighted fidelity loss (default: 0.0, disabled)
        saturation_max : float
            Maximum radiance value before saturation penalty (default: 1.0)
        color_weight : float
            Weight for illumination-aware color consistency (default: 0.02)
        detail_weight : float
            Weight for detail enhancement in dark regions (default: 0.15, reduced to prevent saturation)
        structure_weight : float
            Weight for gradient structure consistency loss L₃ (default: 0.0, disabled)
        target_brightness : float
            Target mean brightness for final output (default: 0.5, optional enhancement)
        sigma_read : float
            Read noise standard deviation for shot noise modeling (default: 0.01)
        poisson_weight : float
            Weight for Poisson likelihood loss L₁ (default: 0.0, removed from ADMM)
        smoothness_weight : float
            Weight for illumination smoothness loss L₄ (default: 0.15)
        exposure_weight : float
            Weight for exposure control loss L₅ (default: 0.0, optional post-processing)
        spatial_weight : float
            Weight for spatial consistency loss L₆ (default: 0.5)
        """
        self.iterations = iterations
        self.alpha = alpha
        self.rho = rho
        self.epsilon = epsilon
        self.beta = beta
        self.use_iteration = use_iteration
        self.image_type = image_type
        self.strategy = strategy
        self.use_priors = use_priors
        self.fidelity_weight = fidelity_weight
        self.saturation_max = saturation_max
        self.color_weight = color_weight
        self.detail_weight = detail_weight
        self.structure_weight = structure_weight
        self.target_brightness = target_brightness
        self.sigma_read = sigma_read
        self.poisson_weight = poisson_weight
        self.smoothness_weight = smoothness_weight
        self.exposure_weight = exposure_weight
        self.spatial_weight = spatial_weight
        
        # Will be set in load()
        self.img_path = None
        self.L = None
        self.row = None
        self.col = None
        self.lambda_0 = None  # Initial photon evidence (replaces T_hat)
        self.lambda_map = None  # Refined photon budget (replaces T)
        self.confidence_map = None  # Confidence map C
        self.R = None  # Enhanced radiance
    
    def linearize(self, img):
        """
        Linearize image based on image type.
        
        Parameters:
        -----------
        img : ndarray
            Input image (float, [0, 1])
            
        Returns:
        --------
        ndarray
            Linearized image normalized to [0, 1]
        """
        if self.image_type.upper() == 'SRGB':
            # Apply inverse gamma correction (power 2.2)
            linearized = np.power(np.clip(img, 0, 1), 2.2)
        elif self.image_type.upper() == 'RAW':
            # Use as-is for RAW
            linearized = img.copy()
        else:
            # Default to sRGB
            linearized = np.power(np.clip(img, 0, 1), 2.2)
        
        # Normalize to [0, 1]
        linearized = np.clip(linearized, 0, 1)
        return linearized
    
    def gamma_correct(self, img):
        """
        Apply gamma correction to convert linear RGB back to sRGB.
        
        Parameters:
        -----------
        img : ndarray
            Linear RGB image (float, [0, 1])
            
        Returns:
        --------
        ndarray
            sRGB image normalized to [0, 1]
        """
        if self.image_type.upper() == 'SRGB':
            # Apply gamma correction (power 1/2.2)
            corrected = np.power(np.clip(img, 0, 1), 1/2.2)
        else:
            # RAW images don't need gamma correction
            corrected = np.clip(img, 0, 1)
        return corrected
    
    def load(self, imgPath):
        """
        Load and preprocess image.
        
        Parameters:
        -----------
        imgPath : str
            Path to input image
        """
        self.img_path = imgPath
        # Load image as float [0, 1]
        img = img_as_float(io.imread(imgPath))
        
        # Apply linearization
        self.L = self.linearize(img)
        
        self.row = self.L.shape[0]
        self.col = self.L.shape[1]
        
        # Initial photon evidence: λ₀ = max_channel(L)
        # Same as LIME's T_hat initialization, but interpreted as photon evidence
        self.lambda_0 = np.max(self.L, axis=2)
        
        # For very dark images after linearization, scale λ₀ to prevent numerical issues
        # This maintains relative photon distribution while ensuring stable division
        lambda_0_mean = np.mean(self.lambda_0)
        if lambda_0_mean < 0.05:  # Very dark after linearization
            scale_factor = 0.05 / (lambda_0_mean + self.epsilon)
            scale_factor = np.clip(scale_factor, 1.0, 50.0)  # Reasonable scaling
            self.lambda_0 = self.lambda_0 * scale_factor
            print(f"[PBD] Dark image detected. Scaled λ₀ by {scale_factor:.2f}x (mean: {lambda_0_mean:.4f} → {np.mean(self.lambda_0):.4f})")
        
        # Precompute derivative matrices
        self.dv = firstOrderDerivative(self.row)
        self.dh = firstOrderDerivative(self.col, -1)
        self.vecDD = toeplitizMatrix(self.row * self.col, self.row)
        
        # Compute weighting strategy
        self.W = self.weightingStrategy()
    
    def weightingStrategy(self):
        """
        Compute weight matrix for structure-aware refinement.
        
        Returns:
        --------
        ndarray
            Weight matrix W
        """
        if self.strategy == 2:
            dTv = self.dv @ self.lambda_0
            dTh = self.lambda_0 @ self.dh
            Wv = 1 / (np.abs(dTv) + 1)
            Wh = 1 / (np.abs(dTh) + 1)
            return np.vstack([Wv, Wh])
        else:
            return np.ones((self.row * 2, self.col))
    
    def compute_shot_noise_variance(self, lambda_val):
        """
        Compute shot noise variance from photon budget.
        
        Pathway 5: Shot Noise Modeling
        σ_shot²(x) = λ(x)  # Standard Poisson variance
        
        For Poisson process: Var(N) = E[N] = λ
        If λ is photon count, variance equals the count itself.
        
        Parameters:
        -----------
        lambda_val : ndarray
            Photon budget map (H, W) - absolute photon count proxy
            
        Returns:
        --------
        ndarray
            Shot noise variance map (H, W)
        """
        # Standard Poisson property: variance = mean = λ
        return lambda_val
    
    def compute_total_noise_variance(self, lambda_val):
        """
        Compute total noise variance (shot noise + read noise).
        
        Pathway 5: Shot Noise Modeling
        σ_total²(x) = σ_shot²(x) + σ_read²
        
        Parameters:
        -----------
        lambda_val : ndarray
            Photon budget map (H, W) - absolute photon count proxy
            
        Returns:
        --------
        ndarray
            Total noise variance map (H, W)
        """
        sigma_shot_sq = self.compute_shot_noise_variance(lambda_val)
        sigma_read_sq = self.sigma_read ** 2
        return sigma_shot_sq + sigma_read_sq
    
    def compute_noise_weights(self, lambda_val):
        """
        Compute noise-aware weights (kept for compatibility, not used in ADMM).
        
        Weight inversely proportional to total noise variance.
        w(x) = 1 / (σ_total²(x) + ε)
        
        Note: This method is kept for evaluation/monitoring but not used
        in ADMM optimization to avoid circular dependencies.
        
        Parameters:
        -----------
        lambda_val : ndarray
            Photon budget map (H, W) - absolute photon count proxy
            
        Returns:
        --------
        ndarray
            Noise weight map (H, W)
        """
        sigma_total_sq = self.compute_total_noise_variance(lambda_val)
        return 1.0 / (sigma_total_sq + self.epsilon)
    
    def poisson_likelihood_loss(self, I, lambda_val, R):
        """
        L₁: Poisson Negative Log-Likelihood Loss
        
        L_poisson = Σ [λ(x)R(x) - I(x)log(λ(x)R(x))]
        
        This replaces L2 fidelity with physically grounded Poisson loss.
        
        Parameters:
        -----------
        I : ndarray
            Observed linearized image (H, W, 3)
        lambda_val : ndarray
            Photon budget map (H, W)
        R : ndarray
            Enhanced radiance (H, W, 3)
            
        Returns:
        --------
        float
            Poisson likelihood loss value
        """
        # Use max channel for lambda-R product
        if R.ndim == 3:
            R_max = np.max(R, axis=2)
            I_max = np.max(I, axis=2)
        else:
            R_max = R
            I_max = I
        
        # Compute λR product
        lambda_R = lambda_val * R_max
        
        # Avoid log(0) by clipping
        lambda_R_clipped = np.maximum(lambda_R, self.epsilon)
        
        # Poisson loss: λR - I*log(λR)
        loss = lambda_R - I_max * np.log(lambda_R_clipped)
        
        return np.sum(loss)
    
    def __lambda_subproblem(self, G, Z, u):
        """
        Solve λ subproblem in ADMM optimization with structure-aware smoothing.
        
        Minimizes: ||λ - λ₀||² + α||W ⊙ ∇λ||₁
        
        Physics-correct: λ is absolute photon count proxy (unbounded).
        Simple L2 fidelity to observations λ₀, with structure-aware smoothness.
        Removed circular Poisson approximation and range constraints.
        
        Parameters:
        -----------
        G : ndarray
            Auxiliary variable for gradient
        Z : ndarray
            Dual variable
        u : float
            ADMM penalty parameter
            
        Returns:
        --------
        ndarray
            Updated photon budget λ (unbounded, only lower-bounded by epsilon)
        """
        # Use lambda_0 directly as target (no exposure adjustment)
        # λ₀ = max_channel(L) is already photon evidence
        lambda_target = self.lambda_0
        
        # Prevent ADMM from collapsing lambda to zero
        # Ensure lambda_target has reasonable values
        lambda_target = np.maximum(lambda_target, 0.01)
        
        X = G - Z / u
        Xv = X[:self.row, :]
        Xh = X[self.row:, :]
        temp = self.dv @ Xv + Xh @ self.dh
        
        # Simple fidelity: λ should match observed photon evidence λ₀
        # For single-image estimation, L2 fidelity is correct (theory: λ ∝ L)
        fidelity_term = 2 * lambda_target
        
        # ADMM update
        numerator = fft.fft(vectorize(fidelity_term + u * temp))
        denominator = fft.fft(self.vecDD * u) + 2
        denominator = denominator + self.epsilon
        lambda_val = fft.ifft(numerator / denominator)
        lambda_val = np.real(reshape(lambda_val, self.row, self.col))
        
        # Only prevent division by zero, but enforce reasonable minimum
        # Photon counts shouldn't collapse to nearly zero
        lambda_val = np.maximum(lambda_val, 0.01)
        
        # Ensure no NaN or Inf values (use 0.01 as fallback)
        lambda_val = np.nan_to_num(lambda_val, nan=0.01, posinf=np.max(lambda_val), neginf=0.01)
        return lambda_val
    
    def __G_subproblem(self, lambda_val, Z, u, W):
        """
        Solve G subproblem in ADMM optimization with illumination smoothness.
        
        L₄: Illumination Smoothness
        L_smooth = Σ |∇λ|²
        
        Parameters:
        -----------
        lambda_val : ndarray
            Current photon budget estimate
        Z : ndarray
            Dual variable
        u : float
            ADMM penalty parameter
        W : ndarray
            Weight matrix
            
        Returns:
        --------
        ndarray
            Updated auxiliary variable G
        """
        dLambda = self.__derivative(lambda_val)
        
        # L₄: Add smoothness regularization
        # The smoothness loss encourages smaller gradients
        smoothness_epsilon = self.alpha * W / u
        if self.use_priors and self.smoothness_weight > 0:
            # Increase regularization strength for smoothness
            smoothness_epsilon = smoothness_epsilon + self.smoothness_weight / u
        
        X = dLambda + Z / u
        return np.sign(X) * np.maximum(np.abs(X) - smoothness_epsilon, 0)
    
    def __Z_subproblem(self, lambda_val, G, Z, u):
        """
        Update dual variable Z in ADMM optimization.
        
        Parameters:
        -----------
        lambda_val : ndarray
            Current photon budget estimate
        G : ndarray
            Auxiliary variable
        Z : ndarray
            Current dual variable
        u : float
            ADMM penalty parameter
            
        Returns:
        --------
        ndarray
            Updated dual variable Z
        """
        dLambda = self.__derivative(lambda_val)
        return Z + u * (dLambda - G)
    
    def __u_subproblem(self, u):
        """
        Update ADMM penalty parameter.
        
        Parameters:
        -----------
        u : float
            Current penalty parameter
            
        Returns:
        --------
        float
            Updated penalty parameter
        """
        return u * self.rho
    
    def __derivative(self, matrix):
        """
        Compute gradient of matrix.
        
        Parameters:
        -----------
        matrix : ndarray
            Input matrix
            
        Returns:
        --------
        ndarray
            Stacked vertical and horizontal gradients
        """
        v = self.dv @ matrix
        h = matrix @ self.dh
        return np.vstack([v, h])
    
    def confidence(self, lambda_val):
        """
        Compute confidence map from photon budget using Poisson probability.
        
        C(x) = 1 - exp(-λ(x))
        
        Physical interpretation: Probability of at least one photon arrival.
        High λ → C ≈ 1 → trust the observation (good photon count)
        Low λ → C ≈ 0 → insufficient photons (poor photon count)
        
        Parameters:
        -----------
        lambda_val : ndarray
            Photon budget estimate (absolute photon count proxy, unbounded)
            
        Returns:
        --------
        ndarray
            Confidence map C in [0, 1]
        """
        # Use corrected Poisson-based confidence formula
        # C(x) = 1 - exp(-λ) represents probability of at least one photon
        return 1.0 - np.exp(-np.maximum(lambda_val, self.epsilon))
    
    def confidence_weighted_fidelity(self, R, I, lambda_val):
        """
        Loss 3: Confidence-Weighted Fidelity (DISABLED by default)
        
        This loss is disabled because it pulls enhanced values back toward dark observations.
        In low-light enhancement, we want to enhance dark regions, not preserve them.
        
        Parameters:
        -----------
        R : ndarray
            Enhanced radiance (H, W, 3)
        I : ndarray
            Original linearized image (H, W, 3)
        lambda_val : ndarray
            Photon budget map (H, W)
            
        Returns:
        --------
        ndarray
            Unchanged radiance (loss disabled)
        """
        # Disabled: This loss suppresses enhancement
        return R
    
    def prevent_saturation(self, R):
        """
        Loss 4: Non-Saturation via Radiance Upper Bound
        
        L_saturation = Σ max(0, R(x) - R_max)²
        
        Prevents extreme over-enhancement without suppressing normal enhancement.
        
        Parameters:
        -----------
        R : ndarray
            Enhanced radiance (H, W, 3)
            
        Returns:
        --------
        ndarray
            Saturation-prevented radiance
        """
        if not self.use_priors:
            return R
        
        # Soft clipping: penalize values above saturation_max
        excess = np.maximum(0, R - self.saturation_max)
        # Apply soft penalty: reduce excess values
        R_corrected = R - excess * 0.5  # Soft reduction
        
        return np.clip(R_corrected, 0, self.saturation_max)
    
    def illumination_aware_color_consistency(self, R, lambda_val):
        """
        Loss 5: Illumination-Aware Color Consistency
        
        L_color = Σ (1 - λ(x)) · ||E[R_r] - E[R_g]|| + ||E[R_g] - E[R_b]||
        
        Only balance colors in LOW λ regions (dark areas where noise dominates).
        Leave colors alone in HIGH λ regions (bright areas with good signal).
        
        Parameters:
        -----------
        R : ndarray
            Enhanced radiance (H, W, 3)
        lambda_val : ndarray
            Photon budget map (H, W)
            
        Returns:
        --------
        ndarray
            Color-balanced radiance
        """
        if not self.use_priors or self.color_weight <= 0:
            return R
        
        # Identify low-λ regions (dark areas)
        low_lambda_mask = lambda_val < 0.5
        
        if np.sum(low_lambda_mask) == 0:
            return R
        
        # Compute mean of each channel in low-λ regions only
        R_low = R[low_lambda_mask]
        mean_r = np.mean(R_low[:, 0])
        mean_g = np.mean(R_low[:, 1])
        mean_b = np.mean(R_low[:, 2])
        
        # Target: average of all channels in low-λ regions
        target_mean = (mean_r + mean_g + mean_b) / 3.0
        
        # Compute scale factors
        scale_r = target_mean / (mean_r + self.epsilon) if mean_r > self.epsilon else 1.0
        scale_g = target_mean / (mean_g + self.epsilon) if mean_g > self.epsilon else 1.0
        scale_b = target_mean / (mean_b + self.epsilon) if mean_b > self.epsilon else 1.0
        
        # Apply color correction only in low-λ regions with weight
        R_corrected = R.copy()
        weight_map = (1.0 - lambda_val)[..., np.newaxis]  # Higher weight for lower λ
        
        R_corrected[:, :, 0] = R[:, :, 0] * (1.0 + (scale_r - 1.0) * weight_map[:, :, 0] * self.color_weight)
        R_corrected[:, :, 1] = R[:, :, 1] * (1.0 + (scale_g - 1.0) * weight_map[:, :, 0] * self.color_weight)
        R_corrected[:, :, 2] = R[:, :, 2] * (1.0 + (scale_b - 1.0) * weight_map[:, :, 0] * self.color_weight)
        
        return np.clip(R_corrected, 0, 1)
    
    def enhance_detail_in_dark_regions(self, R, lambda_val):
        """
        Loss 6: Uniform Dark Region Enhancement
        
        Applies confidence-gated brightness boost to dark regions.
        Uses confidence map C(x) = 1 - exp(-λ) to gate enhancement:
        - Low confidence (C < 0.5): apply boost (uncertain regions need enhancement)
        - High confidence (C > 0.5): minimal boost (trust observation)
        
        Parameters:
        -----------
        R : ndarray
            Enhanced radiance (H, W, 3)
        lambda_val : ndarray
            Photon budget map (H, W) - absolute photon count proxy
            
        Returns:
        --------
        ndarray
            Confidence-gated enhanced radiance
        """
        if not self.use_priors or self.detail_weight <= 0:
            return R
        
        # Compute confidence map: C(x) = 1 - exp(-λ)
        # Low C = low photon count = uncertain = needs enhancement
        C = self.confidence(lambda_val)
        
        # Normalize lambda to [0, 1] range for stable boost calculation
        # Use percentile to handle unbounded λ values
        lambda_percentile_95 = np.percentile(lambda_val, 95)
        lambda_normalized = lambda_val / (lambda_percentile_95 + self.epsilon)
        lambda_normalized = np.clip(lambda_normalized, 0, 1)
        
        # Confidence-gated boost: apply more where confidence is LOW
        # Boost factor: (1 - C) * detail_weight, capped to prevent saturation
        confidence_based_boost = 1.0 + (1.0 - C) * self.detail_weight * 0.5  # Reduced multiplier
        
        # Clip boost to reasonable range (max 1.3x for detail_weight=0.15)
        max_boost = 1.0 + self.detail_weight * 0.5
        brightness_boost = np.clip(confidence_based_boost, 1.0, max_boost)
        
        # Apply boost to all channels
        brightness_boost_3d = np.repeat(brightness_boost[:, :, np.newaxis], 3, axis=2)
        R_enhanced = R * brightness_boost_3d
        
        # Smooth transition: blend more in low-confidence regions
        blend_weight = (1.0 - C)[..., np.newaxis]
        R_final = (1.0 - blend_weight) * R + blend_weight * R_enhanced
        
        return np.clip(R_final, 0, self.saturation_max)
    


    def illumination_smoothness_loss(self, lambda_val):
        """
        L₄: Illumination Smoothness Loss
        
        L_smooth = Σ |∇λ|²
        
        Prevents flickering artifacts and ensures smooth transitions.
        This is NOT redundant with L₃:
        - L₃: λ should match input structure
        - L₄: λ should be locally smooth
        
        Parameters:
        -----------
        lambda_val : ndarray
            Photon budget map (H, W)
            
        Returns:
        --------
        float
            Smoothness loss value
        """
        grad_x = np.gradient(lambda_val, axis=1)
        grad_y = np.gradient(lambda_val, axis=0)
        smoothness = np.sum(grad_x**2 + grad_y**2)
        return smoothness
    
    def gradient_structure_consistency(self, lambda_val, R):
        """
        L₃: Gradient Structure Consistency (DISABLED by default)
        
        This loss is disabled because it's redundant with ADMM structure-aware smoothing.
        The weight matrix W in ADMM already preserves edges while smoothing.
        Forcing ∇λ = ∇I collapses PBD to LIME (λ = I up to constant).
        
        Parameters:
        -----------
        lambda_val : ndarray
            Current photon budget estimate (H, W)
        R : ndarray
            Enhanced radiance (H, W, 3) - not used, kept for compatibility
            
        Returns:
        --------
        ndarray
            Unchanged photon budget λ (no-op when structure_weight=0)
        """
        # Disabled: Structure preservation handled by W in ADMM
        # This loss is redundant and contradictory
        return lambda_val

        
    def apply_brightness_enhancement(self, R):
        """
        Optional brightness enhancement (post-processing, not physics).
        
        This is an optional enhancement applied to radiance R, NOT part of
        the physics model. Photon budget λ is a measurement and cannot be
        modified to meet brightness targets.
        
        Parameters:
        -----------
        R : ndarray
            Enhanced radiance (H, W, 3)
            
        Returns:
        --------
        ndarray
            Brightness-enhanced radiance (if enabled)
        """
        if not self.use_priors or self.exposure_weight <= 0 or self.target_brightness <= 0:
            return R
        
        # Post-processing enhancement: apply brightness adjustment to R only
        # This does NOT modify photon budget λ (which is a measurement)
        R_gray = np.mean(R, axis=2)
        
        # Region-based brightness enhancement
        region_size = max(8, min(self.row, self.col) // 16)  # Adaptive region size
        R_enhanced = R.copy()
        
        for i in range(0, self.row, region_size):
            for j in range(0, self.col, region_size):
                # Extract region
                i_end = min(i + region_size, self.row)
                j_end = min(j + region_size, self.col)
                region = R_gray[i:i_end, j:j_end]
                
                # Compute region mean
                region_mean = np.mean(region)
                
                # Apply brightness enhancement if below target
                if region_mean < self.target_brightness:
                    scale_factor = self.target_brightness / (region_mean + self.epsilon)
                    # Clip scale factor to avoid artifacts
                    scale_factor = np.clip(scale_factor, 1.0, 2.0)
                    
                    # Apply to all channels in region
                    R_enhanced[i:i_end, j:j_end, :] = R[i:i_end, j:j_end, :] * scale_factor
        
        return np.clip(R_enhanced, 0, self.saturation_max)
    
    def spatial_consistency_loss(self, R, I):
        """
        L₆: Spatial Consistency Loss
        
        L_spatial = Σ |(Y_i - Y_j) - (I_i - I_j)|²
        
        Preserves local contrast by maintaining relative differences.
        Physical justification: Relative photon differences should be preserved.
        
        Fixed: Compute differences per-channel consistently (not mixing grayscale I with per-channel R).
        
        Parameters:
        -----------
        R : ndarray
            Enhanced radiance Y (H, W, 3)
        I : ndarray
            Original linearized image (H, W, 3)
            
        Returns:
        --------
        ndarray
            Spatially consistent radiance
        """
        if not self.use_priors or self.spatial_weight <= 0:
            return R
        
        R_corrected = R.copy()
        
        # Apply spatial consistency for 4-connected neighbors
        # Compute differences per-channel consistently (not mixing grayscale I with per-channel R)
        for c in range(3):  # Apply to each channel
            R_channel = R[:, :, c].copy()
            I_channel = I[:, :, c].copy()
            correction = np.zeros_like(R_channel)
            
            # Compute differences with 4-connected neighbors
            # Up: R[i,j] - R[i-1,j] should match I[i,j] - I[i-1,j]
            diff_R_up = np.zeros_like(R_channel)
            diff_I_up = np.zeros_like(I_channel)
            diff_R_up[1:, :] = R_channel[1:, :] - R_channel[:-1, :]
            diff_I_up[1:, :] = I_channel[1:, :] - I_channel[:-1, :]
            error_up = diff_R_up - diff_I_up
            correction[1:, :] += error_up[1:, :]
            
            # Down: R[i,j] - R[i+1,j] should match I[i,j] - I[i+1,j]
            diff_R_down = np.zeros_like(R_channel)
            diff_I_down = np.zeros_like(I_channel)
            diff_R_down[:-1, :] = R_channel[:-1, :] - R_channel[1:, :]
            diff_I_down[:-1, :] = I_channel[:-1, :] - I_channel[1:, :]
            error_down = diff_R_down - diff_I_down
            correction[:-1, :] += error_down[:-1, :]
            
            # Left: R[i,j] - R[i,j-1] should match I[i,j] - I[i,j-1]
            diff_R_left = np.zeros_like(R_channel)
            diff_I_left = np.zeros_like(I_channel)
            diff_R_left[:, 1:] = R_channel[:, 1:] - R_channel[:, :-1]
            diff_I_left[:, 1:] = I_channel[:, 1:] - I_channel[:, :-1]
            error_left = diff_R_left - diff_I_left
            correction[:, 1:] += error_left[:, 1:]
            
            # Right: R[i,j] - R[i,j+1] should match I[i,j] - I[i,j+1]
            diff_R_right = np.zeros_like(R_channel)
            diff_I_right = np.zeros_like(I_channel)
            diff_R_right[:, :-1] = R_channel[:, :-1] - R_channel[:, 1:]
            diff_I_right[:, :-1] = I_channel[:, :-1] - I_channel[:, 1:]
            error_right = diff_R_right - diff_I_right
            correction[:, :-1] += error_right[:, :-1]
            
            # Average correction from all neighbors
            correction = correction / 4.0
            
            # Adjust R to reduce spatial consistency error
            R_corrected[:, :, c] = R_channel - self.spatial_weight * correction
        
        return np.clip(R_corrected, 0, self.saturation_max)
    
    def photonBudgetMap(self):
        """
        Estimate photon budget using ADMM optimization.
        
        Minimizes: ||λ - λ₀||² + α||W ⊙ ∇λ||₁
        
        Returns:
        --------
        ndarray
            Refined photon budget λ
        """
        lambda_val = np.zeros((self.row, self.col))
        G = np.zeros((self.row * 2, self.col))
        Z = np.zeros((self.row * 2, self.col))
        u = 1
        
        for _ in range(self.iterations):
            lambda_val = self.__lambda_subproblem(G, Z, u)
            G = self.__G_subproblem(lambda_val, Z, u, self.W)
            Z = self.__Z_subproblem(lambda_val, G, Z, u)
            u = self.__u_subproblem(u)
        
        self.lambda_map = lambda_val
        return lambda_val
    
    def enhance_iterative(self, num_iterations=3):
        """
        Perform iterative refinement with photon-budget-aligned losses.
        
        for k in iterations:
            λ(k) = refine(λ(k-1)) with range constraint
            R(k) = L / (λ(k) + ε)
            Apply Loss 4: Prevent saturation
            Apply Loss 3: Confidence-weighted fidelity
            Apply Loss 5: Illumination-aware color consistency
            Apply Loss 6: Detail enhancement in dark regions
            Apply Loss 7: λ-R consistency check
            λ(k+1) = consistency_adjusted(β·L + (1-β)·R(k))
        
        Parameters:
        -----------
        num_iterations : int
            Number of refinement iterations (default: 3)
            
        Returns:
        --------
        ndarray
            Final enhanced radiance R
        """
        # Initial photon budget estimation (L₁, L₂, L₄ applied in ADMM)
        lambda_val = self.photonBudgetMap()
        
        for k in range(num_iterations):
            # Estimate radiance: divide by illumination map
            lambda_3d = np.repeat(lambda_val[:, :, np.newaxis], 3, axis=2)
            R = self.L / (lambda_3d + self.epsilon)
            
            # Clip to reasonable range instead of rescaling
            # rescale_intensity normalizes to [0,1] which collapses lambda in iterations
            R = np.clip(R, 0, self.saturation_max)
            
            # Apply photon-budget-aligned losses
            if self.use_priors:
                # Prevent saturation
                R = self.prevent_saturation(R)
                
                # Confidence-weighted fidelity (disabled)
                R = self.confidence_weighted_fidelity(R, self.L, lambda_val)
                
                # Illumination-aware color consistency
                R = self.illumination_aware_color_consistency(R, lambda_val)
                
                # Detail enhancement in dark regions
                R = self.enhance_detail_in_dark_regions(R, lambda_val)
                
                # L₆: Spatial consistency
                R = self.spatial_consistency_loss(R, self.L)
            
            # Re-estimate photon budget from current radiance
            R_max = np.max(R, axis=2)
            # Don't let lambda collapse - use original lambda_0 more heavily 
            # Beta=0.7 means trust initial photon evidence, not estimated R
            # lambda_new = self.beta * self.lambda_0 + (1 - self.beta) * np.clip(R_max, 0.01, 1.0)

            L_max = np.max(self.L, axis=2)
            lambda_new = (
                self.beta * self.lambda_0
                + (1 - self.beta) * (L_max / (R_max + self.epsilon))
            )
                    
            # Prevent lambda collapse: ensure minimum meaningful value
            lambda_new = np.maximum(lambda_new, 0.01)
            
            # L₃: Gradient structure consistency
            if self.use_priors:
                lambda_new = self.gradient_structure_consistency(lambda_new, R)
            
            # Refine photon budget with ADMM (L₁, L₂, L₄ applied)
            self.lambda_0 = lambda_new
            self.W = self.weightingStrategy()
            lambda_val = self.photonBudgetMap()
        
        # Final confidence and radiance computation
        self.lambda_map = lambda_val
        self.confidence_map = self.confidence(lambda_val)
        lambda_3d = np.repeat(lambda_val[:, :, np.newaxis], 3, axis=2)
        R = self.L / (lambda_3d + self.epsilon)
        
        # Clip to reasonable range instead of rescaling
        R = np.clip(R, 0, self.saturation_max)
        
        # Apply final losses
        if self.use_priors:
            R = self.prevent_saturation(R)
            R = self.confidence_weighted_fidelity(R, self.L, lambda_val)
            R = self.illumination_aware_color_consistency(R, lambda_val)
            R = self.enhance_detail_in_dark_regions(R, lambda_val)
            # L₆: Spatial consistency
            R = self.spatial_consistency_loss(R, self.L)
            # Optional brightness enhancement (post-processing, not physics)
            R = self.apply_brightness_enhancement(R)
        
        # Convert back to sRGB for display
        R = self.gamma_correct(R)
        R = np.clip(R, 0, 1)
        
        self.R = img_as_ubyte(R)
        return self.R
    
    def enhance(self):
        """
        Enhance image using PBD algorithm with photon-budget-aligned losses.
        
        R(x) = L(x) / (λ(x) + ε)
        
        Applies photon-budget-aligned loss functions that enhance rather than suppress.
        
        Returns:
        --------
        ndarray
            Enhanced radiance R (uint8)
        """
        if self.use_iteration:
            return self.enhance_iterative()
        
        # Single-pass enhancement
        # Estimate photon budget (L₁, L₂, L₄ applied in ADMM)
        lambda_val = self.photonBudgetMap()
        
        # L₃: Gradient structure consistency
        if self.use_priors:
            lambda_val = self.gradient_structure_consistency(lambda_val, None)
            self.lambda_map = lambda_val
        
        # Compute confidence map (Poisson SNR-based)
        self.confidence_map = self.confidence(lambda_val)
        
        # Estimate radiance: divide by illumination map
        lambda_3d = np.repeat(lambda_val[:, :, np.newaxis], 3, axis=2)
        R = self.L / (lambda_3d + self.epsilon)
        
        # Clip to reasonable range instead of rescaling
        R = np.clip(R, 0, self.saturation_max)
        
        # Apply photon-budget-aligned losses
        if self.use_priors:
            # Prevent saturation
            R = self.prevent_saturation(R)
            
            # Confidence-weighted fidelity (disabled)
            R = self.confidence_weighted_fidelity(R, self.L, lambda_val)
            
            # Illumination-aware color consistency
            R = self.illumination_aware_color_consistency(R, lambda_val)
            
            # Uniform dark region enhancement
            R = self.enhance_detail_in_dark_regions(R, lambda_val)
            
            # L₆: Spatial consistency
            R = self.spatial_consistency_loss(R, self.L)
            
            # Optional brightness enhancement (post-processing, not physics)
            R = self.apply_brightness_enhancement(R)
        
        # Convert back to sRGB for display
        R = self.gamma_correct(R)
        R = np.clip(R, 0, 1)
        
        self.R = img_as_ubyte(R)
        
        return self.R






#!/usr/bin/env python3
"""
Test script for Physics-Based PBD algorithm with 6-loss architecture.

Loss Architecture:
- L₁: Poisson Negative Log-Likelihood (weight: 1.0)
- L₂: Photon Budget Range Constraint (weight: 0.1)
- L₃: Gradient Structure Consistency (weight: 0.05)
- L₄: Illumination Smoothness (weight: 0.15)
- L₅: Exposure Control (weight: 1.0)
- L₆: Spatial Consistency (weight: 0.5)
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from spd import PBD
import os

# Image path
img_path = "/media/hp/c587a0ea-5c63-499c-a609-e5e5362a9766/data/LIME/12.JPG"

# Initialize PBD with physics-based losses and iterative refinement
pbd = PBD(
    # ADMM parameters
    iterations=10,        # ADMM iterations for photon budget estimation
    alpha=2,              # Regularization parameter for structure-aware refinement
    rho=2,                # ADMM penalty parameter multiplier
    epsilon=1e-6,         # Small value to prevent division by zero
    beta=0.7,             # Weight for iterative refinement (anchors to observations)
    
    # Enhancement settings
    use_iteration=True,   # Enable iterative refinement (2-3 iterations)
    image_type='sRGB',    # Image type: 'sRGB' or 'RAW'
    strategy=2,           # Weighting strategy (1 or 2)
    use_priors=True,      # Enable physics-based priors
    
    # Physics-based loss weights (Tier 1: Core Physics)
    structure_weight=0.0, # L₃: Gradient Structure Consistency (disabled, redundant with ADMM)
    poisson_weight=0.0,   # L₁: Poisson Negative Log-Likelihood (removed from ADMM)
    
    # Physics-based loss weights (Tier 2: Perceptual Quality)
    smoothness_weight=0.15, # L₄: Illumination Smoothness
    exposure_weight=0.0,  # L₅: Exposure Control (disabled to prevent saturation)
    spatial_weight=0.5,   # L₆: Spatial Consistency
    
    # Noise modeling
    sigma_read=0.01,      # Read noise standard deviation for shot noise modeling
    
    # Other parameters (updated for saturation prevention)
    fidelity_weight=0.0,  # Confidence-weighted fidelity (disabled)
    saturation_max=1.0,   # Maximum radiance value before saturation penalty (reduced from 2.0)
    color_weight=0.02,    # Weight for illumination-aware color consistency
    detail_weight=0.15,   # Weight for detail enhancement (reduced from 0.5 to prevent saturation)
    target_brightness=0.5 # Target mean brightness for final output
)

print(f"Loading image: {img_path}")
pbd.load(img_path)
print(f"Image shape: {pbd.L.shape}")
print(f"Image path stored: {pbd.img_path}")

print("\n" + "="*60)
print("Physics-Based PBD Algorithm Configuration")
print("="*60)
print(f"Tier 1: Core Physics-Based Losses")
print(f"  L₁ (Poisson Likelihood):      weight = {pbd.poisson_weight} (removed from ADMM)")
print(f"  L₃ (Structure Consistency):  weight = {pbd.structure_weight} (disabled, redundant)")
print(f"\nTier 2: Perceptual Quality Losses")
print(f"  L₄ (Illumination Smoothness): weight = {pbd.smoothness_weight}")
print(f"  L₅ (Exposure Control):        weight = {pbd.exposure_weight} (optional post-processing)")
print(f"  L₆ (Spatial Consistency):     weight = {pbd.spatial_weight}")
print(f"\nNoise Modeling:")
print(f"  Read noise (σ_read):          {pbd.sigma_read}")
print(f"  Photon budget λ:              unbounded (absolute photon count proxy)")
print("="*60)

print("\nRunning PBD enhancement with physics-based losses...")
if pbd.use_iteration:
    print("Using iterative refinement mode")
else:
    print("Using single-pass enhancement mode")

enhanced = pbd.enhance()

print(f"\nEnhancement complete!")
print(f"Enhanced image shape: {enhanced.shape}")
print(f"Enhanced image dtype: {enhanced.dtype}")
print(f"Enhanced image range: [{enhanced.min()}, {enhanced.max()}]")

# Show statistics
if pbd.lambda_map is not None:
    print(f"\nPhoton Budget Statistics:")
    print(f"  Mean λ: {pbd.lambda_map.mean():.4f}")
    print(f"  Min λ:  {pbd.lambda_map.min():.4f}")
    print(f"  Max λ:  {pbd.lambda_map.max():.4f}")

if pbd.confidence_map is not None:
    print(f"\nConfidence Map Statistics:")
    print(f"  Mean confidence: {pbd.confidence_map.mean():.4f}")
    print(f"  Min confidence:  {pbd.confidence_map.min():.4f}")
    print(f"  Max confidence:  {pbd.confidence_map.max():.4f}")

# Save results
output_dir = "/media/hp/c587a0ea-5c63-499c-a609-e5e5362a9766/data/LIME/"
base_name = os.path.splitext(os.path.basename(img_path))[0]

# Save enhanced image
enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.jpg")
io.imsave(enhanced_path, enhanced)
print(f"\nSaved enhanced image: {enhanced_path}")

# Save photon budget map (lambda)
if pbd.lambda_map is not None:
    lambda_path = os.path.join(output_dir, f"{base_name}_photon_budget.png")
    plt.imsave(lambda_path, pbd.lambda_map, cmap='gray')
    print(f"Saved photon budget map: {lambda_path}")

# Save confidence map
if pbd.confidence_map is not None:
    confidence_path = os.path.join(output_dir, f"{base_name}_confidence.png")
    plt.imsave(confidence_path, pbd.confidence_map, cmap='hot')
    print(f"Saved confidence map: {confidence_path}")

print("\nAll results saved successfully!")

