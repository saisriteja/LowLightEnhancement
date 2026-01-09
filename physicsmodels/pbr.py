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
                 use_priors=True, exposure_target=0.7, 
                 lambda_min=0.05, lambda_max=0.95, range_weight=0.05,
                 fidelity_weight=0.0, saturation_max=2.0,
                 color_weight=0.02, detail_weight=0.5, consistency_weight=0.1,
                 target_brightness=0.5):
        """
        Initialize PBD algorithm with photon-budget-aligned loss functions.
        
        Parameters:
        -----------
        iterations : int
            Number of ADMM iterations for photon budget estimation
        alpha : float
            Regularization parameter for structure-aware refinement
        rho : float
            ADMM penalty parameter multiplier
        epsilon : float
            Small value to prevent division by zero
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
        exposure_target : float
            Target average photon budget for exposure control (default: 0.7)
        lambda_min : float
            Minimum valid photon budget value (default: 0.05)
        lambda_max : float
            Maximum valid photon budget value (default: 0.95)
        range_weight : float
            Weight for range constraint loss (default: 0.05)
        fidelity_weight : float
            Weight for confidence-weighted fidelity loss (default: 0.0, disabled)
        saturation_max : float
            Maximum radiance value before saturation penalty (default: 2.0)
        color_weight : float
            Weight for illumination-aware color consistency (default: 0.02)
        detail_weight : float
            Weight for detail enhancement in dark regions (default: 0.5)
        consistency_weight : float
            Weight for λ-R consistency loss (default: 0.1)
        target_brightness : float
            Target mean brightness for final output (default: 0.5)
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
        self.exposure_target = exposure_target
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.range_weight = range_weight
        self.fidelity_weight = fidelity_weight
        self.saturation_max = saturation_max
        self.color_weight = color_weight
        self.detail_weight = detail_weight
        self.consistency_weight = consistency_weight
        self.target_brightness = target_brightness
        
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
    
    def __lambda_subproblem(self, G, Z, u):
        """
        Solve λ subproblem in ADMM optimization with range constraint.
        
        Loss 2: Photon Budget Range Constraint
        L_range = ||max(0, λ_min - λ(x))|| + ||max(0, λ(x) - λ_max)||
        
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
            Updated photon budget λ
        """
        # Adjust lambda_0 to target exposure level (more aggressive for dark images)
        if self.use_priors and self.exposure_target > 0:
            current_mean = np.mean(self.lambda_0)
            if current_mean < self.exposure_target:
                scale_factor = np.power(self.exposure_target / (current_mean + self.epsilon), 0.3)
                scale_factor = np.clip(scale_factor, 1.0, 4.0)  # Allow up to 4x boost
                lambda_target = self.lambda_0 * scale_factor
            else:
                lambda_target = self.lambda_0
        else:
            lambda_target = self.lambda_0
        
        X = G - Z / u
        Xv = X[:self.row, :]
        Xh = X[self.row:, :]
        temp = self.dv @ Xv + Xh @ self.dh
        
        # Add range constraint penalty term
        range_penalty = 0
        if self.use_priors and self.range_weight > 0:
            # This will be applied after solving, but we can add soft constraint
            range_penalty = self.range_weight * u
        
        numerator = fft.fft(vectorize(2 * lambda_target + u * temp))
        denominator = fft.fft(self.vecDD * u) + 2 + range_penalty
        denominator = denominator + self.epsilon
        lambda_val = fft.ifft(numerator / denominator)
        lambda_val = np.real(reshape(lambda_val, self.row, self.col))
        
        # Apply range constraint (Loss 2): clip to valid range
        if self.use_priors:
            lambda_val = np.clip(lambda_val, self.lambda_min, self.lambda_max)
        
        # Ensure no NaN or Inf values
        lambda_val = np.nan_to_num(lambda_val, nan=self.lambda_min, posinf=self.lambda_max, neginf=self.lambda_min)
        return lambda_val
    
    def __G_subproblem(self, lambda_val, Z, u, W):
        """
        Solve G subproblem in ADMM optimization.
        
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
        epsilon = self.alpha * W / u
        X = dLambda + Z / u
        return np.sign(X) * np.maximum(np.abs(X) - epsilon, 0)
    
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
        Compute confidence map from photon budget.
        
        C(x) = 1 - exp(-λ(x))
        
        High λ → C ≈ 1 → trust the observation
        Low λ → C ≈ 0 → insufficient photons
        
        Parameters:
        -----------
        lambda_val : ndarray
            Photon budget estimate
            
        Returns:
        --------
        ndarray
            Confidence map C in [0, 1]
        """
        return 1 - np.exp(-lambda_val * 3)
    
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
        
        Applies uniform brightness boost to ALL pixels in dark regions.
        Dark flat regions: boost brightness uniformly
        Dark textured regions: boost brightness uniformly (edges preserved naturally)
        Bright regions: leave alone
        
        Parameters:
        -----------
        R : ndarray
            Enhanced radiance (H, W, 3)
        lambda_val : ndarray
            Photon budget map (H, W)
            
        Returns:
        --------
        ndarray
            Uniformly enhanced radiance
        """
        if not self.use_priors or self.detail_weight <= 0:
            return R
        
        # Uniform boost in dark regions: every pixel gets brightened
        # Boost factor: stronger in darker regions (lower lambda)
        brightness_boost = 1.0 + (1.0 - lambda_val) * self.detail_weight
        
        # Clip boost to reasonable range
        brightness_boost = np.clip(brightness_boost, 1.0, 1.0 + self.detail_weight)
        
        # Apply uniform boost to all channels
        brightness_boost_3d = np.repeat(brightness_boost[:, :, np.newaxis], 3, axis=2)
        R_enhanced = R * brightness_boost_3d
        
        # Smooth transition: blend more in dark regions, less in bright regions
        blend_weight = (1.0 - lambda_val)[..., np.newaxis]
        R_final = (1.0 - blend_weight) * R + blend_weight * R_enhanced
        
        return np.clip(R_final, 0, self.saturation_max)
    


    def photon_budget_consistency(self, lambda_val, R):
        """
        Loss 7: Gradient Structure Consistency
        
        L_consistency = ||∇λ - ∇max_c(L)||²
        
        Preserves structural relationships without collapsing to identity.
        """
        if not self.use_priors or self.consistency_weight <= 0:
            return lambda_val
        
        # Compute gradients of λ
        grad_lambda_x = np.gradient(lambda_val, axis=1)
        grad_lambda_y = np.gradient(lambda_val, axis=0)
        
        # Compute gradients of observed max channel
        L_max = np.max(self.L, axis=2)
        grad_L_x = np.gradient(L_max, axis=1)
        grad_L_y = np.gradient(L_max, axis=0)
        
        # Consistency: gradients should align
        grad_error_x = grad_lambda_x - grad_L_x
        grad_error_y = grad_lambda_y - grad_L_y
        
        # Smooth adjustment based on gradient error
        lambda_adjusted = lambda_val - self.consistency_weight * (grad_error_x + grad_error_y) * 0.5
        
        return np.clip(lambda_adjusted, self.lambda_min, self.lambda_max)

        
    def enforce_target_brightness(self, R):
        """
        Loss 8: Target Brightness Enforcement
        
        Ensures output reaches desired brightness regardless of intermediate losses.
        If mean brightness is below target, scale up uniformly.
        
        Parameters:
        -----------
        R : ndarray
            Enhanced radiance (H, W, 3)
            
        Returns:
        --------
        ndarray
            Brightness-enforced radiance
        """
        if not self.use_priors or self.target_brightness <= 0:
            return R
        
        # Compute mean brightness
        R_gray = np.mean(R, axis=2)
        current_mean = np.mean(R_gray)
        
        # If below target, scale up
        if current_mean < self.target_brightness:
            scale_factor = self.target_brightness / (current_mean + self.epsilon)
            # Don't over-scale (max 2x to avoid artifacts)
            scale_factor = np.clip(scale_factor, 1.0, 2.0)
            R_enforced = R * scale_factor
        else:
            R_enforced = R
        
        return np.clip(R_enforced, 0, self.saturation_max)
    
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
        # Initial photon budget estimation
        lambda_val = self.photonBudgetMap()
        
        for k in range(num_iterations):
            # Estimate radiance: divide by illumination map
            lambda_3d = np.repeat(lambda_val[:, :, np.newaxis], 3, axis=2)
            R = self.L / (lambda_3d + self.epsilon)
            R = exposure.rescale_intensity(R, (0, 1))
            
            # Apply photon-budget-aligned losses
            if self.use_priors:
                # Loss 4: Prevent saturation
                R = self.prevent_saturation(R)
                
                # Loss 3: Confidence-weighted fidelity
                R = self.confidence_weighted_fidelity(R, self.L, lambda_val)
                
                # Loss 5: Illumination-aware color consistency
                R = self.illumination_aware_color_consistency(R, lambda_val)
                
                # Loss 6: Detail enhancement in dark regions
                R = self.enhance_detail_in_dark_regions(R, lambda_val)
            
            # Re-estimate photon budget from current radiance
            R_max = np.max(R, axis=2)
            lambda_new = self.beta * self.lambda_0 + (1 - self.beta) * R_max
            
            # Loss 7: Photon budget consistency
            if self.use_priors:
                lambda_new = self.photon_budget_consistency(lambda_new, R)
            
            # Refine photon budget with ADMM
            self.lambda_0 = lambda_new
            self.W = self.weightingStrategy()
            lambda_val = self.photonBudgetMap()
        
        # Final confidence and radiance computation
        self.lambda_map = lambda_val
        self.confidence_map = self.confidence(lambda_val)
        lambda_3d = np.repeat(lambda_val[:, :, np.newaxis], 3, axis=2)
        R = self.L / (lambda_3d + self.epsilon)
        R = exposure.rescale_intensity(R, (0, 1))
        
        # Apply final losses
        if self.use_priors:
            R = self.prevent_saturation(R)
            R = self.confidence_weighted_fidelity(R, self.L, lambda_val)
            R = self.illumination_aware_color_consistency(R, lambda_val)
            R = self.enhance_detail_in_dark_regions(R, lambda_val)
            # Loss 8: Enforce target brightness
            R = self.enforce_target_brightness(R)
        
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
        # Estimate photon budget (with Loss 2: range constraint)
        lambda_val = self.photonBudgetMap()
        
        # Compute confidence map
        self.confidence_map = self.confidence(lambda_val)
        
        # Estimate radiance: divide by illumination map
        lambda_3d = np.repeat(lambda_val[:, :, np.newaxis], 3, axis=2)
        R = self.L / (lambda_3d + self.epsilon)
        
        # Normalize to [0, 1]
        R = exposure.rescale_intensity(R, (0, 1))
        
        # Apply photon-budget-aligned losses
        if self.use_priors:
            # Loss 4: Prevent saturation
            R = self.prevent_saturation(R)
            
            # Loss 3: Confidence-weighted fidelity (disabled)
            R = self.confidence_weighted_fidelity(R, self.L, lambda_val)
            
            # Loss 5: Illumination-aware color consistency
            R = self.illumination_aware_color_consistency(R, lambda_val)
            
            # Loss 6: Uniform dark region enhancement
            R = self.enhance_detail_in_dark_regions(R, lambda_val)
            
            # Loss 8: Enforce target brightness
            R = self.enforce_target_brightness(R)
        
        # Convert back to sRGB for display
        R = self.gamma_correct(R)
        R = np.clip(R, 0, 1)
        
        self.R = img_as_ubyte(R)
        
        return self.R

