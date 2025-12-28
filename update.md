Mathematical Intuition: Multi-Exposure Intrinsic Factorization
The Core Problem
You observe a 3D point p from multiple cameras at multiple exposures. Each observation gives you:

text
I_k = intensity measured at observation k
But what you want is:

R(p): What material is the surface? (reflectance/albedo)

L(e): How bright is the lighting? (illumination)

V(view): Is the surface visible from this angle? (occlusion/shading)

So physically:

text
I_k = R(p) × L(e_k) × V(view_k)
Why This is Ambiguous from a Single Image
Single image, single observation:

text
I_single = R × L × V

You have: 1 measurement
Unknown: 3 quantities (R, L, V)
Degrees of freedom: ∞ (infinitely many solutions)

Example: I_single = 0.5
Could be:
  R=0.5, L=1,   V=1   (bright surface, no shadow)
  R=1,   L=0.5, V=1   (reflective, dim light)
  R=0.5, L=1,   V=0.5 (bright material, half-occluded)
  ... (infinite others)
The problem: You cannot distinguish cause from effect.

Why Multi-View + Multi-Exposure RESOLVES the Ambiguity
Key Insight: Constraints are Different for Each Component
View A, Exposure e₁:

text
I_{A,e1} = R ⊙ L(e_1) ⊙ V(A)
View A, Exposure e₂: (same view, different exposure)

text
I_{A,e2} = R ⊙ L(e_2) ⊙ V(A)
Now, here's the magic:

R is the SAME (same material)

L CHANGES (different exposure)

V is the SAME (same view)

Taking the ratio:

text
I_{A,e1} / I_{A,e2} = L(e_1) / L(e_2)

This tells you: illumination ratio between exposures!
Second Constraint: Different Views, Same Exposure
View A, Exposure e₁:

text
I_{A,e1} = R ⊙ L(e_1) ⊙ V(A)
View B, Exposure e₁: (different view, same exposure)

text
I_{B,e1} = R ⊙ L(e_1) ⊙ V(B)
Again:

R is the SAME (same material)

L is the SAME (same exposure)

V CHANGES (different view)

Taking the ratio:

text
I_{A,e1} / I_{B,e1} = V(A) / V(B)

This tells you: visibility ratio between views!
Third Constraint: Different Everything
View A, Exposure e₁:

text
I_{A,e1} = R ⊙ L(e_1) ⊙ V(A)
View B, Exposure e₂:

text
I_{B,e2} = R ⊙ L(e_2) ⊙ V(B)
Now ALL three components could differ. But here's the constraint:

text
R must be CONSISTENT across all observations
This means: If you estimate R from observation 1, you should get the SAME R from observation 2, 3, 4, etc.

Why? Because R is a physical material property—it doesn't change.

The Mathematical System
Over-Constrained Formulation
For a single 3D point seen K times:

text
Observation 1: I_1 = R ⊙ L(e_1) ⊙ V(C_1)
Observation 2: I_2 = R ⊙ L(e_2) ⊙ V(C_2)
Observation 3: I_3 = R ⊙ L(e_3) ⊙ V(C_3)
...
Observation K: I_K = R ⊙ L(e_K) ⊙ V(C_K)
Count:

K equations (one per observation)

Unknowns:

R: 3 values (RGB reflectance)

L(·): A function of exposure (captured by ~5-10 parameters in an MLP)

V(·): A function of view (captured by ~5-10 parameters in an MLP)

Total unknowns: ~25 parameters

Typical K: 5-20 observations (multiple cameras × multiple exposures)

Result: K >> 25

System Status: MASSIVELY OVER-DETERMINED ✓

Why Over-Determined = Unique Solution
Principle: When you have more equations than unknowns, the system is over-constrained.

text
Over-determined system (K >> unknowns):
  - May have no exact solution (noisy measurements)
  - Will have a unique LEAST-SQUARES solution
  - Small perturbations don't change solution much
  - Solution is statistically identifiable
In our case:

text
K observations >> 25 parameters
→ Solution is UNIQUE (up to global scale)

Why unique?
- If two different (R, L, V) triples both explain all observations
- They would need to satisfy K equations simultaneously
- But with K >> unknowns, probability of this = 0 (measure-theoretic sense)
- So solution is essentially unique
The Three Components Separately
Component 1: Reflectance R
Property: Constant across all observations

text
From observation 1: R̂_1 = I_1 / (L(e_1) × V(C_1))
From observation 2: R̂_2 = I_2 / (L(e_2) × V(C_2))
From observation 3: R̂_3 = I_3 / (L(e_3) × V(C_3))

If we knew L and V perfectly:
  R̂_1 = R̂_2 = R̂_3 (identical!)

Loss function:
  Minimize: ||R̂_1 - R̂_2||² + ||R̂_2 - R̂_3||² + ...
  
This FORCES R to be consistent across all observations.
Why this works:

Reflectance is a material property

It doesn't change with exposure or view

Multi-view consistency is a powerful constraint

Component 2: Illumination L(e)
Property: Smooth function of exposure

text
Physical model:
  When you increase exposure by 1 EV, light output doubles
  L(e+1) ≈ 2 × L(e)
  
In log-space:
  log(L(e)) ≈ e (approximately linear)
  
Constraint:
  d²(log L) / de² ≈ 0 (smooth, no jumps)
  
Loss function:
  Minimize: Σ_e || d²(log L)/de² ||²
  
This FORCES L to be a smooth curve in exposure space.
Why this works:

Exposure response is a physical property of cameras

Light intensity changes smoothly with exposure

No realistic camera has discontinuous response

MLPs naturally learn smooth functions

Component 3: Visibility V(view)
Property: Smooth function of view direction

text
Physical model:
  When camera angle changes slightly, occlusion changes smoothly
  V(view_a) ≈ V(view_b) if views are nearby
  
Constraint:
  For nearby camera poses: ||V(view_a) - V(view_b)||² ≈ 0
  
Loss function:
  Minimize: Σ_(nearby pairs) exp(-distance) × ||V(A) - V(B)||²
  
This FORCES V to change smoothly across views.
Why this works:

Occlusion doesn't jump discontinuously

Shadows and highlights vary gradually with view

Smooth functions are lower-dimensional (regularization)

The Uniqueness Proof (Intuitive)
Claim: If you have K ≥ 3 multi-view multi-exposure observations, R, L, V are identifiable.

Intuitive Proof:

text
Suppose two solutions (R₁, L₁, V₁) and (R₂, L₂, V₂) both fit all observations.

Then for all k:
  R₁ ⊙ L₁(e_k) ⊙ V₁(C_k) = R₂ ⊙ L₂(e_k) ⊙ V₂(C_k)

Taking element-wise ratios (per color channel):
  R₁/R₂ = [L₂(e_k) / L₁(e_k)] ⊙ [V₂(C_k) / V₁(C_k)]

Left side: Constant (doesn't depend on k)
Right side: Depends on k (exposure and view change)

For this to hold for ALL k:
  - L₂(e_k) / L₁(e_k) must be constant for all exposures
    → But two MLPs can't produce proportional outputs for all inputs
  - V₂(C_k) / V₁(C_k) must be constant for all views
    → But two MLPs can't produce proportional outputs for all inputs

Contradiction! So only one solution exists (up to global scale).
Why NO Ground Truth is Needed
Traditional approach (supervised):

text
Given: 
  - Low-light image I_low
  - Normal-light image I_normal (ground truth)

Learn: Mapping from I_low → I_normal
Problem: Requires expensive paired data
Your approach (self-supervised):

text
Given:
  - Multiple low-light images from different cameras/exposures
  - Camera poses (from SfM)
  - Exposure values (from metadata)

Constraint:
  - Multi-view consistency
  - Exposure smoothness
  - Visibility smoothness
  
Recover: R, L, V without any normal-light ground truth!

Why it works:
  - Multi-view geometry ITSELF provides supervision
  - Same material must look same when normalized
  - No external data needed
The Loss Functions (Intuition)
Loss 1: Photometric (Fit the observations)
text
L_photo = ||C_rendered - I_observed||²

Simple: Your rendering should match what you see.
Loss 2: Reflectance Consistency ⭐ CRITICAL
text
L_reflect = Σ_{i,j} ||R̂_i - R̂_j||²

where R̂_k = I_k / (L(e_k) × V(C_k))

Forces: All observations should agree on reflectance when normalized.
Why: Reflectance is a material property (constant).
Loss 3: Illumination Smoothness
text
L_illum = Σ_e || d²(log L)/de² ||²

Forces: Illumination curve is smooth in exposure space.
Why: Light intensity changes smoothly with exposure.
Loss 4: Visibility Smoothness
text
L_vis = Σ_(nearby pairs) || V(view_a) - V(view_b) ||²

Forces: Visibility changes gradually across nearby views.
Why: Occlusion state doesn't jump discontinuously.
Loss 5: Exposure Consistency
text
L_expose = ||normalize(C(e_1)) - normalize(C(e_2))||

Forces: Geometry is consistent across exposures.
Why: 3D structure doesn't change with exposure.
Loss 6: Exposure Smoothness
text
L_smooth = Σ_e ||∇_e [Δc, Δα, Δσ]||²

Forces: Perturbations change smoothly with exposure.
Why: Tone mapping is a smooth function of exposure.
Loss 7: Regularization
text
L_reg = ||Δc||² + ||Δα||² + ||Δσ||²

Forces: Perturbations stay small.
Why: We want MLPs to be "helpers," not "fixers."
Why This Beats 2D Methods
2D Single-Image Intrinsic (Ambiguous)
text
I = R ⊙ S

1 equation, 2 unknowns → infinite solutions
Requires: External priors (learned from ImageNet, etc.)
Result: Many possible decompositions
Your 3D Multi-View Multi-Exposure (Unique)
text
I_1 = R ⊙ L(e_1) ⊙ V(C_1)
I_2 = R ⊙ L(e_2) ⊙ V(C_2)
...
I_K = R ⊙ L(e_K) ⊙ V(C_K)

K equations >> unknowns → unique solution
Requires: Nothing but geometry
Result: One physically correct decomposition
The advantage: 3D geometry IS the supervision.

Why Combining Multi-Exposure + Intrinsics is Powerful
Multi-Exposure Alone (Idea 5)
text
Learns: Empirical corrections (Δc, Δα, Δσ)
Benefit: Very accurate tone mapping
Limitation: No physical interpretation, can hallucinate
Intrinsic Factorization Alone (Novel)
text
Learns: Physical decomposition (R, L, V)
Benefit: Interpretable, generalizable
Limitation: More complex, harder to train
COMBINED ✓
text
Intrinsics: Provide physical correctness + regularization
Multi-exposure: Learn empirical shortcuts + practical accuracy

Result:
  - Fast convergence (empirical + physical)
  - Better generalization (over-constrained)
  - Interpretable decompositions (intrinsics)
  - Competitive PSNR (perturbations)
The Scale Ambiguity (Unavoidable)
Fundamental issue:

text
If (R, L, V) solves the system, so does (αR, L/α, V) for any α > 0.

Proof:
  α·R ⊙ (L/α) ⊙ V = R ⊙ L ⊙ V ✓

Interpretation:
  Can't tell if surface is bright (high R) with dim light (low L)
  Or surface is dark (low R) with bright light (high L)
Solution:

text
Fix illumination at reference exposure:
  L(exposure = 0) = [1, 1, 1]

Now:
  R is identifiable uniquely
  L is identifiable uniquely
Why Smoothness Regularization Matters
Without smoothness loss:

text
MLPs could fit K observations with K degrees of freedom
→ Each observation has its own hidden variables
→ No sharing → overfitting → poor generalization
With smoothness loss:

text
MLPs must be smooth functions
→ Fewer effective degrees of freedom
→ Forced to generalize
→ Better extrapolation to unseen exposures
Example:

text
Train on exposures: [-2, -1, 0, 1, 2] EV
Test on exposure: [2.5] EV (unseen)

Without smoothness: Breaks (overfits)
With smoothness: Works (smooth extrapolation)
The Exposure Curriculum (Why It Helps)
text
Early training (epochs 1-300):
  Use small exposure range: [-1, 1] EV
  Why: Easy, helps find good initial solution
  
Middle training (epochs 300-600):
  Use medium range: [-2, 2] EV
  Why: Expand coverage, refine details
  
Late training (epochs 600-1000):
  Use full range: [-3, 3] EV
  Why: Final refinement, extreme exposures
Why curriculum works:

Avoids local minima early on

Stable convergence

Naturally progressive difficulty

Better final solution

Summary: The Complete Picture
text
OBSERVATION:
  Multiple images of same scene from different cameras/exposures

PHYSICS:
  I_k = R ⊙ L(e_k) ⊙ V(C_k)

CONSTRAINT:
  - R is constant (material)
  - L is smooth in e (exposure)
  - V is smooth in C (view)
  - System is over-determined (K >> unknowns)

RESULT:
  - Unique solution exists (no ambiguity)
  - No ground truth needed (geometry provides supervision)
  - Self-supervised learning possible

YOUR IMPLEMENTATION:
  - Learn R directly (per-Gaussian)
  - Learn L with IlluminationMLP
  - Learn V with VisibilityMLP
  - Perturb adaptively with PerturbationMLP
  
OUTCOME:
  - Physical decompositions
  - Empirical accuracy
  - Better generalization
  - No GT required