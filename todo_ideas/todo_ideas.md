Idea 5: Multi-Exposure Gaussian Decomposition (Base + Perturbations) ⭐ Best Feasibility
Why it matters: Exposure is a continuous dimension; instead of learning per-image exposure scalars, learn a base scene + small learnable perturbations per exposure. Enables interpolation/extrapolation of exposures at test time without retraining.

Technical Approach:

Gaussian Perturbation Field:

G
rendered
(
e
)
=
G
base
+
Δ
(
e
)
G 
rendered
 (e)=G 
base
 +Δ(e)
where:

G
base
G 
base
  = Standard 3D Gaussians at canonical (normal-light) exposure.

Δ
(
e
)
Δ(e) = Small MLP-predicted perturbations for exposure 
e
e.

Perturbation Network (lightweight, per-Gaussian):

Δ
i
(
e
)
=
MLP
Δ
(
x
i
,
view
,
e
)
→
[
Δ
c
i
,
Δ
α
i
,
Δ
s
i
]
Δ 
i
 (e)=MLP 
Δ
 (x 
i
 ,view,e)→[Δc 
i
 ,Δα 
i
 ,Δs 
i
 ]
Input: Gaussian position, view direction, exposure.

Output: Color shift, opacity shift, scale shift.

Small MLPs (2–3 layers, 32 dims): ~10 params per Gaussian, lightweight.

Exposure Curriculum Training:

Start with images within 1 EV of median exposure.

Gradually expand exposure range during training.

Forces perturbation network to learn meaningful variations.

Consistency Loss (across exposures):

L
consist
=
∣
∣
normalize
(
G
i
+
Δ
(
e
1
)
)
−
normalize
(
G
i
+
Δ
(
e
2
)
)
∣
∣
1
L 
consist
 =∣∣normalize(G 
i
 +Δ(e 
1
 ))−normalize(G 
i
 +Δ(e 
2
 ))∣∣ 
1
 
Ensures geometry doesn't change drastically across exposures (only appearance).

Inference:

Set target exposure 
e
target
e 
target
  (e.g., normal-light exposure).

Render: 
G
base
+
Δ
(
e
target
)
G 
base
 +Δ(e 
target
 ).

Automatic relighting, no separate enhancement network.

Advantages:

✓ Minimal parameters: only small MLPs per Gaussian.

✓ Exposure becomes controllable dimension.

✓ Can interpolate/extrapolate exposures smoothly.

✓ Interpretable: what changes as exposure increases?

✓ Fast: no iterative optimization at inference.

Challenges:

✗ MLP inference per Gaussian per view (slight compute overhead).

✗ Overfitting if regularization is weak.

Mitigation:

Use shared MLP weight parameters across Gaussians (weight sharing reduces params 10×).

Very fast MLPs (2 layers, 32 hidden dims).

Strong L2 regularization on 
Δ
Δ, sparse perturbations.












Idea 4: Implicit 3D Illumination Field with Inverse Rendering ⭐ Highest Impact
Why it matters: True intrinsic decomposition (geometry + reflectance + illumination) is the holy grail of inverse rendering. Enables arbitrary relighting without retraining.
​

Technical Approach:

Separate 3D illumination from radiance:

R
(
x
,
ω
)
=
A
(
x
)
⊙
L
(
x
,
ω
in
)
R(x,ω)=A(x)⊙L(x,ω 
in
 )
where:

A
(
x
)
A(x) = Albedo (intrinsic reflectance, per-Gaussian).

L
(
x
,
ω
in
)
L(x,ω 
in
 ) = 3D illumination field (neural MLP).

⊙
⊙ = Element-wise product (Lambertian + specular).

Illumination MLP:

L
MLP
(
x
,
ω
in
∣
θ
)
→
[
R
diff
,
R
spec
]
∈
R
3
L 
MLP
 (x,ω 
in
 ∣θ)→[R 
diff
 ,R 
spec
 ]∈R 
3
 
Takes 3D position + view direction, outputs spatially-varying illumination SH coefficients.

Decoupling Losses:

L
=
L
rendering
+
λ
albedo
L
albedo
(
A
∈
[
0
,
1
]
)
+
λ
smooth
L
smooth
(
∇
L
)
+
λ
phys
L
physics
(
L
)
L=L 
rendering
 +λ 
albedo
 L 
albedo
 (A∈[0,1])+λ 
smooth
 L 
smooth
 (∇L)+λ 
phys
 L 
physics
 (L)
Training with paired exposures (LOM normal-light views):

Low-light: 
C
low
=
A
⊙
(
L
⋅
e
low
)
C 
low
 =A⊙(L⋅e 
low
 )

Normal-light: 
C
normal
=
A
⊙
(
L
⋅
e
normal
)
C 
normal
 =A⊙(L⋅e 
normal
 )

Enforce: Normalize out exposure, match intrinsic: 
L
consistency
=
∣
∣
A
(
l
)
−
A
(
n
)
∣
∣
1
L 
consistency
 =∣∣A(l)−A(n)∣∣ 
1
 

Inference (relighting):

Render with target illumination: 
L
target
=
L
base
+
δ
L
gain
L 
target
 =L 
base
 +δL 
gain
 

No retraining needed; arbitrary lighting possible.

Advantages:

✓ True physics-based: enables relighting, material editing.

✓ Handles extreme low-light: separates signal from degradation.

✓ Generalizes beyond training exposures.

✓ Connects to inverse rendering literature (GI-GS, OSDR-GS, GS-ID ).
​

Challenges:

✗ Ambiguity: many (A, L) pairs explain same image (classic inverse rendering problem).

✗ Requires regularization or additional supervision.

Mitigation:

Use normal-light GT to regularize albedo (consistency loss).

Add diffusion prior on rendered albedo (LL-Gaussian style).

Photometric loss on low-light + high-light pairs.









Idea 1: Normalizing Flow-Based Enhancement Branch ⭐ Recommended
Why it matters: Normalizing flow (LLFlow, BGFlow, ADANF) successfully model the one-to-many relationship in low-light enhancement—one dark image corresponds to many plausible normal-light versions. Standard L1 loss treats all possibilities equally; flows model the conditional distribution.
​

Technical Approach:

Each Gaussian learns a latent code 
z
i
∈
R
8
z 
i
 ∈R 
8
  during training.

A conditional encoder extracts illumination-invariant features from low-light rendered views: 
f
i
=
E
(
C
low
,
view
)
f 
i
 =E(C 
low
 ,view).

A reversible flow decoder maps latent codes through affine coupling layers, conditioned on features:

z
i
→
Flow
(
z
i
;
f
i
)
→
Δ
c
i
z 
i
 →Flow(z 
i
 ;f 
i
 )→Δc 
i
 
where 
Δ
c
i
Δc 
i
  is a color enhancement offset.

Wavelet-domain brightness guidance (from BGFlow): Extract multi-scale brightness features in the wavelet domain of rendered low-light views, then inject into coupling layers to guide luminance separation from texture.

Inference: To render normal-light, sample from a learned prior distribution (or set 
z
i
=
0
z 
i
 =0) and decode.

Loss Function:

L
=
L
NLL
(
rendered
,
GT
low
)
+
λ
flow
⋅
L
flow
(
z
i
,
f
i
→
GT
normal
)
+
λ
smooth
⋅
L
smooth
(
Δ
c
)
L=L 
NLL
 (rendered,GT 
low
 )+λ 
flow
 ⋅L 
flow
 (z 
i
 ,f 
i
 →GT 
normal
 )+λ 
smooth
 ⋅L 
smooth
 (Δc)
Advantages:

✓ Probabilistic: captures uncertainty in enhancement (multimodal outputs).

✓ Invertible: no adversarial training, stable gradients.

✓ Feature-guided: wavelet-domain guidance separates luminance from texture preservation.

✓ Proven in 2D: LLFlow and BGFlow are SOTA on LOL-v1, SID, MIT-Adobe.

Feasibility: 7/10 (invertible networks are established; main challenge is conditioning architecture).
Novelty: 9/10 (first application of conditional normalizing flows to 3D low-light scene reconstruction).
ECCV Strength: 8/10 (strong methodological contribution, clear comparison to Luminance-GS).
Timeline: 2–3 weeks.