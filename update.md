Below is the **same decomposition + losses**, but written in **3D Gaussian Splatting (3DGS)** form (alpha-compositing / “over” operator), keeping the spirit of your NeRF equations.

---

## 1) Retinex-style decomposition (kept the same)

Low-light RGB image (per pixel) is modeled as
[
\mathbf{C}*{low}(\mathbf{p}) ;=; \mathbf{C}*{nor}(\mathbf{p}) \odot I(\mathbf{p}),
]
where (\mathbf{p}) is a pixel, (\mathbf{C}_{nor}\in \mathbb{R}^3), and (I\in\mathbb{R}) is the **illuminance transition** (relative illumination ratio).

---

## 2) Gaussian Splatting rendering for (\mathbf{C}_{nor}) and (I)

Assume a set of Gaussians ({\mathcal{G}*j}*{j=1}^{M}). Each Gaussian (j) has:

* 3D mean (\boldsymbol\mu_j), covariance/shape (\Sigma_j) (or scale+rotation),
* opacity parameter (or density-like) (\alpha_j),
* **normal-light color** (\mathbf{c}_j(\mathbf{v}) \in \mathbb{R}^3) (may be view-dependent via SH; (\mathbf{v}) is view direction),
* **illuminance transition value** (i_j \in \mathbb{R}) (world-centered → view-independent).

For a pixel (\mathbf{p}), project each Gaussian to the image plane → gives a 2D Gaussian footprint (G_j(\mathbf{p})\in[0,1]).
Define the per-pixel alpha contribution:
[
a_j(\mathbf{p}) ;=; \mathrm{clamp}\big(\alpha_j , G_j(\mathbf{p}),, 0,, 1\big).
]

Sort Gaussians by depth along the ray for (\mathbf{p}) (front-to-back). Define transmittance:
[
T_1(\mathbf{p}) = 1,\qquad
T_j(\mathbf{p}) = \prod_{k<j}\big(1-a_k(\mathbf{p})\big).
]

Define the standard 3DGS compositing weights:
[
w_j(\mathbf{p}) ;=; T_j(\mathbf{p}), a_j(\mathbf{p}).
]

### Render normal-light color

[
\widehat{\mathbf{C}}*{nor}(\mathbf{p})
;=;
\sum*{j=1}^{M} w_j(\mathbf{p}) , \mathbf{c}_j(\mathbf{v}).
]

### Render illuminance transition (same weights, but scalar attribute)

[
\widehat{I}(\mathbf{p})
;=;
\sum_{j=1}^{M} w_j(\mathbf{p}) , i_j.
]

### Reconstruct low-light pixel

[
\widehat{\mathbf{C}}*{low}(\mathbf{p})
;=;
\widehat{\mathbf{C}}*{nor}(\mathbf{p}) \odot \widehat{I}(\mathbf{p}).
]

That is the **Gaussian-splatting analogue** of your Eq. (6)/(8)/(3): both branches share the same visibility weights (w_j), but predict different per-primitive attributes.

---

## 3) Losses (Gaussian-splatting form)

### (A) Tone-rebalanced reconstruction loss (pixel MSE)

Use the same inverse tone curve (\phi(\cdot)) and epsilon (\varepsilon):
[
\phi(x)= \frac{1}{2} - \sin!\left(\frac{\sin^{-1}(1-2x)}{3}\right),
\qquad \varepsilon = 10^{-3}.
]

Then (over all pixels (\mathbf{p}) in the training images):
[
\mathcal{L}_{MSE}
=================

\sum_{\mathbf{p}}
\left|
\widehat{\mathbf{C}}_{low}(\mathbf{p})
--------------------------------------

\phi!\left(\mathbf{C}_{low}(\mathbf{p}) + \varepsilon\right)
\right|_2^2 .
]

*(This is exactly your NeRF regression loss, just with (\widehat{\mathbf{C}}_{low}) coming from splatting instead of volume rendering.)*

---

### (B) Illumination correction loss (global intensity target on (\widehat{\mathbf{C}}_{nor}))

Let (e) be the desired illumination level (e.g., 0.45). With global average pooling over pixels:
[
\mathcal{L}_{IC}
================

\left(
\mathrm{GAP}\big(\widehat{\mathbf{C}}_{nor}\big) - e
\right)^2.
]

---

### (C) Total loss

[
\mathcal{L}
===========

\mathcal{L}*{MSE}
+
\lambda,\mathcal{L}*{IC}.
]

