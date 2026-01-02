I’ll assume **standard 3DGS (Inria)** as the base. we are upgrading to rose

---

# Step-by-Step RoSe-style Gaussian Splatting

---

## **Step 1: Start from Standard 3D Gaussian Splatting**

Each Gaussian ( G_i ) has:
[
G_i = (\mu_i, \Sigma_i, \alpha_i, c_i)
]

Rendering:
[
\hat{C}(p) = \sum_i w_i(p), c_i
]

This renders **observed color** directly.

---

## **Step 2: Redefine the Image Formation Model**

Replace direct color rendering with:

[
\boxed{
\hat{C}*{low}(p) = \hat{C}*{nor}(p) \odot \hat{I}(p)
}
]

where:

* ( \hat{C}_{nor} ): normal-light color
* ( \hat{I} ): illuminance transition

---

## **Step 3: Extend Gaussian Parameters**

Each Gaussian now stores:

[
G_i =
(\mu_i,\Sigma_i,\alpha_i,; c_i,; i_i)
]

| Parameter              | Meaning                |
| ---------------------- | ---------------------- |
| (c_i \in \mathbb{R}^3) | Normal-light RGB       |
| (i_i \in \mathbb{R}^1) | Illuminance transition |

⚠️ **Constraint**
(i_i) is **view-independent** (no SH).

---

## **Step 4: Normal-Light Color Rendering**

Use standard GS compositing:

[
\boxed{
\hat{C}_{nor}(p) = \sum_i w_i(p), c_i
}
]

Weights (w_i) come from:

* projected Gaussian
* opacity
* depth ordering

---

## **Step 5: Illuminance Transition Rendering**

Use the **same weights**:

[
\boxed{
\hat{I}(p) = \sum_i w_i(p), i_i
}
]

✔ ensures perfect alignment
✔ preserves geometry consistency

---

## **Step 6: Low-Light Image Reconstruction**

Final predicted image:

[
\boxed{
\hat{C}*{low}(p) = \hat{C}*{nor}(p) \odot \hat{I}(p)
}
]

This is the **only supervised output**.

---

## **Step 7: Initialization**

Initialize parameters as:

[
c_i \leftarrow \text{mean observed color}
]

[
i_i \leftarrow 1.0
]

This avoids trivial collapse.

---

## **Step 8: Inverse Tone Curve (Pre-Loss)**

Dark pixels have weak gradients → rebalance them:

[
\phi(x) =
\frac{1}{2} -
\sin!\left(
\frac{\sin^{-1}(1 - 2x)}{3}
\right)
]

Apply to GT only:
[
C'*{low} = \phi(C*{low} + \epsilon)
]

---

## **Step 9: Reconstruction Loss (Main Loss)**

[
\boxed{
\mathcal{L}_{rec}
=================

\sum_p
\left|
\hat{C}*{low}(p) - C'*{low}(p)
\right|^2
}
]

✔ Uses **only low-light images**
✔ Standard GS training loop remains intact

---

## **Step 10: Illumination Correction Loss**

Without supervision, brightness scale is ambiguous.

Enforce a target illumination level:

[
\boxed{
\mathcal{L}_{IC}
================

\left(
\text{mean}(\hat{C}_{nor}) - e
\right)^2
}
]

Where:

* (e = 0.45)

✔ prevents (c \downarrow, i \uparrow) degeneracy

---

## **Step 11: Low-Rank Illumination Regularization**

Illumination is:

* smooth
* spatially correlated
* low-rank

Noise is high-rank.

---

### Neighborhood-based low-rank loss

For spatial neighbors ((i,j)):

[
\boxed{
\mathcal{L}_{LR}
================

\sum_{(i,j)}
| i_i - i_j |^2
}
]

✔ suppresses noise
✔ preserves geometry

---

## **Step 12: Illuminance Range Constraint**

Avoid instability:

[
0.1 \le i_i \le 1.5
]

Implementation:
[
i_i = 0.1 + 1.4 \cdot \sigma(\tilde{i}_i)
]

(No explicit loss term needed.)

---

## **Step 13: Training Schedule (Mandatory)**

### Phase 1 — Geometry warm-up

* Fix (i_i = 1)
* Optimize standard GS
* ~1–2k iterations

---

### Phase 2 — Joint optimization

* Unfreeze (i_i)
* Enable all losses
* Continue training

---

## **Step 14: Final Loss Function**

[
\boxed{
\mathcal{L}
===========

\mathcal{L}*{rec}
+
\lambda*{IC}\mathcal{L}*{IC}
+
\lambda*{LR}\mathcal{L}_{LR}
}
]

Recommended:

```text
λ_IC = 1e-3
λ_LR = 1e-4
```

---

## **Step 15: Inference**

Render:
[
\hat{C}_{nor},; \hat{I}
]

Optionally output:

* enhanced image: ( \hat{C}_{nor} )
* exposure-controlled image: ( \hat{C}_{nor} \odot \alpha )

---
