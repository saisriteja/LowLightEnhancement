# `update.md`

## Commit

* **From**: `89256e34577ff6a1173ed9abcf44d7aef3359898`
* **To**: `5cd591fba705e8a6b1b940c5dc91fff9dcfc5cbb`
* **Intent**: Stabilize training by improving loss function robustness and adjusting hyperparameters

---

## 1. Mathematical Change

**What changed**

* Variance clamping in Poisson-Gaussian NLL loss: `variance = rendered.detach().clamp(min=1e-2, max=10.0) + eps`
* Log term weighting: `residual + 0.1 * log_term` (reduced from `residual + log_term`)

**Why**

* Unbounded variance could cause numerical instability and negative loss values
* Log term dominated loss signal, making training less stable
* Clamping prevents extreme variance values that destabilize gradients

---

## 2. Mathematical Effect

**Before**

* Variance could become very small or very large, causing log term to dominate
* Loss could go negative due to large log term contribution
* Training was unstable with frequent gradient explosions

**After**

* Variance is clamped to [1e-2, 10.0], ensuring numerical stability
* Log term contributes only 10% weight, reducing its dominance
* Loss remains positive and gradients are more stable

This change makes the model:

* More stable during training
* Less prone to gradient explosions
* Better at converging with consistent loss signals

---

## 3. Code Additions (New Functions)

Functions introduced to implement the above math:

* None (modifications to existing functions)

---

## 4. Code Changes (Modified Functions)

Existing functions updated to support the new formulation:

* `poisson_gaussian_nll_loss`
* `LossComputer.compute_ratio_loss`
* `Runner.__init__`
* `Runner.train_step`
* `Runner.plot_losses`

---

## 5. Execution Flow (After Commit)

High-level flow:

1. Training loop calls `poisson_gaussian_nll_loss` with clamped variance
2. Variance computation applies `.clamp(min=1e-2, max=10.0)` before adding epsilon
3. Loss combines residual term with weighted log term (0.1x)
4. Ratio loss computation uses `apply_exposure=True` for consistent exposure handling
5. Gradient clipping increased to max_norm=5.0 for stability
6. Learning rates reduced by 0.5x to prevent overshooting
7. Pipeline continues with more stable optimization

---

## 6. One-Line Summary

> **Adds variance clamping and log term weighting via `poisson_gaussian_nll_loss` to fix training instability from unbounded variance and dominant log terms.**

# Results
The results are too messy and didnt improve so dropping this method and switching back to original init verisons if necessary.