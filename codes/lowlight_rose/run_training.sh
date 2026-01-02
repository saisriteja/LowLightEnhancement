# #!/bin/bash

# # ============================================================================
# # RoSe Low-Light Gaussian Splatting Training Script
# # ============================================================================
# # This script provides examples for running training with different configurations
# # for ablation studies and hyperparameter tuning.
# # ============================================================================

# # Base command (without CUDA_VISIBLE_DEVICES - set it separately)
# BASE_CMD="python simple_trainer.py default"

# # Base arguments (result_dir will be overridden per experiment)
# BASE_ARGS_BASE="--data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/buu --data_factor 1 --disable_viewer"

# # Set CUDA device
# export CUDA_VISIBLE_DEVICES=0

# # ============================================================================
# # ABLATION STUDY EXAMPLES
# # ============================================================================

# # 1. Standard 3DGS (baseline - no RoSe features)
# echo "=== Standard 3DGS (Baseline) ==="
# # Note: enable_rose defaults to False, so we omit it
# # eval "$BASE_CMD $BASE_ARGS_BASE --result_dir ./results/buu/baseline"

# # 2. Full RoSe (all features enabled)
# echo "=== Full RoSe Implementation ==="
# eval "$BASE_CMD $BASE_ARGS_BASE --result_dir ./results/buu/full_rose --enable_rose --enable_inverse_tone_curve --enable_illumination_correction --enable_low_rank_reg --enable_two_phase_training --enable_illuminance_constraint"

# # 3. Ablation: Without Inverse Tone Curve
# echo "=== RoSe without Inverse Tone Curve ==="
# eval "$BASE_CMD $BASE_ARGS_BASE --result_dir ./results/buu/ablation_no_tone_curve --enable_rose --no-enable_inverse_tone_curve --enable_illumination_correction --enable_low_rank_reg --enable_two_phase_training"

# # 4. Ablation: Without Illumination Correction Loss
# echo "=== RoSe without Illumination Correction Loss ==="
# eval "$BASE_CMD $BASE_ARGS_BASE --result_dir ./results/buu/ablation_no_ic_loss --enable_rose --enable_inverse_tone_curve --no-enable_illumination_correction --enable_low_rank_reg --enable_two_phase_training"

# # 5. Ablation: Without Low-Rank Regularization
# echo "=== RoSe without Low-Rank Regularization ==="
# eval "$BASE_CMD $BASE_ARGS_BASE --result_dir ./results/buu/ablation_no_lr_reg --enable_rose --enable_inverse_tone_curve --enable_illumination_correction --no-enable_low_rank_reg --enable_two_phase_training"

# # 6. Ablation: Without Two-Phase Training (joint optimization from start)
# echo "=== RoSe without Two-Phase Training ==="
# eval "$BASE_CMD $BASE_ARGS_BASE --result_dir ./results/buu/ablation_no_two_phase --enable_rose --enable_inverse_tone_curve --enable_illumination_correction --enable_low_rank_reg --no-enable_two_phase_training"

# # 7. Ablation: Without Illuminance Constraint
# echo "=== RoSe without Illuminance Range Constraint ==="
# eval "$BASE_CMD $BASE_ARGS_BASE --result_dir ./results/buu/ablation_no_constraint --enable_rose --enable_inverse_tone_curve --enable_illumination_correction --enable_low_rank_reg --enable_two_phase_training --no-enable_illuminance_constraint"

# # 8. Ablation: Only Inverse Tone Curve
# echo "=== Only Inverse Tone Curve (no other RoSe features) ==="
# eval "$BASE_CMD $BASE_ARGS_BASE --result_dir ./results/buu/ablation_only_tone_curve --enable_rose --enable_inverse_tone_curve --no-enable_illumination_correction --no-enable_low_rank_reg --no-enable_two_phase_training"

# # 9. Ablation: Only Illumination Correction
# echo "=== Only Illumination Correction Loss ==="
# eval "$BASE_CMD $BASE_ARGS_BASE --result_dir ./results/buu/ablation_only_ic --enable_rose --no-enable_inverse_tone_curve --enable_illumination_correction --no-enable_low_rank_reg --no-enable_two_phase_training"

# # 10. Ablation: Only Low-Rank Regularization
# echo "=== Only Low-Rank Regularization ==="
# eval "$BASE_CMD $BASE_ARGS_BASE --result_dir ./results/buu/ablation_only_lr_reg --enable_rose --no-enable_inverse_tone_curve --no-enable_illumination_correction --enable_low_rank_reg --no-enable_two_phase_training"

# # ============================================================================
# # HYPERPARAMETER TUNING EXAMPLES
# # ============================================================================

# # Tune illumination correction weight
# echo "=== Tune Lambda IC ==="
# eval "$BASE_CMD $BASE_ARGS_BASE --result_dir ./results/buu/tune_lambda_ic_1e2 --enable_rose --lambda_ic 1e-2"

# # Tune low-rank regularization weight
# echo "=== Tune Lambda LR ==="
# eval "$BASE_CMD $BASE_ARGS_BASE --result_dir ./results/buu/tune_lambda_lr_1e3 --enable_rose --lambda_lr 1e-3"

# # Tune geometry warmup steps
# echo "=== Tune Geometry Warmup Steps ==="
# eval "$BASE_CMD $BASE_ARGS_BASE --result_dir ./results/buu/tune_warmup_1000 --enable_rose --geometry_warmup_steps 1000"

# # Tune target illumination
# echo "=== Tune Target Illumination ==="
# eval "$BASE_CMD $BASE_ARGS_BASE --result_dir ./results/buu/tune_target_illum_0.5 --enable_rose --target_illumination 0.5"

# # Tune illuminance learning rate
# echo "=== Tune Illuminance Learning Rate ==="
# eval "$BASE_CMD $BASE_ARGS_BASE --result_dir ./results/buu/tune_illum_lr_5e4 --enable_rose --illuminance_lr 5e-4"

# # Tune KNN neighbors for low-rank loss
# echo "=== Tune KNN Neighbors ==="
# eval "$BASE_CMD $BASE_ARGS_BASE --result_dir ./results/buu/tune_knn_16 --enable_rose --knn_neighbors 16"

# ============================================================================
# FLAG DESCRIPTIONS
# ============================================================================
# --enable_rose: Master flag to enable RoSe mode (enables all RoSe features)
#                 When false, code behaves as standard 3DGS
#
# --enable_inverse_tone_curve: Apply inverse tone curve φ(x) to ground truth
#                               for gradient rebalancing of dark pixels
#
# --enable_illumination_correction: Enable L_IC loss to prevent brightness
#                                   scale ambiguity (c↓, i↑ degeneracy)
#
# --enable_low_rank_reg: Enable L_LR loss for spatial smoothness of illuminance
#                        (suppresses noise while preserving geometry)
#
# --enable_two_phase_training: Use Phase 1 (geometry warmup with i_i=1) +
#                              Phase 2 (joint optimization)
#
# --enable_illuminance_constraint: Constrain illuminance to [0.1, 1.5] range
#                                  for stability
#
# --lambda_ic: Weight for illumination correction loss (default: 1e-3)
#
# --lambda_lr: Weight for low-rank regularization loss (default: 1e-4)
#
# --target_illumination: Target mean illumination level (default: 0.45)
#
# --geometry_warmup_steps: Steps for Phase 1 training (default: 2000)
#
# --illuminance_lr: Learning rate for illuminance parameters (default: 1e-3)
#
# --illuminance_min: Minimum illuminance value (default: 0.1)
#
# --illuminance_max: Maximum illuminance value (default: 1.5)
#
# --knn_neighbors: Number of neighbors for low-rank loss (default: 8)
#
# --epsilon_tone: Small epsilon for inverse tone curve (default: 1e-6)
#
# ============================================================================
# USAGE
# ============================================================================
# 1. Uncomment the configuration you want to run
# 2. Modify parameters as needed
# 3. Run: bash run_training.sh
#
# Or run directly:
# export CUDA_VISIBLE_DEVICES=0
# python simple_trainer.py default \
#     --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/buu \
#     --data_factor 4 \
#     --result_dir ./results/buu \
#     --enable_rose true \
#     --enable_inverse_tone_curve true \
#     --enable_illumination_correction true \
#     --enable_low_rank_reg true \
#     --enable_two_phase_training true
# ============================================================================

# export CUDA_VISIBLE_DEVICES=1
# python simple_trainer.py default \
#     --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/buu \
#     --data_factor 1 \
#     --result_dir ./results/buu/brighter_tune_v1 \
#     --enable_rose \
#     --target_illumination 0.65 \
#     --lambda_ic 5e-3 \
#     --illuminance_max 2.0 \
#     --illuminance_lr 2e-3



export CUDA_VISIBLE_DEVICES=1
python simple_trainer.py default \
    --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/buu \
    --data_factor 1 \
    --result_dir ./results/buu/brighter_tune_v2_fixed \
    --enable_rose \
    --target_illumination 0.65 \
    --lambda_ic 0.05 \
    --lambda_lr 1e-4 \
    --illuminance_max 1.5 \
    --illuminance_min 0.3 \
    --illuminance_lr 5e-4 \
    # --enable_two_phase_training \
    # --geometry_warmup_steps 3000