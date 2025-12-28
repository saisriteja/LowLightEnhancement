# Distributed training on multiple GPUs with multi-exposure and intrinsic factorization
# Updated with new loss weights and parameters from update.md
# CUDA_VISIBLE_DEVICES=1,3 python simple_trainer.py default \
#     --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/bike \
#     --result_dir ./results/bike_multi_exposure_v3_debug \
#     --enable_multi_exposure \
#     --enable_intrinsic_factorization \
#     --steps_scaler 1 \
#     --lambda_consist 5.0 \
#     --lambda_smooth 10.0 \
#     --lambda_reg 0.01 \
#     --lambda_structure 2.0 \
#     --mlp_lr 1e-4 \
#     --lambda_reflect 10.0 \
#     --lambda_illum_smooth 1.0 \
#     --lambda_vis_smooth 0.1 \
#     --lambda_reflect_spatial 1.0 \
#     --illumination_mlp_lr 1e-4 \
#     --visibility_mlp_lr 1e-4 \
#     --max_delta_c 0.1 \
#     --max_delta_alpha 0.05 \
#     --max_delta_sigma 0.05 \
#     --exposure_curriculum_phases 3000 10000 20000 \
#     --consist_loss_freq 1 \
#     --debug_save_interval 5000

# # ============================================================================
# # DEBUGGING COMMANDS - Test one at a time, sequentially
# # ============================================================================

# # Step 1: Baseline - Disable ALL MLPs (test base Gaussian rendering)
# # Expected: Should render normally without any MLP effects
# # If this fails → base Gaussian training is broken
# # If this works → proceed to Step 2
# CUDA_VISIBLE_DEVICES=1,3 python simple_trainer.py default \
#     --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/bike \
#     --result_dir ./results/debug_step1_no_mlps \
#     --enable_multi_exposure \
#     --enable_intrinsic_factorization \
#     --debug_disable_perturbation_mlp \
#     --debug_disable_illumination_mlp \
#     --debug_disable_visibility_mlp \
#     --debug_save_interval 5000 \
#     --steps_scaler 1 \
#     --lambda_consist 5.0 \
#     --lambda_smooth 10.0 \
#     --lambda_reg 0.01 \
#     --lambda_structure 2.0 \
#     --mlp_lr 1e-4 \
#     --lambda_reflect 10.0 \
#     --lambda_illum_smooth 1.0 \
#     --lambda_vis_smooth 0.1 \
#     --lambda_reflect_spatial 1.0 \
#     --illumination_mlp_lr 1e-4 \
#     --visibility_mlp_lr 1e-4 \
#     --max_delta_c 0.1 \
#     --max_delta_alpha 0.05 \
#     --max_delta_sigma 0.05 \
#     --exposure_curriculum_phases 3000 10000 20000 \
#     --consist_loss_freq 1

# # Step 2: Enable ONLY Illumination MLP (disable perturbation MLP)
# # Expected: Should only change brightness, NOT structure/geometry
# # If structure changes → illumination MLP is too powerful
# # If only brightness changes → illumination MLP is working correctly
CUDA_VISIBLE_DEVICES=1,3 python simple_trainer.py default \
    --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/bike \
    --result_dir ./results/debug_step2_illum_only_v5 \
    --enable_multi_exposure \
    --enable_intrinsic_factorization \
    --debug_disable_perturbation_mlp \
    --debug_save_interval 5000 \
    --steps_scaler 1 \
    --lambda_consist 5.0 \
    --lambda_smooth 10.0 \
    --lambda_reg 0.01 \
    --lambda_structure 2.0 \
    --mlp_lr 1e-4 \
    --lambda_reflect 10.0 \
    --lambda_illum_smooth 1.0 \
    --lambda_vis_smooth 0.1 \
    --lambda_reflect_spatial 1.0 \
    --illumination_mlp_lr 1e-4 \
    --visibility_mlp_lr 1e-4 \
    --max_delta_c 0.1 \
    --max_delta_alpha 0.05 \
    --max_delta_sigma 0.05 \
    --exposure_curriculum_phases 3000 10000 20000 \
    --consist_loss_freq 1

# # Step 3: Enable ONLY Visibility MLP (disable perturbation and illumination MLPs)
# # Expected: Should only change view-dependent effects, NOT structure
# CUDA_VISIBLE_DEVICES=1,3 python simple_trainer.py default \
#     --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/bike \
#     --result_dir ./results/debug_step3_vis_only \
#     --enable_multi_exposure \
#     --enable_intrinsic_factorization \
#     --debug_disable_perturbation_mlp \
#     --debug_disable_illumination_mlp \
#     --debug_save_interval 5000 \
#     --steps_scaler 1 \
#     --lambda_consist 5.0 \
#     --lambda_smooth 10.0 \
#     --lambda_reg 0.01 \
#     --lambda_structure 2.0 \
#     --mlp_lr 1e-4 \
#     --lambda_reflect 10.0 \
#     --lambda_illum_smooth 1.0 \
#     --lambda_vis_smooth 0.1 \
#     --lambda_reflect_spatial 1.0 \
#     --illumination_mlp_lr 1e-4 \
#     --visibility_mlp_lr 1e-4 \
#     --max_delta_c 0.1 \
#     --max_delta_alpha 0.05 \
#     --max_delta_sigma 0.05 \
#     --exposure_curriculum_phases 3000 10000 20000 \
#     --consist_loss_freq 1

# # Step 4: Enable ONLY Perturbation MLP (disable illumination and visibility MLPs)
# # Expected: Should change geometry/structure based on exposure
# CUDA_VISIBLE_DEVICES=1,3 python simple_trainer.py default \
#     --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/bike \
#     --result_dir ./results/debug_step4_perturb_only \
#     --enable_multi_exposure \
#     --enable_intrinsic_factorization \
#     --debug_disable_illumination_mlp \
#     --debug_disable_visibility_mlp \
#     --debug_save_interval 5000 \
#     --steps_scaler 1 \
#     --lambda_consist 5.0 \
#     --lambda_smooth 10.0 \
#     --lambda_reg 0.01 \
#     --lambda_structure 2.0 \
#     --mlp_lr 1e-4 \
#     --lambda_reflect 10.0 \
#     --lambda_illum_smooth 1.0 \
#     --lambda_vis_smooth 0.1 \
#     --lambda_reflect_spatial 1.0 \
#     --illumination_mlp_lr 1e-4 \
#     --visibility_mlp_lr 1e-4 \
#     --max_delta_c 0.1 \
#     --max_delta_alpha 0.05 \
#     --max_delta_sigma 0.05 \
#     --exposure_curriculum_phases 3000 10000 20000 \
#     --consist_loss_freq 1

# # Step 5: Enable Illumination + Visibility MLPs (disable perturbation MLP)
# # Expected: Should combine illumination and visibility effects without geometry changes
# CUDA_VISIBLE_DEVICES=1,3 python simple_trainer.py default \
#     --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/bike \
#     --result_dir ./results/debug_step5_illum_vis \
#     --enable_multi_exposure \
#     --enable_intrinsic_factorization \
#     --debug_disable_perturbation_mlp \
#     --debug_save_interval 5000 \
#     --steps_scaler 1 \
#     --lambda_consist 5.0 \
#     --lambda_smooth 10.0 \
#     --lambda_reg 0.01 \
#     --lambda_structure 2.0 \
#     --mlp_lr 1e-4 \
#     --lambda_reflect 10.0 \
#     --lambda_illum_smooth 1.0 \
#     --lambda_vis_smooth 0.1 \
#     --lambda_reflect_spatial 1.0 \
#     --illumination_mlp_lr 1e-4 \
#     --visibility_mlp_lr 1e-4 \
#     --max_delta_c 0.1 \
#     --max_delta_alpha 0.05 \
#     --max_delta_sigma 0.05 \
#     --exposure_curriculum_phases 3000 10000 20000 \
#     --consist_loss_freq 1

