#!/bin/bash
# Training script for Multi-Exposure Gaussian Splatting

# Single GPU training with multi-exposure
# CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
#     --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/bike \
#     --result_dir ./results/bike_multi_exposure \
#     --enable_multi_exposure \
#     --steps_scaler 1

# Distributed training on multiple GPUs with multi-exposure
CUDA_VISIBLE_DEVICES=1,3 python simple_trainer.py default \
    --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/bike \
    --result_dir ./results/bike_multi_exposure \
    --enable_multi_exposure \
    --steps_scaler 1 \
    --lambda_consist 0.5 \
    --lambda_smooth 0.1 \
    --lambda_reg 1e-4 \
    --mlp_lr 1e-4

# MCMC strategy with multi-exposure:
# CUDA_VISIBLE_DEVICES=1,3 python simple_trainer.py mcmc \
#     --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/bike \
#     --result_dir ./results/bike_multi_exposure_mcmc \
#     --enable_multi_exposure \
#     --init_opa 0.5 \
#     --init_scale 0.1 \
#     --opacity_reg 0.01 \
#     --scale_reg 0.01 \
#     --lambda_consist 0.5 \
#     --lambda_smooth 0.1 \
#     --lambda_reg 1e-4

# Standard training without multi-exposure (for comparison):
# CUDA_VISIBLE_DEVICES=1,3 python simple_trainer.py default \
#     --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/bike \
#     --result_dir ./results/bike_standard \
#     --enable_multi_exposure False \
#     --steps_scaler 1