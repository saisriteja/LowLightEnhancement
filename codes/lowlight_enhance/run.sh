#!/bin/bash
# Training script for HDR Gaussian Splatting (original standalone version)

# Single GPU training
# CUDA_VISIBLE_DEVICES=0 python train.py default \
#     --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/bike \
#     --result_dir ./results/bike_lit_hdr

# Distributed training on multiple GPUs
CUDA_VISIBLE_DEVICES=1,3 python train.py default \
    --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/bike \
    --result_dir ./results/bike_lit_hdr \
    --steps_scaler 1

# MCMC strategy example:
# CUDA_VISIBLE_DEVICES=1,3 python train.py mcmc \
#     --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/bike \
#     --result_dir ./results/bike_lit_hdr_mcmc \
#     --init_opa 0.5 \
#     --init_scale 0.1 \
#     --opacity_reg 0.01 \
#     --scale_reg 0.01