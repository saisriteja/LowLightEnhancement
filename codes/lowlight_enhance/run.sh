#!/bin/bash
# Training script using new modular structure

# Option 1: Use TOML config file (recommended)
# CUDA_VISIBLE_DEVICES=1,3 python train.py --config config.toml

# Option 2: Run specific experiment from config
# CUDA_VISIBLE_DEVICES=1,3 python train.py --config config.toml --experiment_idx 0

# Option 3: Use old hdr_trainer.py (deprecated, kept for reference)
# CUDA_VISIBLE_DEVICES=1,3 python hdr_trainer.py default \
#     --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/bike \
#     --result_dir ./results/bike_lit_hdr \
#     --enable_intrinsic_decomp \
#     --enable_synthetic_exposures \
#     --init_opa 0.5 \
#     --init_scale 0.1 \
#     --opacity_reg 0.01 \
#     --scale_reg 0.01

# New modular approach - update config.toml with your settings, then run:
CUDA_VISIBLE_DEVICES=1,3 python train.py --config config.toml