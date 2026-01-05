
export CUDA_VISIBLE_DEVICES=1
python simple_trainer.py default \
    --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/buu \
    --data_factor 1 \
    --result_dir ./results/buu_ssim/