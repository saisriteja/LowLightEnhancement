# CUDA_VISIBLE_DEVICES=1,3 python simple_trainer.py default \
#     --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/bike 
#     --result_dir ./results/bike_lit


CUDA_VISIBLE_DEVICES=1,3 python hdr_trainer.py default \
    --data_dir /mnt/data0/teja/lowlight/LowLightEnhancement/dataset/LOM_full/bike \
    --result_dir ./results/bike_lit_hdr