this repo is for low light enhancement


get dataset
python -m venv env
source env/bin/activate
pip install gdown 
mkdir dataset
cd dataset
gdown 1orgKEGApjwCm6G8xaupwHKxMbT2s9IAG



source /mnt/data0/anaconda_dir/miniconda3/bin/activate teja_gsplat && export CUDA_HOME=$CONDA_PREFIX && export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH && export C_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$C_INCLUDE_PATH && echo "CUDA_HOME: $CUDA_HOME" && echo "Added include path: $CONDA_PREFIX/targets/x86_64-linux/include" && pip install git+https://github.com/rahul-goel/fused-ssim/ -v --no-build-isolation 2>&1 | tail -30

export CUDA_HOME=$CONDA_PREFIX
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$C_INCLUDE_PATH