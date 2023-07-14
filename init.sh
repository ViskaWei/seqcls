#!/bin/bash

source ~/.bashrc

# Where's my Python
source /datascope/slurm/miniconda3/bin/activate viska-torch
export PYTHONPATH=.:../pysynphot:../SciScript-Python/py3

# Where's my PFS 
export ROOT=/home/swei20/NLP/
export DATA=/datascope/subaru/user/swei20/huggingface/datasets/
# export TEST=/scratch/ceph/swei20/data/ae/test


# Work around issues with saving weights when running on multiple threads
export HDF5_USE_FILE_LOCKING=FALSE

# Disable tensorflow deprecation warnings
export TF_CPP_MIN_LOG_LEVEL=2

# Enable more cores for numexpr
export NUMEXPR_MAX_THREADS=32


cd $ROOT

echo "LoRA, LoRA, LoRA!"
echo "Huggingface Data directory is $DATA"