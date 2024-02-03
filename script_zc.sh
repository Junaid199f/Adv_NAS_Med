#!/bin/bash
#SBATCH -A girimasg -p grantgpu 
#SBATCH --gres=gpu:1 --constraint=gpu1080|gpup100|gpuv100
#SBATCH --job-name="ZCNAS"

source ~/.bashrc

module load python/Anaconda3
source activate adv

python3 main_args.py
