#!/bin/bash

#SBATCH --gres=gpu:1 --constraint=gpup100|gpuv100|gpua100
#SBATCH --job-name="GA-NAS"
#SBATCH -p publicgpu

source ~/.bashrc
module rm compilers/intel17
module load python/Anaconda3
module rm compilers/intel17
module load compilers/cuda-10.0
module rm compilers/intel17
source activate adv
module rm compilers/intel17

python3 main_args.py

