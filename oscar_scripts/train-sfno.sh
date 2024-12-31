#!/bin/bash

# Request runtime (HH:MM:SS):
#SBATCH --time=07:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1 --gres-flags=enforce-binding

# Default resources are 1 core with 2.8GB of memory.

#SBATCH --mem=140G

# Specify a job name:
#SBATCH -J TrainAceSto0.2

# Specify an output file
#SBATCH -o TrainAceSto0.2.out
#SBATCH -e TrainAceSto0.2.out


# Set up the environment by loading modules
module load cuda cudnn torch

# Run a script
source modulus.venv/bin/activate
python ../sfno-sai/sfno_train_val.py "AceSto0.2 (no SST)" -r 0.2
rm -rf ../.cache/wandb/artifacts/

