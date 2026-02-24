#!/bin/bash

#SBATCH --job-name=medgemma
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=50G

# optional
#SBATCH --partition=gpu-v100

#SBATCH --output=.cache/medgemma/medgemma-%j.out
#SBATCH --error=.cache/medgemma/medgemma-%j.err

export MEDGEMMA_DEVICE=cuda

# source .venv/medgemma/bin/activate
# echo "VIRTUAL_ENV=$VIRTUAL_ENV"

source $HOME/.bashrc
conda activate medgemma

which python


nvidia-smi

root_dir=${1:-"."}

python medrax/tools/vqa/medgemma/medgemma.py --root-dir "$root_dir"
