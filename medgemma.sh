#!/bin/bash

#SBATCH --job-name=medgemma
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=50G
#SBATCH --reservation=mkoziarski_gpu
#SBATCH --output=.cache/medgemma/medgemma-%j.out
#SBATCH --error=.cache/medgemma/medgemma-%j.err

export MEDGEMMA_DEVICE=cuda

source ~/.bashrc
conda activate medrax_python

source .venv/medgemma/bin/activate

nvidia-smi

root_dir=${1:-"."}

python medrax/tools/vqa/medgemma/medgemma.py --root-dir "$root_dir"
