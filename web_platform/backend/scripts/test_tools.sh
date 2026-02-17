#!/bin/bash

#SBATCH --job-name=test-medrax-tools
#SBATCH -c 4
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=16:00:00
#SBATCH --mem=50G
#SBATCH --output=medgemma-%j.out
#SBATCH --error=medgemma-%j.err

export MODELWEIGHTS="/home/andrew.lian/wanglab/MedRAX2/model-weights"
export MODEL_CACHE_DIR="$HOME/.cache/huggingface"

tool=$1

image_path="/home/andrew.lian/wanglab/mimic/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
phrase="Nodule"
question="Identify the problem in this chest x-ray image and provide a brief explanation."
prompt="Identify the problem in this chest x-ray image and provide a brief explanation."
query="What is a pleural effusion?"

python web_platform/backend/scripts/test_tool.py --tool $1 --image-path "$image_path" --question "$question" --phrase "$phrase" --prompt "$prompt" --query "$query" --json