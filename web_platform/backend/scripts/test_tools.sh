#!/bin/bash

#SBATCH --job-name=test-medrax-tools
#SBATCH -c 4
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=16:00:00
#SBATCH --mem=50G
#SBATCH --output=medgemma-%j.out
#SBATCH --error=medgemma-%j.err

tool=$1

image_path=$IMG_PATH_JPG
phrase="Nodule"
question="Identify the problem in this chest x-ray image and provide a brief explanation."
prompt="Identify the problem in this chest x-ray image and provide a brief explanation."
query="What is a pleural effusion?"

python web_platform/backend/scripts/test_tool.py --tool $1 --image-path "$image_path" --question "$question" --phrase "$phrase" --prompt "$prompt" --query "$query" --json