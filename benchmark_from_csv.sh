#!/bin/bash

#SBATCH --job-name=csv_bench
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=100G

#SBATCH --output=.cache/csv_bench/%j.out
#SBATCH --error=.cache/csv_bench/%j.err

echo "-----------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_NODELIST"
echo "Start: $(date)"
echo "-----------------------------"

source ~/.bashrc

conda activate medrax_python
source .venv/medrax/bin/activate

filename=${1:-"medrax_evals_vqa.csv"}
basedir="benchmarking/eval_sets"

python -m benchmarking.scripts.run_from_csv $basedir/$filename --jobs 1

echo "-----------------------------"
echo "End: $(date)"
echo "-----------------------------"