#!/bin/bash

#SBATCH --job-name=csv_bench
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu-v100

#SBATCH --output=.cache/csv_bench/%j.out
#SBATCH --error=.cache/csv_bench/%j.err

echo "-----------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_NODELIST"
echo "Start: $(date)"
echo "-----------------------------"

source ~/.bashrc

conda activate medrax
# source .venv/medrax/bin/activate

export HYDRA_FULL_ERROR=1

filename=${1:-"medrax_evals_vqa.csv"}
rows=${2:-""}
jobs=${3:-1}
basedir="benchmarking/eval_sets"

cmd=(python -m benchmarking.scripts.run_from_csv "$basedir/$filename" --jobs "$jobs")
if [[ -n "$rows" ]]; then
  cmd+=(--rows "$rows")
fi

"${cmd[@]}"

echo "-----------------------------"
echo "End: $(date)"
echo "-----------------------------"
