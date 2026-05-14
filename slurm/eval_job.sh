#!/bin/bash
#SBATCH --job-name=culture_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval.%A_%a.out
#SBATCH --error=slurm/eval.%A_%a.err

# Array job: one task per condition. Each task runs NormAd + CARE for that condition.
# Indices map to CONDITIONS in env.sh.

set -euo pipefail
source env.sh

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval] condition=$COND"

python3.12 evaluate/eval_normad.py --condition "$COND"
python3.12 evaluate/eval_care.py   --condition "$COND"
