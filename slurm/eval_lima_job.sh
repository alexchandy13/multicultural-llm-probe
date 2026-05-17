#!/bin/bash
#SBATCH --job-name=culture_eval_lima
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=slurm/eval_lima.%A_%a.out
#SBATCH --error=slurm/eval_lima.%A_%a.err

# Evaluation for the LIMA-variant conditions only (C2b, C4b). Two tasks:
# task 0 -> sft_lima, task 1 -> sftdpo_lima.
#
# Condition list is hard-coded here so this job doesn't accidentally inherit
# the primary CONDITIONS from env.sh. Mirrors slurm/eval_alpaca_job.sh.

set -euo pipefail
source env.sh

CONDS=(sft_lima sftdpo_lima)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_lima] condition=$COND"

python3.12 evaluate/eval_normad.py --condition "$COND"
