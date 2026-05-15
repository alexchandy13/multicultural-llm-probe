#!/bin/bash
#SBATCH --job-name=culture_eval_alpaca
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=slurm/eval_alpaca.%A_%a.out
#SBATCH --error=slurm/eval_alpaca.%A_%a.err

# Evaluation for the Alpaca-variant conditions only (C2a, C4a). Two tasks:
# task 0 -> sft_alpaca, task 1 -> sftdpo_alpaca.
#
# The condition list is hard-coded here so this job doesn't accidentally inherit
# the primary `CONDITIONS` from env.sh; the primary eval_job.sh continues to use
# that variable for the HH-RLHF conditions.

set -euo pipefail
source env.sh

CONDS=(sft_alpaca sftdpo_alpaca)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_alpaca] condition=$COND"

python3.12 evaluate/eval_normad.py --condition "$COND"
