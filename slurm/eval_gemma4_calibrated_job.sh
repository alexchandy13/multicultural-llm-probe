#!/bin/bash
#SBATCH --job-name=culture_eval_gemma4_cal
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_gemma4_cal.%A_%a.out
#SBATCH --error=slurm/eval_gemma4_cal.%A_%a.err

# Calibrated-only eval for gemma4 — runs after uncalibrated results already exist.
# Submit with: CONDITIONS="base sft dpo sftdpo" sbatch slurm/eval_gemma4_calibrated_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_gemma4_cal] condition=$COND"

python evaluate/eval_normad.py --condition "$COND" --model-size gemma4 --calibrate
