#!/bin/bash
#SBATCH --job-name=culture_eval_8b_mc
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_8b_mc.%A_%a.out
#SBATCH --error=slurm/eval_8b_mc.%A_%a.err

# MC-format (letter A/B/C) eval for Llama 3.1 8B.
# Scores single letter tokens instead of words to avoid surface-form competition
# (Holtzman et al. 2021, arXiv 2104.08315) — "neutral" is structurally underscored
# because high-frequency 'n'-initial tokens suppress its log-prob regardless of prompt.
# Submit with: CONDITIONS="base sft dpo sftdpo" sbatch slurm/eval_8b_mc_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_8b_mc] condition=$COND"

python evaluate/eval_normad.py --condition "$COND" --model-size 8b --mc-format
python evaluate/eval_normad.py --condition "$COND" --model-size 8b --mc-format --calibrate
