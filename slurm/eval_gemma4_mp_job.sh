#!/bin/bash
#SBATCH --job-name=culture_eval_gemma4_mp
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_gemma4_mp.%A_%a.out
#SBATCH --error=slurm/eval_gemma4_mp.%A_%a.err

# Multi-prompt numbered eval for Gemma4 9B. Scores 4 NormAd-style prompt
# variants with shuffled 1/2 options; log-probs averaged before argmax.
# Outputs: normad_{cond}_gemma4_mp.json and normad_{cond}_gemma4_fs2_mp.json
# Submit with: CONDITIONS="base sft dpo sftdpo" sbatch slurm/eval_gemma4_mp_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_gemma4_mp] condition=$COND"

python evaluate/eval_normad.py --condition "$COND" --model-size gemma4 --multi-prompt --us-probe
python evaluate/eval_normad.py --condition "$COND" --model-size gemma4 --multi-prompt --few-shot 2 --us-probe
