#!/bin/bash
#SBATCH --job-name=culture_eval_8b_mpw
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_8b_mpw.%A_%a.out
#SBATCH --error=slurm/eval_8b_mpw.%A_%a.err

# Multi-prompt word eval for Llama 3.1 8B. Scores 4 NormAd-style prompt
# variants with direct yes/no answers; log-probs averaged before argmax.
# Outputs: normad_{cond}_8b_mpw_usprobe.json and normad_{cond}_8b_fs2_mpw_usprobe.json
# Submit with: CONDITIONS="base sft dpo sftdpo" sbatch slurm/eval_8b_mpw_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_8b_mpw] condition=$COND"

python evaluate/eval_normad.py --condition "$COND" --model-size 8b --multi-prompt-word --us-probe
python evaluate/eval_normad.py --condition "$COND" --model-size 8b --multi-prompt-word --few-shot 2 --us-probe
