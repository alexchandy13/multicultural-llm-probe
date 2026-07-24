#!/bin/bash
#SBATCH --job-name=culture_eval_gemma4_blend_nfs
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_gemma4_blend_nfs.%A_%a.out
#SBATCH --error=slurm/eval_gemma4_blend_nfs.%A_%a.err

# BLEnD neutral few-shot eval for Gemma 4 12B.
# 4 culturally-agnostic shots (one per letter A/B/C/D) teach MCQ format only.
# Output: blend_{cond}_gemma4_nfs_usprobe.json
# Submit with: CONDITIONS="base sft dpo sftdpo" sbatch slurm/eval_gemma4_blend_nfs_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_gemma4_blend_nfs] condition=$COND"

python evaluate/eval_blend.py --condition "$COND" --model-size gemma4 --neutral-fewshot --us-probe
