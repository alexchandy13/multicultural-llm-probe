#!/bin/bash
#SBATCH --job-name=culture_eval_tulu3_blend
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=slurm/eval_tulu3_blend.%A_%a.out
#SBATCH --error=slurm/eval_tulu3_blend.%A_%a.err

# BLEnD eval for Tulu 3 (allenai/Llama-3.1-Tulu-3-8B-SFT and -DPO).
# Full merged models — no adapter loading needed.
# Submit with: sbatch slurm/eval_tulu3_blend_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

CONDS=(tulu3_sft tulu3_dpo)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_tulu3_blend] condition=$COND"

python evaluate/eval_blend.py --condition "$COND" --model-size 8b --us-probe
python evaluate/eval_blend.py --condition "$COND" --model-size 8b --multi-prompt --us-probe
python evaluate/eval_blend.py --condition "$COND" --model-size 8b --few-shot 2 --us-probe
python evaluate/eval_blend.py --condition "$COND" --model-size 8b --few-shot 2 --multi-prompt --us-probe
