#!/bin/bash
#SBATCH --job-name=culture_eval_gemma4_normad_nfs
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_gemma4_normad_nfs.%A_%a.out
#SBATCH --error=slurm/eval_gemma4_normad_nfs.%A_%a.err

# NormAd neutral few-shot eval for Gemma 4 12B.
# 2 culturally-agnostic shots (1 yes + 1 no) teach task format only.
# Outputs: normad_{cond}_gemma4_nfs_yn_usprobe.json
#          normad_{cond}_gemma4_nfs_mpw_usprobe.json
# Submit with: CONDITIONS="base sft dpo sftdpo" sbatch slurm/eval_gemma4_normad_nfs_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_gemma4_normad_nfs] condition=$COND"

python evaluate/eval_normad.py --condition "$COND" --model-size gemma4 --neutral-fewshot --yn-only --us-probe
python evaluate/eval_normad.py --condition "$COND" --model-size gemma4 --neutral-fewshot --multi-prompt-word --us-probe
