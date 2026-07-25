#!/bin/bash
#SBATCH --job-name=culture_eval_tulu3_normad
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=slurm/eval_tulu3_normad.%A_%a.out
#SBATCH --error=slurm/eval_tulu3_normad.%A_%a.err

# NormAd eval for Tulu 3 (allenai/Llama-3.1-Tulu-3-8B-SFT and -DPO).
# Full merged models — no adapter loading needed.
# Submit with: CONDITIONS="tulu3_sft tulu3_dpo" sbatch slurm/eval_tulu3_normad_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

CONDS=(tulu3_sft tulu3_dpo)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_tulu3_normad] condition=$COND"

python evaluate/eval_normad.py --condition "$COND" --model-size 8b
python evaluate/eval_normad.py --condition "$COND" --model-size 8b --calibrate
