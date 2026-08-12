#!/bin/bash
#SBATCH --job-name=eval_instruct_gemma4
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=18:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/eval_instruct_gemma4.%j.out
#SBATCH --error=slurm/eval_instruct_gemma4.%j.err

# Full eval suite for Gemma 4 12B Instruct (Google's official instruct model)
# using the standard protocol: nfs + mpw + us-probe + batch calibration.
# Submit with: sbatch slurm/eval_instruct_gemma4_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MS=gemma4_instruct
echo "[eval_instruct_gemma4] model_size=$MS"

# NormAd
python evaluate/eval_normad.py --condition base --model-size $MS --neutral-fewshot --multi-prompt-word --us-probe
python scripts/calibrate_normad_batch.py outputs/behavioral/normad_base_${MS}_nfs_mpw_usprobe.json

# BLEnD
python evaluate/eval_blend.py --condition base --model-size $MS --neutral-fewshot --us-probe

# NLU
python evaluate/eval_nlu.py --condition base --model-size $MS --dataset boolq --neutral-fewshot
python evaluate/eval_nlu.py --condition base --model-size $MS --dataset csqa  --neutral-fewshot
