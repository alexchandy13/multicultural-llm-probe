#!/bin/bash
#SBATCH --job-name=olmo3_zs
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/olmo3_zs.%A_%a.out
#SBATCH --error=slurm/olmo3_zs.%A_%a.err
#
# Zero-shot companion to eval_olmo3_suite_job.sh, matching how the aya conditions
# were run zero-shot so the two ladders are comparable:
#
#   normad         --yn-only            (2-way; the 3-way task is degenerate)
#   culturalbench  --no-fewshot         (drops the hardcoded neutral pair)
#   blend          no prefix flag
#
# --us-probe throughout, matching the existing zero-shot aya files, which all
# carry us_pred.
#
# Expect a high exact-tie rate here: with no prefix the yes/no margin often falls
# below bf16's resolution (0.0625 at this logit magnitude), and base_8b hit 19.7%.
# NormAd breaks those ties to "yes" and CulturalBench to "no", so read zero-shot
# skew with that in mind.
#
#   sbatch --array=0-2 slurm/eval_olmo3_zeroshot_job.sh

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra SIZES <<< "${SIZES:-olmo3 olmo3_sft olmo3_dpo}"
SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}
echo "[olmo3_zs] model_size=$SIZE"

echo "--- normad (0-shot, yn) ---"
python evaluate/eval_normad.py --condition base --model-size "$SIZE" --yn-only --us-probe

echo "--- culturalbench (0-shot) ---"
python evaluate/eval_culturalbench.py --condition base --model-size "$SIZE" --no-fewshot --us-probe

echo "--- blend (0-shot) ---"
python evaluate/eval_blend.py --condition base --model-size "$SIZE" --us-probe
