#!/bin/bash
#SBATCH --job-name=blend_nc_g4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --time=2:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=slurm/eval_blend_nc_g4.%A_%a.out
#SBATCH --error=slurm/eval_blend_nc_g4.%A_%a.err

# BLEnD with the "in <country>" phrase removed from every question.
#
# BLEnD's four options are each the correct answer for a DIFFERENT country, so
# with no country named the question has no right answer -- whichever option
# the model picks is the culture it defaults to. Each prediction records
# pred_country via the row's choice_countries map, so the output is a
# distribution over all 16 countries rather than a binary agreement with the
# US.
#
# Removal is one regex ("in X" / "in the X"), verified on all 2,400 sampled
# rows across all 16 countries, and leaves a grammatical question:
#   "What is a common snack for preschool kids in the UK?"
#   -> "What is a common snack for preschool kids?"
# So unlike NormAd there is no coherent/drop distinction to run.
#
# Scoped to the two SFT+DPO conditions, matching the NormAd ablation. No
# --us-probe: the swapped predictions already exist in the
# blend_*_nfs_usprobe.json files, and no --shuffle-options (refused, since
# choice_countries is keyed by the original letters).
#
# 23,032 items, one forward pass each: ~50 min on Gemma.
#
# Submit with: sbatch slurm/eval_blend_nocountry_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

CONDS=(sftdpo_aya_cult sftdpo_aya_nocult)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

echo "[blend_nocountry] condition=$COND model=gemma4"

python evaluate/eval_blend.py \
    --condition "$COND" \
    --model-size gemma4 \
    --precision matched_bf16 \
    --neutral-fewshot \
    --no-country
