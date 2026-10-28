#!/bin/bash
#SBATCH --job-name=normad_nc_g4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --time=1:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-0
#SBATCH --output=slurm/eval_normad_nc_g4.%A_%a.out
#SBATCH --error=slurm/eval_normad_nc_g4.%A_%a.err

# NormAd _fs2_yn with the country removed from the prompt.
#
# Gives the baseline the US-swap has been missing: us_pred answers "what does
# the model say if told this is the US", and this answers "what if told
# nothing". Agreement between the two separates real US-anchoring from the
# model simply ignoring the country -- which matters, because substituting any
# single country into every NormAd question costs only ~2 points of accuracy.
#
# The country-free prompt comes from a hand-written paired template
# (NOCOUNTRY_TEMPLATES), so the instruction no longer refers to a country
# either -- deleting the Country: line alone would leave the instruction
# promising one, which is not a neutral control.
#
# The few-shot shots are stripped to match the item, so the demonstrated
# format and the scored format are the same.
#
# Scoped to the generic SFT+DPO condition only; 1 task per model.
#
# 1,818 examples, one forward pass each (fs2_yn is single-template), so ~2 min
# on 8b and ~4 on Gemma. Gemma-4-12B is ~24 GB in bf16, too tight for one
# A5000, so its tasks request two cards.
#
# Submit with: sbatch slurm/eval_normad_nocountry_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

CONDS=(sftdpo_aya_nocult)

COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

echo "[normad_nocountry] condition=$COND model=gemma4"

python evaluate/eval_normad.py \
    --condition "$COND" \
    --model-size gemma4 \
    --precision matched_bf16 \
    --few-shot 2 \
    --yn-only \
    --no-country
