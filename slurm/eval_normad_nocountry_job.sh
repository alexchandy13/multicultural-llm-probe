#!/bin/bash
#SBATCH --job-name=normad_nocountry
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=1:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_normad_nocountry.%A_%a.out
#SBATCH --error=slurm/eval_normad_nocountry.%A_%a.err

# NormAd _fs2_yn with the country removed from the prompt.
#
# Gives the baseline the US-swap has been missing: us_pred answers "what does
# the model say if told this is the US", and this answers "what if told
# nothing". Agreement between the two separates real US-anchoring from the
# model simply ignoring the country -- which matters, because substituting any
# single country into every NormAd question costs only ~2 points of accuracy.
#
# Two modes per condition:
#   coherent  also drops " in this culture", so the question is well formed
#   drop      removes only the Country: line, leaving a dangling reference
# Both are run because "coherent" changes two things at once while "drop"
# asks a malformed question; if they agree the choice does not matter.
# culnig/dataset_ext.py records what a dangling reference did to the
# CulturalBench control, which is why "coherent" is the default.
#
# The few-shot shots are stripped to match the item, so the demonstrated
# format and the scored format are the same.
#
# Scoped to the two SFT+DPO conditions (cultural and generic); base and SFT
# are not run. 2 conditions x 2 modes = 4 tasks per model.
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

CONDS=(sftdpo_aya_cult sftdpo_aya_nocult)
MODES=(coherent drop)

N_MODES=${#MODES[@]}
COND=${CONDS[$(( SLURM_ARRAY_TASK_ID / N_MODES ))]}
MODE=${MODES[$(( SLURM_ARRAY_TASK_ID % N_MODES ))]}

echo "[normad_nocountry] condition=$COND mode=$MODE model=8b"

python evaluate/eval_normad.py \
    --condition "$COND" \
    --model-size 8b \
    --precision matched_bf16 \
    --few-shot 2 \
    --yn-only \
    --no-country "$MODE"
