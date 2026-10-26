#!/bin/bash
#SBATCH --job-name=normad_nocountry
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=1:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-6
#SBATCH --output=slurm/eval_normad_nocountry.%A_%a.out
#SBATCH --error=slurm/eval_normad_nocountry.%A_%a.err

# NormAd with the Country: line removed, everything else identical.
#
# Gives the baseline the US-swap has been missing: us_pred answers "what does
# the model say if told it is the US", and this answers "what does it say if
# told nothing". Agreement between the two separates genuine US-anchoring from
# the model simply ignoring the country.
#
# Paper setting: --neutral-fewshot --multi-prompt-word, so 4 templates per
# example over 1,818 examples = 7,272 forward passes per run. ~8 min on 8b,
# ~18 min on Gemma; 12 runs is about 2.5 GPU-hours.
#
# No --us-probe: the swapped predictions already exist in the
# *_nfs_mpw_usprobe_batch_calibrated.json files.
#
# Gemma-4-12B is ~24 GB in bf16, too tight for one A5000, so those tasks ask
# for two cards (device_map="auto" splits the layers).
#
# Submit with: sbatch slurm/eval_normad_nocountry_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

TASKS=(
  "8b     base"
  "8b     sft_aya_cult"
  "8b     sft_aya_nocult"
  "8b     sftdpo_aya_cult"
  "8b     sftdpo_aya_nocult"
  "8b     tulu3_sft"
  "8b     tulu3_dpo"
  "gemma4 base"
  "gemma4 sft_aya_cult"
  "gemma4 sft_aya_nocult"
  "gemma4 sftdpo_aya_cult"
  "gemma4 sftdpo_aya_nocult"
)

read -r SIZE COND <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "[normad_nocountry] size=$SIZE condition=$COND"

python evaluate/eval_normad.py \
    --condition "$COND" \
    --model-size "$SIZE" \
    --precision matched_bf16 \
    --neutral-fewshot \
    --multi-prompt-word \
    --no-country

OUT="outputs/behavioral/normad_${COND}_${SIZE}_nfs_mpw_nocountry.json"
echo "[normad_nocountry] batch-calibrating $OUT"
python scripts/calibrate_normad_batch.py "$OUT"
