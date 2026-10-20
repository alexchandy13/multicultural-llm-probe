#!/bin/bash
#SBATCH --job-name=probe_blend_full
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-52
#SBATCH --output=slurm/eval_probe_blend_full.%A_%a.out
#SBATCH --error=slurm/eval_probe_blend_full.%A_%a.err

# Bring every BLEnD cluster-probe cell to a full 23,032-example probe pass.
#
# 31 of the 84 cells are already full and are not in this list. The other 53
# are either missing (18) or were run at --sample-frac 0.13 (35) and get
# overwritten at full. Running them all full removes the n=2932 vs n=23032
# mismatch between countries, and between sizes for netherlands.
#
# ~23 min per 8b task, ~48 min per gemma4 task; 28 and 25 of each, so ~31
# GPU-hours in total, spread over the array.
#
# NOT --skip-existing: the 35 resampled cells must be overwritten. Back them up
# first (178 MB) so a failed task cannot lose the 13% data it replaces:
#   mkdir -p outputs/behavioral/_probe13_backup
#   for f in outputs/behavioral/blend_*_nfs_*probe.json; do
#     python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('probe_sample_frac')==0.13 else 1)" "$f" \
#       && cp "$f" outputs/behavioral/_probe13_backup/
#   done
#
# Submit with: sbatch slurm/eval_probe_blend_full_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

# "<model-size> <condition> <probe-country>"
TASKS=(
  "8b     base              hungary"
  "8b     base              bosnia_and_herzegovina"
  "8b     base              mexico"
  "8b     base              philippines"
  "8b     sft_aya_cult      hungary"
  "8b     sft_aya_cult      bosnia_and_herzegovina"
  "8b     sft_aya_cult      mexico"
  "8b     sft_aya_cult      philippines"
  "8b     sft_aya_nocult    hungary"
  "8b     sft_aya_nocult    bosnia_and_herzegovina"
  "8b     sft_aya_nocult    mexico"
  "8b     sft_aya_nocult    philippines"
  "8b     sftdpo_aya_cult   hungary"
  "8b     sftdpo_aya_cult   bosnia_and_herzegovina"
  "8b     sftdpo_aya_cult   mexico"
  "8b     sftdpo_aya_cult   philippines"
  "8b     sftdpo_aya_nocult hungary"
  "8b     sftdpo_aya_nocult bosnia_and_herzegovina"
  "8b     sftdpo_aya_nocult mexico"
  "8b     sftdpo_aya_nocult philippines"
  "8b     tulu3_sft         hungary"
  "8b     tulu3_sft         bosnia_and_herzegovina"
  "8b     tulu3_sft         mexico"
  "8b     tulu3_sft         philippines"
  "8b     tulu3_dpo         hungary"
  "8b     tulu3_dpo         bosnia_and_herzegovina"
  "8b     tulu3_dpo         mexico"
  "8b     tulu3_dpo         philippines"
  "gemma4 base              netherlands"
  "gemma4 base              hungary"
  "gemma4 base              bosnia_and_herzegovina"
  "gemma4 base              mexico"
  "gemma4 base              philippines"
  "gemma4 sft_aya_cult      netherlands"
  "gemma4 sft_aya_cult      hungary"
  "gemma4 sft_aya_cult      bosnia_and_herzegovina"
  "gemma4 sft_aya_cult      mexico"
  "gemma4 sft_aya_cult      philippines"
  "gemma4 sft_aya_nocult    netherlands"
  "gemma4 sft_aya_nocult    hungary"
  "gemma4 sft_aya_nocult    bosnia_and_herzegovina"
  "gemma4 sft_aya_nocult    mexico"
  "gemma4 sft_aya_nocult    philippines"
  "gemma4 sftdpo_aya_cult   netherlands"
  "gemma4 sftdpo_aya_cult   hungary"
  "gemma4 sftdpo_aya_cult   bosnia_and_herzegovina"
  "gemma4 sftdpo_aya_cult   mexico"
  "gemma4 sftdpo_aya_cult   philippines"
  "gemma4 sftdpo_aya_nocult netherlands"
  "gemma4 sftdpo_aya_nocult hungary"
  "gemma4 sftdpo_aya_nocult bosnia_and_herzegovina"
  "gemma4 sftdpo_aya_nocult mexico"
  "gemma4 sftdpo_aya_nocult philippines"
)

read -r SIZE COND COUNTRY <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
BASE="outputs/behavioral/blend_${COND}_${SIZE}_nfs_usprobe.json"

echo "[probe_blend_full] size=$SIZE condition=$COND probe_country=$COUNTRY"
echo "[probe_blend_full] base=$BASE"

python evaluate/add_probe.py \
    --base "$BASE" \
    --probe-country "$COUNTRY" \
    --condition "$COND" \
    --model-size "$SIZE" \
    --sample-frac 1.0
