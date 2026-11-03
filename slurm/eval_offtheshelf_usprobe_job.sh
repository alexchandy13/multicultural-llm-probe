#!/bin/bash
#SBATCH --job-name=ots_usprobe
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=3:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --output=slurm/eval_ots_usprobe.%A_%a.out
#SBATCH --error=slurm/eval_ots_usprobe.%A_%a.err

# Own-country + US-swap pass for Llama-3.1-8B-Instruct, raw-prompted.
#
# Step 1 of 2 for the instruct probe grid: add_probe.py reads `pred` (the
# own-country answer) out of a _usprobe.json and runs only the swap pass, so
# all 7 remaining probe countries need this file first.
#
# --raw-prompt bypasses the chat template, which is what puts this arm on the
# same footing as everything else in the results table -- sftdpo_aya_* and
# tulu3_sft/tulu3_dpo are all scored as flat strings, because is_instruct()
# keys off the size label and they run at model_size=8b. Outputs gain _raw.
#
# BLEnD is included even though blend_base_8b_instruct_nfs_usprobe.json exists:
# that file was chat-templated, so it is not a valid base for a raw-prompted
# probe pass. The raw run writes _usprobe_raw and the two coexist.
#
# Each benchmark skips and exits 0 if its output is already there, so a requeue
# after scavenger preemption costs one model load.
#
# 3 tasks. NormAd 1,818 examples, CulturalBench 4,905, BLEnD 23,032 -- two
# passes each, so BLEnD sets the walltime.
#
# Submit with: sbatch slurm/eval_offtheshelf_usprobe_job.sh
# Then feed the returned job id to the probe grid:
#   sbatch --dependency=afterok:<jobid> slurm/eval_offtheshelf_probe_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODEL=8b_instruct
BENCHES=(normad culturalbench blend)
BENCH=${BENCHES[$SLURM_ARRAY_TASK_ID]}

echo "[ots_usprobe] benchmark=$BENCH condition=base model=$MODEL raw=yes"

case "$BENCH" in
  normad)
    OUT="outputs/behavioral/normad_base_${MODEL}_fs2_yn_usprobe_raw.json"
    if [[ -f "$OUT" ]]; then echo "[ots_usprobe] $OUT exists — nothing to do"; exit 0; fi
    python evaluate/eval_normad.py \
        --condition base --model-size "$MODEL" --precision matched_bf16 \
        --few-shot 2 --yn-only --us-probe --raw-prompt
    ;;
  culturalbench)
    OUT="outputs/behavioral/culturalbench_base_${MODEL}_nfs4_usprobe_raw.json"
    if [[ -f "$OUT" ]]; then echo "[ots_usprobe] $OUT exists — nothing to do"; exit 0; fi
    python evaluate/eval_culturalbench.py \
        --condition base --model-size "$MODEL" --precision matched_bf16 \
        --neutral-shots 4 --us-probe --raw-prompt
    ;;
  blend)
    OUT="outputs/behavioral/blend_base_${MODEL}_nfs_usprobe_raw.json"
    if [[ -f "$OUT" ]]; then echo "[ots_usprobe] $OUT exists — nothing to do"; exit 0; fi
    python evaluate/eval_blend.py \
        --condition base --model-size "$MODEL" --precision matched_bf16 \
        --neutral-fewshot --us-probe --raw-prompt
    ;;
esac
