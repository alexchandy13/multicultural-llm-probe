#!/bin/bash
#SBATCH --job-name=nm_raw
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=2:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/nm_raw.%A_%a.out
#SBATCH --error=slurm/nm_raw.%A_%a.err
#
# Does OLMo 3 alignment actually cost cultural knowledge, or does the gap just
# come from the prompt format changing?
#
# Base scores 0.808 at fs2, sft 0.715, dpo 0.729. But base and the aligned
# checkpoints never saw the same input: base gets one flat string, while the
# chat-templated checkpoints get the few-shot examples as separate chat turns,
# and single-token log-prob scoring is off-distribution for a chat model. So the
# 9-point gap confounds changed weights with changed prompt format.
#
# --raw-prompt bypasses the chat template, making the aligned checkpoints read a
# prompt byte-identical to base's. Then:
#   raw ~= 0.80  -> the gap was format, not knowledge
#   raw ~= 0.72  -> alignment really did cost cultural knowledge
#
# No --us-probe: `scores` is written either way, which is all AUROC needs, and
# skipping the probe halves runtime.
#
#   sbatch --array=0-3 slurm/eval_normad_olmo3_rawprompt_job.sh

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

# model_size:shots — 0 shots means plain 0-shot.
SPECS=("olmo3_sft:2" "olmo3_sft:0" "olmo3_dpo:2" "olmo3_dpo:0")
SPEC=${SPECS[$SLURM_ARRAY_TASK_ID]}
SIZE=${SPEC%%:*}
SHOTS=${SPEC##*:}

if [[ "$SHOTS" == "0" ]]; then
    OUT="outputs/behavioral/normad_base_${SIZE}_yn_raw.json"
    FS=()
else
    OUT="outputs/behavioral/normad_base_${SIZE}_fs${SHOTS}_yn_raw.json"
    FS=(--few-shot "$SHOTS")
fi

if [[ -f "$OUT" ]]; then
    echo "[nm_raw] $OUT exists — nothing to do"
    exit 0
fi

echo "[nm_raw] size=$SIZE shots=$SHOTS -> $OUT"
python evaluate/eval_normad.py \
    --condition base --model-size "$SIZE" \
    --yn-only --raw-prompt ${FS[@]+"${FS[@]}"}
