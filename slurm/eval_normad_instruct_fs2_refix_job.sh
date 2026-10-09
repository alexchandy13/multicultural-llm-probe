#!/bin/bash
#SBATCH --job-name=nm_ifix
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=8:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/nm_ifix.%A_%a.out
#SBATCH --error=slurm/nm_ifix.%A_%a.err
#
# Re-run NormAd few-shot for the chat-templated checkpoints after fixing a bug
# that made --few-shot N a silent no-op on them.
#
# eval_normad.py builds the few-shot context two ways: a raw string `prefix` for
# base models, and `fewshot_turns` (chat turns) for instruct models. The
# --neutral-fewshot branch set both; the --few-shot N branch set only `prefix`,
# which the instruct code path never reads. So every chat-templated checkpoint
# run with --few-shot N was actually evaluated 0-shot. Caught because
# normad_base_olmo3_sft_fs2_yn_usprobe.json came back md5-identical to the
# 0-shot file, and the prefix builder is seeded and model-independent, so equal
# output is only possible if the prefix was dropped.
#
# The aya conditions are unaffected: they run at model_size 8b / gemma4, which
# are not in CHAT_TEMPLATED_SIZES, so they took the prefix path all along.
# CulturalBench and NLU already set fewshot_turns in both branches; BLEnD had
# the same bug in its --few-shot branch but we only ever ran it --neutral-fewshot.
#
# The stale files are moved to *.prefixbug.bak rather than deleted, so the old
# numbers stay auditable.
#
#   sbatch --array=0-5 slurm/eval_normad_instruct_fs2_refix_job.sh

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

# model_size:shots — 8b_instruct was run at fs3, the rest at fs2.
SPECS=(
    "olmo3_sft:2"
    "olmo3_dpo:2"
    "olmoe_sft:2"
    "olmoe_instruct:2"
    "gemma4_moe_instruct:2"
    "8b_instruct:3"
)
SPEC=${SPECS[$SLURM_ARRAY_TASK_ID]}
SIZE=${SPEC%%:*}
SHOTS=${SPEC##*:}

# 8b_instruct's stale file is the bare fs3 variant (no --yn-only / --us-probe).
if [[ "$SIZE" == "8b_instruct" ]]; then
    OUT="outputs/behavioral/normad_base_${SIZE}_fs${SHOTS}.json"
    EXTRA=()
else
    OUT="outputs/behavioral/normad_base_${SIZE}_fs${SHOTS}_yn_usprobe.json"
    EXTRA=(--yn-only --us-probe)
fi

if [[ -f "$OUT" && ! -f "$OUT.prefixbug.bak" ]]; then
    mv "$OUT" "$OUT.prefixbug.bak"
    echo "[nm_ifix] stale 0-shot result parked at $OUT.prefixbug.bak"
fi

echo "[nm_ifix] size=$SIZE shots=$SHOTS -> $OUT"
python evaluate/eval_normad.py \
    --condition base --model-size "$SIZE" \
    --few-shot "$SHOTS" "${EXTRA[@]}"
