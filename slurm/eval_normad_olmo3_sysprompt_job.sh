#!/bin/bash
#SBATCH --job-name=nm_sys
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=2:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/nm_sys.%A_%a.out
#SBATCH --error=slurm/nm_sys.%A_%a.err
#
# How much of the OLMo 3 base-vs-aligned gap is the injected system prompt?
#
# build_chat_prompt passed no system message, so Olmo-3-7B-Instruct-*'s chat
# template inserted its own default:
#
#   "You are a helpful function-calling AI assistant."
#   "You do not currently have access to any functions."
#
# That sat in front of every NormAd question for olmo3_sft and olmo3_dpo, while
# base — which has no chat template — saw nothing. So the measured gap (fs2 AUROC
# 0.892 base vs 0.825/0.817 aligned) is confounded with the difference between
# "answer this question" and "you are a function-calling assistant with no
# functions, answer this question".
#
# --system-prompt neutral replaces it with "You are a helpful assistant." Compare
# against the existing *_fs2_yn_usprobe.json files, which are identical in every
# other respect.
#
#   recovers to ~0.89  -> the gap was the system prompt
#   stays at   ~0.82   -> the gap is real, and this rules out the last confound
#
#   sbatch --array=0-1 slurm/eval_normad_olmo3_sysprompt_job.sh

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra SIZES <<< "${SIZES:-olmo3_sft olmo3_dpo}"
SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}

OUT="outputs/behavioral/normad_base_${SIZE}_fs2_yn_usprobe_sys.json"
if [[ -f "$OUT" ]]; then
    echo "[nm_sys] $OUT exists — nothing to do"
    exit 0
fi

echo "[nm_sys] size=$SIZE -> $OUT"
python evaluate/eval_normad.py \
    --condition base --model-size "$SIZE" \
    --few-shot 2 --yn-only --us-probe --system-prompt neutral
