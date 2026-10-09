#!/bin/bash
#SBATCH --job-name=nlu_sys
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/nlu_sys.%A_%a.out
#SBATCH --error=slurm/nlu_sys.%A_%a.err
#
# Re-run OLMo 3 general NLU with an explicit neutral system prompt.
#
# The existing nlu_*_olmo3_{sft,dpo}* files were produced before --system-prompt
# existed, so they carry Olmo-3-7B-Instruct-*'s chat-template default:
#   "You are a helpful function-calling AI assistant."
#   "You do not currently have access to any functions."
# The base checkpoint has no chat template and never saw that text, so the NLU
# control comparison is between models given different instructions.
#
# Only sft and dpo are re-run; --system-prompt is a no-op on olmo3 base, whose
# existing files are already the neutral case.
#
# On NormAd this was worth under a point (0.715 -> 0.707 sft, 0.729 -> 0.727 dpo),
# so expect a small effect here too — but the NLU result is load-bearing for the
# claim that the loss is cultural rather than general, so it should not rest on an
# uncontrolled prompt difference.
#
# Both prefix settings, to match what already exists:
#   0-shot -> nlu_{ds}_base_{size}_sys.json
#   nfs    -> nlu_{ds}_base_{size}_nfs_sys.json
#
#   sbatch --array=0-1 slurm/eval_nlu_olmo3_sysprompt_job.sh

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra SIZES <<< "${SIZES:-olmo3_sft olmo3_dpo}"
SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}
read -ra DATASETS <<< "${DATASETS:-boolq csqa qnli mrpc}"

echo "[nlu_sys] model_size=$SIZE datasets=${DATASETS[*]}"
for DS in "${DATASETS[@]}"; do
    for MODE in zeroshot nfs; do
        if [[ "$MODE" == "nfs" ]]; then
            OUT="outputs/behavioral/nlu_${DS}_base_${SIZE}_nfs_sys.json"
            EXTRA=(--neutral-fewshot)
        else
            OUT="outputs/behavioral/nlu_${DS}_base_${SIZE}_sys.json"
            EXTRA=()
        fi
        if [[ -f "$OUT" ]]; then
            echo "--- $DS/$MODE: exists, skipping ---"
            continue
        fi
        echo "--- $DS/$MODE ---"
        python evaluate/eval_nlu.py --condition base --model-size "$SIZE" \
            --dataset "$DS" --system-prompt neutral ${EXTRA[@]+"${EXTRA[@]}"}
    done
done
