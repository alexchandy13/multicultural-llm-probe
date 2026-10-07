#!/bin/bash
#SBATCH --job-name=cb_fs4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=2:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-6
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/cb_fs4.%A_%a.out
#SBATCH --error=slurm/cb_fs4.%A_%a.err

# CulturalBench few-shot, in one of two modes. Both use gold-matched labels
# (1 yes / 3 no ~ the 27% yes rate) rather than balanced ones, since a 1:1 prefix
# signals a 50% yes-rate and pushes predictions further in the direction every
# condition already over-predicts.
#
#   MODE=nfs  (default)  N country-free neutral shots. Nothing is held out,
#                        so eval n stays 4905 and no cultural knowledge leaks.
#   MODE=fs              N real demonstrations from the New Zealand holdout.
#                        All 44 NZ examples are dropped, so n is 4861.
#
#   MODE=nfs MODEL=8b     sbatch slurm/eval_culturalbench_fs4_job.sh
#   MODE=fs  MODEL=8b     sbatch slurm/eval_culturalbench_fs4_job.sh
#   MODE=nfs MODEL=gemma4 sbatch --array=0-4 --gres=gpu:rtxa6000:1 slurm/eval_culturalbench_fs4_job.sh
#
# tulu3 conditions exist only at 8b; the array indices below put them last so
# --array=0-4 covers exactly the conditions gemma4 has.

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODEL="${MODEL:-8b}"
MODE="${MODE:-nfs}"
SHOTS="${SHOTS:-4}"
CONDS=(base sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult tulu3_sft tulu3_dpo)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

if [[ "$MODEL" != "8b" && "$COND" == tulu3_* ]]; then
    echo "[cb_$MODE] $COND exists only at 8b — nothing to do for $MODEL"
    exit 0
fi

case "$MODE" in
    nfs) FLAG=(--neutral-shots "$SHOTS") ;;
    fs)  FLAG=(--few-shot     "$SHOTS") ;;
    *)   echo "MODE must be nfs or fs, got '$MODE'" >&2; exit 1 ;;
esac

echo "[cb_$MODE] condition=$COND model=$MODEL shots=$SHOTS"
python evaluate/eval_culturalbench.py \
    --condition "$COND" --model-size "$MODEL" "${FLAG[@]}"
