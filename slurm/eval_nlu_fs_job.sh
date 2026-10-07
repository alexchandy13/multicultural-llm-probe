#!/bin/bash
#SBATCH --job-name=nlu_fs
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=4:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-6
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/nlu_fs.%A_%a.out
#SBATCH --error=slurm/nlu_fs.%A_%a.err

# QNLI and MRPC with real demonstrations drawn from each dataset's train split.
#
# The hand-written neutral shots teach the wrong decision boundary: a synthetic
# MRPC "no" is two unrelated sentences, while a real one differs by a single
# clause, and a real MRPC "yes" can disagree on a number and still count as a
# paraphrase. That mismatch — not the label ratio — is why nfs drove tulu3_dpo
# from -3.4 skew at 0-shot to -31, and why the gold-matched nfs4 prefix (3y/1n)
# left it unchanged at 63.0%.
#
# Train and validation are disjoint GLUE splits, so nothing is held out and the
# label ratio matches validation for free (qnli 50.0 vs 49.5% yes, mrpc 67.4 vs
# 68.4%).
#
#   MODEL=8b           sbatch slurm/eval_nlu_fs_job.sh
#   MODEL=8b   SHOTS=8 sbatch slurm/eval_nlu_fs_job.sh
#   MODEL=gemma4       sbatch --array=0-4 slurm/eval_nlu_fs_job.sh   # tulu3 is 8b-only

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODEL="${MODEL:-8b}"
SHOTS="${SHOTS:-4}"
read -ra DATASETS <<< "${DATASETS:-qnli mrpc}"
read -ra CONDS <<< "${CONDS:-base sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult tulu3_sft tulu3_dpo}"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

if [[ "$MODEL" != "8b" && "$COND" == tulu3_* ]]; then
    echo "[nlu_fs] $COND exists only at 8b — nothing to do for $MODEL"
    exit 0
fi

for DS in "${DATASETS[@]}"; do
    echo "[nlu_fs] condition=$COND model=$MODEL dataset=$DS shots=$SHOTS"
    python evaluate/eval_nlu.py \
        --condition "$COND" --model-size "$MODEL" --dataset "$DS" \
        --few-shot "$SHOTS"
done
