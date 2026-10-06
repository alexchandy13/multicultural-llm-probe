#!/bin/bash
#SBATCH --job-name=eval_bare
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=8:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/eval_bare.%A_%a.out
#SBATCH --error=slurm/eval_bare.%A_%a.err

# Bare-bones eval: no few-shot prefix, no option shuffling, no US probe, no
# calibration. All three benchmarks, one condition per array task.
#
#   MODEL=8b     sbatch slurm/eval_barebones_job.sh              # all 5 conditions
#   MODEL=gemma4 sbatch --array=0-2 slurm/eval_barebones_job.sh  # tulu3 is 8b-only
#
# gemma4 needs 2x A6000 (~46 GB); override --gres for it.
#
# NormAd keeps --yn-only: the 3-way task is degenerate, models never emit
# "neutral" at all, so it scores ~36% which is just "always yes".
# CulturalBench needs --no-fewshot because its neutral prefix is otherwise
# applied unconditionally.

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODEL="${MODEL:-8b}"
CONDS=(base sft_aya_nocult sftdpo_aya_nocult tulu3_sft tulu3_dpo)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

if [[ "$MODEL" != "8b" && "$COND" == tulu3_* ]]; then
    echo "[eval_bare] $COND exists only at 8b — nothing to do for $MODEL"
    exit 0
fi

echo "[eval_bare] condition=$COND model=$MODEL"

python evaluate/eval_normad.py        --condition "$COND" --model-size "$MODEL" --yn-only
python evaluate/eval_blend.py         --condition "$COND" --model-size "$MODEL"
python evaluate/eval_culturalbench.py --condition "$COND" --model-size "$MODEL" --no-fewshot
