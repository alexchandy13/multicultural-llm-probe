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

# CulturalBench with 4 real few-shot demonstrations drawn from the New Zealand
# holdout. Labels are gold-matched (1 yes / 3 no ~ the 27% yes rate) rather than
# balanced, and all 44 New Zealand examples are excluded so no evaluated item
# concerns a demonstrated country. Eval n drops 4905 -> 4861.
#
#   MODEL=8b     sbatch slurm/eval_culturalbench_fs4_job.sh
#   MODEL=gemma4 sbatch --array=0-4 --gres=gpu:rtxa6000:1 slurm/eval_culturalbench_fs4_job.sh
#
# tulu3 conditions exist only at 8b; the array indices below put them last so
# --array=0-4 covers exactly the conditions gemma4 has.

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODEL="${MODEL:-8b}"
SHOTS="${SHOTS:-4}"
CONDS=(base sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult tulu3_sft tulu3_dpo)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

if [[ "$MODEL" != "8b" && "$COND" == tulu3_* ]]; then
    echo "[cb_fs4] $COND exists only at 8b — nothing to do for $MODEL"
    exit 0
fi

echo "[cb_fs4] condition=$COND model=$MODEL shots=$SHOTS"
python evaluate/eval_culturalbench.py \
    --condition "$COND" --model-size "$MODEL" --few-shot "$SHOTS"
