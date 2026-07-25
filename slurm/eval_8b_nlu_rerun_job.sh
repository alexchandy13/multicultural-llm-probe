#!/bin/bash
#SBATCH --job-name=culture_eval_8b_nlu_rerun
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_8b_nlu_rerun.%A_%a.out
#SBATCH --error=slurm/eval_8b_nlu_rerun.%A_%a.err

# Re-run QNLI and MRPC for 8B after switching to yes/no scoring.
# Submit with: CONDITIONS="base sft dpo sftdpo" sbatch slurm/eval_8b_nlu_rerun_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_8b_nlu_rerun] condition=$COND"

for DATASET in qnli mrpc; do
    echo "=== 0-shot $DATASET ==="
    python evaluate/eval_nlu.py --condition "$COND" --model-size 8b --dataset "$DATASET"

    echo "=== nfs $DATASET ==="
    python evaluate/eval_nlu.py --condition "$COND" --model-size 8b --dataset "$DATASET" --neutral-fewshot
done
