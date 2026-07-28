#!/bin/bash
#SBATCH --job-name=culture_eval_aya_cult_nlu
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_aya_cult_nlu.%A_%a.out
#SBATCH --error=slurm/eval_aya_cult_nlu.%A_%a.err

# NLU eval (BoolQ, CSQA, QNLI, MRPC) for Aya cultural split conditions, Llama 3.1 8B.
# Requires sft_aya_cult_8b, sft_aya_nocult_8b, sftdpo_aya_cult_8b, sftdpo_aya_nocult_8b
# checkpoints to exist.
# Submit with: sbatch slurm/eval_aya_cult_nlu_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

CONDS=(sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_aya_cult_nlu] condition=$COND"

for DATASET in boolq csqa qnli mrpc; do
    echo "=== 0-shot $DATASET ==="
    python evaluate/eval_nlu.py --condition "$COND" --model-size 8b --dataset "$DATASET"

    echo "=== nfs $DATASET ==="
    python evaluate/eval_nlu.py --condition "$COND" --model-size 8b --dataset "$DATASET" --neutral-fewshot
done
