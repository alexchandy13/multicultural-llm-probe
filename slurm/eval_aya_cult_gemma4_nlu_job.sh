#!/bin/bash
#SBATCH --job-name=eval_aya_cult_gemma4_nlu
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_aya_cult_gemma4_nlu.%A_%a.out
#SBATCH --error=slurm/eval_aya_cult_gemma4_nlu.%A_%a.err

# NLU eval (BoolQ, CSQA, QNLI, MRPC) for Aya cultural split conditions, Gemma 4 12B.
# Requires sft_aya_cult_gemma4, sft_aya_nocult_gemma4, sftdpo_aya_cult_gemma4,
# sftdpo_aya_nocult_gemma4 checkpoints to exist.
# Submit with: sbatch slurm/eval_aya_cult_gemma4_nlu_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

CONDS=(sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_aya_cult_gemma4_nlu] condition=$COND"

for DATASET in boolq csqa qnli mrpc; do
    echo "=== 0-shot $DATASET ==="
    python evaluate/eval_nlu.py --condition "$COND" --model-size gemma4 --dataset "$DATASET"

    echo "=== nfs $DATASET ==="
    python evaluate/eval_nlu.py --condition "$COND" --model-size gemma4 --dataset "$DATASET" --neutral-fewshot
done
