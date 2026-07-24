#!/bin/bash
#SBATCH --job-name=culture_eval_8b_nlu
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_8b_nlu.%A_%a.out
#SBATCH --error=slurm/eval_8b_nlu.%A_%a.err

# General NLU eval for Llama 3.1 8B.
# Runs all 4 datasets in 0-shot and neutral-fewshot modes.
# Outputs: nlu_{dataset}_{cond}_8b.json
#          nlu_{dataset}_{cond}_8b_nfs.json
# Submit with: CONDITIONS="base sft dpo sftdpo" sbatch slurm/eval_8b_nlu_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_8b_nlu] condition=$COND"

for DATASET in boolq csqa qnli mrpc; do
    echo "=== 0-shot $DATASET ==="
    python evaluate/eval_nlu.py --condition "$COND" --model-size 8b --dataset "$DATASET"

    echo "=== nfs $DATASET ==="
    python evaluate/eval_nlu.py --condition "$COND" --model-size 8b --dataset "$DATASET" --neutral-fewshot
done
