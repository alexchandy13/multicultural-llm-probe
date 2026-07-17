#!/bin/bash
#SBATCH --job-name=culture_eval_qwen35
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5
#SBATCH --output=slurm/eval_qwen35.%A_%a.out
#SBATCH --error=slurm/eval_qwen35.%A_%a.err

# Array job — eval all 6 conditions at qwen35 on CLIP A6000.
# Submit with: CONDITIONS="base sft dpo_coig dpo_pku sftdpo_coig sftdpo_pku" sbatch ...

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_qwen35] condition=$COND"

python evaluate/eval_normad.py --condition "$COND" --model-size qwen35
