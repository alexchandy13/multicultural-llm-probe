#!/bin/bash
#SBATCH --job-name=culture_eval_8b_fs
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_8b_fs.%A_%a.out
#SBATCH --error=slurm/eval_8b_fs.%A_%a.err

# Few-shot eval for Llama 3.1 8B — tests whether base model's poor 0-shot
# performance is a format artifact (if base+few-shot ≈ sft+0-shot, it was).
# Submit with: CONDITIONS="base sft dpo sftdpo" sbatch slurm/eval_8b_fewshot_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_8b_fs] condition=$COND"

python evaluate/eval_normad.py --condition "$COND" --model-size 8b --few-shot 3
python evaluate/eval_normad.py --condition "$COND" --model-size 8b --few-shot 3 --calibrate
