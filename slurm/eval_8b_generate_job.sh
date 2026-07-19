#!/bin/bash
#SBATCH --job-name=culture_eval_8b_gen
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_8b_gen.%A_%a.out
#SBATCH --error=slurm/eval_8b_gen.%A_%a.err

# Generation-based eval for Llama 3.1 8B. Greedy-decodes up to 10 tokens
# and parses yes/no/neutral from the output. Unparseable responses are counted
# as wrong (not silently assigned to neutral like NormAd's parser does).
# Submit with: CONDITIONS="base sft dpo sftdpo" sbatch slurm/eval_8b_generate_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_8b_gen] condition=$COND"

# 0-shot generation
python evaluate/eval_normad.py --condition "$COND" --model-size 8b --generate

# few-shot + generation (recommended: few-shot gives base model the format)
python evaluate/eval_normad.py --condition "$COND" --model-size 8b --few-shot 3 --generate
