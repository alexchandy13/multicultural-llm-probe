#!/bin/bash
#SBATCH --job-name=cult_cb_g4
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --output=slurm/eval_culturalbench_gemma4.%A_%a.out
#SBATCH --error=slurm/eval_culturalbench_gemma4.%A_%a.err

# CulturalBench eval for Gemma 4 12B, with US probe.
# Array: 5 conditions (0-4)
# Submit with: sbatch slurm/eval_culturalbench_gemma4_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

CONDS=(base sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

echo "[culturalbench_gemma4] condition=$COND"

python evaluate/eval_culturalbench.py \
    --condition "$COND" \
    --model-size gemma4 \
    --us-probe
