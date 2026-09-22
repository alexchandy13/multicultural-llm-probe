#!/bin/bash
#SBATCH --job-name=normy_8b
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa4000:2
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --output=slurm/eval_culturalbench_normy_8b.%A_%a.out
#SBATCH --error=slurm/eval_culturalbench_normy_8b.%A_%a.err

# CulturalBench-Normy eval for LLaMA 3.1 8B, with US probe.
# Array: 5 conditions (0-4)
# Submit with: sbatch slurm/eval_culturalbench_normy_8b_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

CONDS=(base sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

echo "[culturalbench_normy_8b] condition=$COND"

python evaluate/eval_culturalbench_normy.py \
    --condition "$COND" \
    --model-size 8b \
    --us-probe
