#!/bin/bash
#SBATCH --job-name=culture_eval_8b_usprobe
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_8b_usprobe.%A_%a.out
#SBATCH --error=slurm/eval_8b_usprobe.%A_%a.err

# US-probe eval for Llama 3.1 8B. For each non-US example, also scores the
# same prompt with country replaced by "United States". Roughly 2x eval time.
# Outputs: normad_{cond}_8b_yn_usprobe.json and normad_{cond}_8b_fs2_yn_usprobe.json
# Submit with: CONDITIONS="base sft dpo sftdpo" sbatch slurm/eval_8b_usprobe_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_8b_usprobe] condition=$COND"

python evaluate/eval_normad.py --condition "$COND" --model-size 8b --yn-only --us-probe
python evaluate/eval_normad.py --condition "$COND" --model-size 8b --yn-only --few-shot 2 --us-probe
