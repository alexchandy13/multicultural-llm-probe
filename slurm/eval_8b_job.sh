#!/bin/bash
#SBATCH --job-name=culture_eval_8b
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --exclude=tron[49-61]
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_8b.%A_%a.out
#SBATCH --error=slurm/eval_8b.%A_%a.err

# See sft_8b_job.sh for why --nodelist=tron47 (only A5000 with python3.12).

# Array job — eval all 4 conditions at 8B. Indices map to CONDITIONS in env.sh.
# Outputs go to outputs/behavioral/normad_{condition}_8b.json (size suffix
# applied by eval_normad.py when --model-size != 3b).
#
# --precision matched_bf16 is the default; pass --precision qlora_4bit to
# reproduce the legacy precision regime for back-compat checks.

set -euo pipefail
source env.sh

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_8b] condition=$COND"

python3.12 evaluate/eval_normad.py --condition "$COND" --model-size 8b
