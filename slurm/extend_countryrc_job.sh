#!/bin/bash
#SBATCH --job-name=extend_countryrc
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --output=slurm/extend_countryrc.%A_%a.out
#SBATCH --error=slurm/extend_countryrc.%A_%a.err

# Extend countryrc_max_scores.json for all 10 condition dirs (5 conds × 2 sizes).
# Adds Germany, Japan, India, Russia, Brazil, Zimbabwe (normad/culturalbench coverage)
# plus the blend-only countries. Already-scored countries are skipped automatically.
# Array maps: 0-4 = 8b conditions, 5-9 = gemma4 conditions (same 5 CONDITIONS order).

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"

if [[ $SLURM_ARRAY_TASK_ID -lt 5 ]]; then
    MODEL_SIZE="8b"
    COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
else
    MODEL_SIZE="gemma4"
    COND=${CONDS[$((SLURM_ARRAY_TASK_ID - 5))]}
fi

echo "[extend_countryrc] condition=$COND model_size=$MODEL_SIZE"

python scripts/extend_countryrc.py \
    --condition "$COND" \
    --model-size "$MODEL_SIZE" \
    --precision matched_bf16 \
    --all-extra
