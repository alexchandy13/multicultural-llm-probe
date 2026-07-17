#!/bin/bash
#SBATCH --job-name=culture_eval_gemma4
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_gemma4.%A_%a.out
#SBATCH --error=slurm/eval_gemma4.%A_%a.err

# Array job — eval all 4 conditions at gemma4 on CLIP A6000 (48GB).

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_gemma4] condition=$COND"

python evaluate/eval_normad.py --condition "$COND" --model-size gemma4
python evaluate/eval_normad.py --condition "$COND" --model-size gemma4 --calibrate
