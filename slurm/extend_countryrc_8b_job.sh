#!/bin/bash
#SBATCH --job-name=extend_crc_8b
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/extend_countryrc_8b.%A_%a.out
#SBATCH --error=slurm/extend_countryrc_8b.%A_%a.err

# legacygpu nodes excluded: their GPUs predate the compute capability our
# torch build ships kernels for (cudaErrorNoKernelImageForDevice on load).

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[extend_countryrc_8b] condition=$COND"

python scripts/extend_countryrc.py \
    --condition "$COND" \
    --model-size 8b \
    --precision matched_bf16 \
    --all-extra
