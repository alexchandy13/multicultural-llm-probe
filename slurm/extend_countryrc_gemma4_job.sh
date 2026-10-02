#!/bin/bash
#SBATCH --job-name=extend_crc_g4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/extend_countryrc_gemma4.%A_%a.out
#SBATCH --error=slurm/extend_countryrc_gemma4.%A_%a.err

# Always name a GPU type, never a bare count: scavenger spans 11 GB (rtx2080ti)
# to 141 GB cards, and device_map="auto" answers an undersized allocation by
# placing the model on CPU, which runs ~40x too slow to finish. Gemma4-12B needs
# ~46 GB for the backward pass, so 2x48 GB rtxa6000.
# legacygpu nodes excluded: their GPUs predate the compute capability our
# torch build ships kernels for (cudaErrorNoKernelImageForDevice on load).

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[extend_countryrc_gemma4] condition=$COND"

python scripts/extend_countryrc.py \
    --condition "$COND" \
    --model-size gemma4 \
    --precision matched_bf16 \
    --all-extra
