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

# Always name a GPU type, never a bare count: scavenger spans 11 GB (rtx2080ti)
# to 141 GB cards, and device_map="auto" answers an undersized allocation by
# placing the model on CPU, which runs ~40x too slow to finish. 8b bf16 weights
# are ~16 GB, but retained activations and grads across 5 modules x 32 layers
# push the `base` condition past 24 GB: it OOMs in the backward pass on the same
# 112 MiB allocation with the same 63 MiB free on every 24 GB card (verified
# byte-identical on two separate rtxa5000 nodes), while the adapter-merged
# conditions fit. Deterministic, so 48 GB is required rather than just safer.
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
