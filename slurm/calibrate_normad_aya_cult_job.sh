#!/bin/bash
#SBATCH --job-name=calibrate_normad_aya_cult
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=default
#SBATCH --time=2:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/calibrate_normad_aya_cult.out
#SBATCH --error=slurm/calibrate_normad_aya_cult.err

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

for COND in sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult; do
    echo "=== calibrating $COND ==="
    python scripts/calibrate_normad.py \
        --input outputs/behavioral/normad_${COND}_8b_nfs_mpw_usprobe.json \
        --condition "$COND" --model-size 8b
done
