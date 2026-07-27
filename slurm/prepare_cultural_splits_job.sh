#!/bin/bash
#SBATCH --job-name=prepare_cultural_splits
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm/prepare_cultural_splits.%j.out
#SBATCH --error=slurm/prepare_cultural_splits.%j.err

# Streams CultureMarkers (5.6M rows) and joins with Aya + UltraFeedback to produce
# four training splits: data/aya_cult, data/aya_nocult, data/uf_cult, data/uf_nocult
# No GPU needed — CPU-only data processing.
# Submit with: sbatch slurm/prepare_cultural_splits_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python data/prepare_cultural_splits.py --out-dir data
