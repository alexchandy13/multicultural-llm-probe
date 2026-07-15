#!/bin/bash
#SBATCH --job-name=culture_sftdpo_gemma4
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/sftdpo_gemma4.%j.out
#SBATCH --error=slurm/sftdpo_gemma4.%j.err

# C4 at gemma4 — DPO on top of the merged gemma4 SFT adapter (Gemma 4 12B).
#
# Depends on checkpoints/sft_gemma4/ from sft_gemma4_job.sh. Submit with:
#   SFT_ID=$(sbatch --parsable slurm/sft_gemma4_job.sh)
#   sbatch --dependency=afterok:$SFT_ID slurm/sftdpo_gemma4_job.sh
#
# Runs on CLIP A6000 (48GB).

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python finetune/sftdpo_train.py \
    --config finetune/configs/sftdpo_gemma4_config.yaml \
    --sft-adapter-path checkpoints/sft_gemma4
