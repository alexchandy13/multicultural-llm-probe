#!/bin/bash
#SBATCH --job-name=culture_sftdpo_qwen35_pku
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/sftdpo_qwen35_pku.%j.out
#SBATCH --error=slurm/sftdpo_qwen35_pku.%j.err

# SFT+DPO on PKU-SafeRLHF-single-dimension for Qwen3.5 9B Base (sftdpo_pku condition).
# Depends on checkpoints/sft_qwen35. Submit with:
#   SFT_ID=$(sbatch --parsable slurm/sft_qwen35_job.sh)
#   sbatch --dependency=afterok:$SFT_ID slurm/sftdpo_qwen35_pku_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python finetune/sftdpo_train.py \
    --config finetune/configs/sftdpo_qwen35_config.yaml \
    --sft-adapter-path checkpoints/sft_qwen35 \
    --dataset-name PKU-Alignment/PKU-SafeRLHF-single-dimension \
    --dataset-path data/pku-safe-rlhf-single \
    --output-dir checkpoints/sftdpo_pku_qwen35
