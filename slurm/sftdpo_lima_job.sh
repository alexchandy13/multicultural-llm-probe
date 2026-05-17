#!/bin/bash
#SBATCH --job-name=culture_sftdpo_lima
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/sftdpo_lima.%j.out
#SBATCH --error=slurm/sftdpo_lima.%j.err

# C4b — DPO on top of an existing LIMA SFT adapter.
#
# Depends on slurm/sft_lima_job.sh finishing first (loads checkpoints/sft_lima).
# Use SLURM --dependency=afterok:<sft_lima_jobid> when submitting both at once:
#   SFT_LIMA_ID=$(sbatch --parsable slurm/sft_lima_job.sh)
#   sbatch --dependency=afterok:$SFT_LIMA_ID slurm/sftdpo_lima_job.sh
#
# Explicit --sft-adapter-path makes the override visible alongside the config's
# init_adapter_path (which already points at checkpoints/sft_lima).

set -euo pipefail
source env.sh

python3.12 finetune/sftdpo_train.py \
    --config finetune/configs/sftdpo_lima_config.yaml \
    --sft-adapter-path checkpoints/sft_lima
