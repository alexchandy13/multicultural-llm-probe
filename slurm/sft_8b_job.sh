#!/bin/bash
#SBATCH --job-name=culture_sft_8b
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/sft_8b.%j.out
#SBATCH --error=slurm/sft_8b.%j.err

# SFT on Alpaca for Llama 3.1 8B (C2 at 8B). Mirrors sft_job.sh but on the 8B
# config. Memory bumped to 48G and wall-time bumped to 24h to account for the
# ~3-4× slower per-step time at 8B vs 3B.

set -euo pipefail
source env.sh

python3.12 finetune/sft_train.py --config finetune/configs/sft_8b_config.yaml
