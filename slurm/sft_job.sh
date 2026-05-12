#!/bin/bash
#SBATCH --job-name=culture_sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm/sft.%j.out
#SBATCH --error=slurm/sft.%j.err

set -euo pipefail
source env.sh

python finetune/sft_train.py --config finetune/configs/sft_config.yaml
