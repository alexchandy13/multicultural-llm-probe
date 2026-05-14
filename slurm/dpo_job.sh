#!/bin/bash
#SBATCH --job-name=culture_dpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm/dpo.%j.out
#SBATCH --error=slurm/dpo.%j.err

set -euo pipefail
source env.sh

python3.12 finetune/dpo_train.py --config finetune/configs/dpo_config.yaml
