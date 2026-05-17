#!/bin/bash
#SBATCH --job-name=culture_sft_lima
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/sft_lima.%j.out
#SBATCH --error=slurm/sft_lima.%j.err

# C2b — SFT on LIMA (second robustness variant, 1000 examples × 15 epochs).
#
# Fast: ~30-60 min wall clock on A5000 (1000 examples / batch 32 = 31 steps/epoch
# × 15 epochs = ~470 optimizer steps; well under a 2-hour cap).

set -euo pipefail
source env.sh

python3.12 finetune/sft_train.py --config finetune/configs/sft_lima_config.yaml
