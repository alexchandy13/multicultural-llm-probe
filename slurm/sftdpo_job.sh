#!/bin/bash
#SBATCH --job-name=culture_sftdpo
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/sftdpo.%j.out
#SBATCH --error=slurm/sftdpo.%j.err

# C4a — DPO on top of an existing Alpaca SFT adapter (robustness variant for C4).
#
# Depends on the Alpaca SFT checkpoint at checkpoints/sft/ being present
# (it was transferred from a prior local run). No SLURM dependency required —
# Alpaca SFT is not being trained on Nexus as part of this pipeline.
#
# Explicit --sft-adapter-path makes the override visible alongside the config's
# init_adapter_path (which already points at checkpoints/sft).

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python finetune/sftdpo_train.py \
    --config finetune/configs/sftdpo_config.yaml \
    --sft-adapter-path checkpoints/sft_3b
