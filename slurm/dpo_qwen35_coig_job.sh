#!/bin/bash
#SBATCH --job-name=culture_dpo_qwen35_coig
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=20:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/dpo_qwen35_coig.%j.out
#SBATCH --error=slurm/dpo_qwen35_coig.%j.err

# DPO on COIG-P (chat domain) for Qwen3.5 9B Base (dpo_coig condition).

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python finetune/dpo_train.py \
    --config finetune/configs/dpo_qwen35_config.yaml \
    --dataset-name m-a-p/COIG-P \
    --dataset-path data/coig-p \
    --output-dir checkpoints/dpo_coig_qwen35
