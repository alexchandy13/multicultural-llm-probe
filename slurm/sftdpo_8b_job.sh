#!/bin/bash
#SBATCH --job-name=culture_sftdpo_8b
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/sftdpo_8b.%j.out
#SBATCH --error=slurm/sftdpo_8b.%j.err

# C4 at 8B — DPO on top of the merged 8B SFT adapter (Llama 3.1 8B).
#
# Depends on checkpoints/sft_8b/ from sft_8b_job.sh. Submit with:
#   SFT_ID=$(sbatch --parsable slurm/sft_8b_job.sh)
#   sbatch --dependency=afterok:$SFT_ID slurm/sftdpo_8b_job.sh
#
# This loads the base in bf16 (merge_and_unload incompatibility with 4-bit),
# so memory is higher than dpo_8b — bumped to 64G to be safe.

set -euo pipefail
source env.sh

python3.12 finetune/sftdpo_train.py \
    --config finetune/configs/sftdpo_8b_config.yaml \
    --sft-adapter-path checkpoints/sft_8b
