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

# C4 condition: DPO trained on top of an existing SFT adapter (sequential SFT+DPO).
#
# IMPORTANT: this job depends on the SFT job — the training script loads
# checkpoints/sft/checkpoint-4878 and will fail with FileNotFoundError if
# SFT hasn't completed.
#
# Two ways to enforce ordering:
#   (1) Wait until SFT finishes, then `sbatch slurm/sftdpo_job.sh`.
#   (2) Submit both at once with SLURM dependency on the SFT job ID:
#         SFT_ID=$(sbatch --parsable slurm/sft_job.sh)
#         sbatch --dependency=afterok:$SFT_ID slurm/sftdpo_job.sh
# Option (2) is preferred — both jobs queue immediately and SLURM holds the
# C4 job until SFT exits 0.

set -euo pipefail
source env.sh

python3.12 finetune/sftdpo_train.py --config finetune/configs/sftdpo_config.yaml
