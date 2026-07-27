#!/bin/bash
#SBATCH --job-name=culture_dpo_nocult_8b
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/dpo_nocult_8b.%j.out
#SBATCH --error=slurm/dpo_nocult_8b.%j.err

# DPO on non-cultural preference pairs (UF + PRISM combined), Llama 3.1 8B.
# Matched control for dpo_cult_8b. Plain DPO on base — no SFT first.
# Submit with: sbatch slurm/dpo_nocult_8b_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python finetune/dpo_train.py --config finetune/configs/dpo_uf_nocult_8b_config.yaml
