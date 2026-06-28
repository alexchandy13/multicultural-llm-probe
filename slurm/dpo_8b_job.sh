#!/bin/bash
#SBATCH --job-name=culture_dpo_8b
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --nodelist=tron46,tron47,tron48
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/dpo_8b.%j.out
#SBATCH --error=slurm/dpo_8b.%j.err

# DPO on HH-RLHF for Llama 3.1 8B (C3 at 8B). DPO is forward-twice (policy +
# ref) so memory and time at 8B are heavier than SFT. 48G RAM, 24h time.
#
# See sft_8b_job.sh for why --nodelist=tron47 (python3.12 only on that A5000).

set -euo pipefail
source env.sh

python3.12 finetune/dpo_train.py --config finetune/configs/dpo_8b_config.yaml
