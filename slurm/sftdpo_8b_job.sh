#!/bin/bash
#SBATCH --job-name=culture_sftdpo_8b
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --nodelist=tron47
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
# MEMORY-TIGHT WARNING:
# This loads the base in bf16 (not 4-bit) because PeftModel.merge_and_unload()
# doesn't compose with bitsandbytes 4-bit. At 8B bf16 base ≈ 16 GB; add
# activations + DPO LoRA + ref-model path → peak ~18-22 GB. A5000 has 24 GB
# so this is tight but should fit. We've already dropped max_length 512 → 384
# in sftdpo_8b_config.yaml to give ~2 GB headroom. If it still OOMs, drop
# again to 256.
# Class partition doesn't expose A6000 (which would have had 48 GB).
#
# --nodelist=tron47: same OS-image reason as the other 8B jobs.

set -euo pipefail
source env.sh

python3.12 finetune/sftdpo_train.py \
    --config finetune/configs/sftdpo_8b_config.yaml \
    --sft-adapter-path checkpoints/sft_8b
