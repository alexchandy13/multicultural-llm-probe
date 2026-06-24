#!/bin/bash
#SBATCH --job-name=culture_sftdpo_8b
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
# TODO: set --nodelist=<a-known-good-A6000-node> after running:
#   slurm/check_python_on_nodes.sh rtxa6000 0 5
#   slurm/check_python_on_nodes.sh --summary rtxa6000 0 5
# A6000 fleet is tron00-tron05; we don't yet know which have python3.12.
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
# WHY A6000 INSTEAD OF A5000:
# This loads the base in bf16, not 4-bit, because PeftModel.merge_and_unload()
# (needed to fold the SFT adapter into the base before DPO trains on top)
# doesn't compose with bitsandbytes 4-bit. At 8B, bf16 base ≈ 16 GB; add
# activations + gradient buffers + DPO LoRA + ref-model path and peak GPU
# memory lands around 18-22 GB. That's *tight* on A5000 (24 GB) — one bad
# batch and you OOM. A6000 (48 GB) gives comfortable headroom.
# If A6000 is contended, you can try A5000 + drop max_length 512 → 384 in
# sftdpo_8b_config.yaml as a fallback.

set -euo pipefail
source env.sh

python3.12 finetune/sftdpo_train.py \
    --config finetune/configs/sftdpo_8b_config.yaml \
    --sft-adapter-path checkpoints/sft_8b
