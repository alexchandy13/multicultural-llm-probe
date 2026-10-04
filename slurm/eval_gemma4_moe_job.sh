#!/bin/bash
#SBATCH --job-name=eval_g4moe
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/eval_g4moe.%A_%a.out
#SBATCH --error=slurm/eval_g4moe.%A_%a.err

# Behavioral eval for Gemma 4 26B-A4B (sparse MoE, 4B active). Array 0 = base,
# 1 = instruct. Only the `base` condition applies — no adapters trained on these.
#
# 26B total in bf16 is ~52 GB and all experts stay resident regardless of the 4B
# active, so size is set by total params: 2x48 GB rtxa6000. One A6000 is short.
#
# Variants mirror the 8b/gemma4 jobs so numbers are comparable: NormAd fs2+yn,
# BLEnD NFS (the CULNIG-paired format), CulturalBench (hardcodes _nfs).

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

VARIANTS=(gemma4_moe gemma4_moe_instruct)
SIZE=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_g4moe] base condition, model-size=$SIZE"

python evaluate/eval_normad.py --condition base --model-size "$SIZE" \
    --few-shot 2 --yn-only --us-probe

python evaluate/eval_blend.py --condition base --model-size "$SIZE" \
    --neutral-fewshot --us-probe

python evaluate/eval_culturalbench.py --condition base --model-size "$SIZE" \
    --us-probe
