#!/bin/bash
#SBATCH --job-name=eval_olmoe
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/eval_olmoe.%j.out
#SBATCH --error=slurm/eval_olmoe.%j.err

# Behavioral eval for allenai/OLMoE-1B-7B-0924-Instruct (sparse MoE, 64 experts
# top-8, 16 layers). Only the `base` condition applies: the model is already
# instruction-tuned and we train no adapters on it, same as 8b_instruct.
#
# ~7B params in bf16 is ~14 GB. All 64 experts stay resident even though only 8
# activate per token, so size is set by total params, not active ones — a single
# 24 GB card is enough.
#
# Variants mirror eval_8b_normad_nfs_job.sh / eval_8b_blend_nfs_job.sh /
# eval_culturalbench_8b_job.sh exactly, so OLMoE numbers are comparable with the
# existing 8b and gemma4 result set. --neutral-fewshot is what makes the
# behavioral prompts match the CULNIG scoring format.

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

echo "[eval_olmoe] base condition, model-size=olmoe_instruct"

# NormAd fs2 + yn. Chosen over NFS/mpw on measurement quality: on 8b and gemma4
# base this gives the highest accuracy and the tightest yes-bias (+5.5 / +1.5),
# needs no post-hoc calibration, and avoids mpw's 4x prompt cost. Note this is NOT
# the CULNIG-comparable format — fs2 draws its shots from holdout countries, whose
# cultural content would contaminate gradient attribution (CULNIG maxes over the
# sequence, so prefix tokens compete). Use _nfs runs for anything paired with CULNIG.
python evaluate/eval_normad.py --condition base --model-size olmoe_instruct \
    --few-shot 2 --yn-only --us-probe

# BLEnD NFS, kept deliberately: this is the CULNIG-paired format (culnig blend
# scoring uses the same NFS prefix). BLEnD is A/B/C/D with option balance already
# within ~4pt of gold under NFS, so fs2 has no bias problem to fix here — it buys
# <1pt accuracy on 8b while worsening option balance and dropping 2.7k examples.
python evaluate/eval_blend.py --condition base --model-size olmoe_instruct \
    --neutral-fewshot --us-probe

# CulturalBench. No prefix flag exists — the script hardcodes _nfs — so this is
# always the NFS variant. Read its yes-bias alongside accuracy: gold is only 27%
# yes, and most conditions over-predict yes by 15-36pt.
python evaluate/eval_culturalbench.py --condition base --model-size olmoe_instruct \
    --us-probe
