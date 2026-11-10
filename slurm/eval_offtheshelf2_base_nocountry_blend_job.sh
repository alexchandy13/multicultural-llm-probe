#!/bin/bash
#SBATCH --job-name=ots2_base_nc
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=2:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --array=0-1
#SBATCH --output=slurm/eval_ots2_base_nc.%A_%a.out
#SBATCH --error=slurm/eval_ots2_base_nc.%A_%a.err

# BLEnD country ablation for the pretrained-only (non-instruct) checkpoints
# of the second off-the-shelf tier, to pair against the already-run
# blend_offshelf_{olmo3,qwen35}_instruct_nfs_nocountry_raw.json files from
# eval_instruct_nocountry_job.sh:
#   olmo3   allenai/Olmo-3-1025-7B      (pretrained only)
#   qwen35  Qwen/Qwen3.5-9B-Base        (pretrained only)
#
# condition=offshelf, matching the instruct arm's slug (both are vendor
# releases, no adapter of ours).
#
# BLEnD only -- this is for the base-vs-instruct no-country entropy/US-lean
# comparison, which doesn't need NormAd or CulturalBench.
#
# --raw-prompt is a no-op on these (is_instruct() is false for bare
# "olmo3"/"qwen35"), kept only so the filename tail matches the instruct
# arm's _raw tail and the two are easy to pair up.
#
# Submit with: sbatch slurm/eval_offtheshelf2_base_nocountry_blend_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODELS=(olmo3 qwen35)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "[ots2_base_nc] benchmark=blend condition=offshelf model=$MODEL"

python evaluate/eval_blend.py \
    --condition offshelf --model-size "$MODEL" --precision matched_bf16 \
    --neutral-fewshot --no-country --raw-prompt
