#!/bin/bash
#SBATCH --job-name=culnig_shard_g4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/culnig_shard_g4.%A_%a.out
#SBATCH --error=slurm/culnig_shard_g4.%A_%a.err

# Shard one condition x dataset across 8 parallel tasks. Each task scores every
# 8th sample and writes a dense .npz to outputs/neurons/{cond}_gemma4/_shards/.
# Preemption costs one shard (~2h) instead of a whole 15h run, and the shards
# run concurrently, so wall clock drops to roughly one shard.
#
# Usage:
#   sbatch --export=ALL,COND=sftdpo_aya_nocult,DS=blend,TD=all \
#       slurm/culnig_sharded_gemma4_job.sh
#
# Then merge (afterok so a requeued shard still gates it):
#   sbatch --dependency=afterok:<arrayjobid> --partition=scavenger \
#       --account=scavenger --qos=scavenger --wrap \
#       "source env.sh && source /fs/nexus-scratch/\$USER/miniforge/etc/profile.d/conda.sh \
#        && conda activate llm \
#        && python scripts/merge_shards.py --condition sftdpo_aya_nocult \
#           --model-size gemma4 --dataset blend"
#
# legacygpu nodes excluded: their GPUs predate the compute capability our torch
# build ships kernels for (cudaErrorNoKernelImageForDevice on load).

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

: "${COND:?set COND, e.g. --export=ALL,COND=sftdpo_aya_nocult}"
: "${DS:?set DS, e.g. --export=ALL,DS=blend}"
TD="${TD:-neuron}"

# Fixed, not derived from SLURM_ARRAY_TASK_COUNT: resubmitting one failed shard
# with --array=3 would otherwise set the count to 1 and silently rescore the
# whole dataset into a partial named 3of1.
N="${NSHARDS:-8}"
if (( SLURM_ARRAY_TASK_ID >= N )); then
    echo "shard id $SLURM_ARRAY_TASK_ID >= NSHARDS=$N — set NSHARDS to match --array" >&2
    exit 1
fi
echo "[culnig_shard_g4] cond=$COND ds=$DS target_data=$TD shard=$SLURM_ARRAY_TASK_ID/$N"

python culnig/calc_neuron_score.py \
    --condition "$COND" \
    --model-size gemma4 \
    --precision matched_bf16 \
    --dataset-names "$DS" \
    --target-data "$TD" \
    --shard "$SLURM_ARRAY_TASK_ID/$N"
