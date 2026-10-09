#!/bin/bash
#SBATCH --job-name=culnig_shard
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/culnig_shard.%A_%a.out
#SBATCH --error=slurm/culnig_shard.%A_%a.err
#
# Model-agnostic version of culnig_sharded_gemma4_job.sh, which hardcodes
# --model-size gemma4 and 2 GPUs. 8B fits on one A6000, so override --gres when
# scoring gemma4 through this one.
#
# Shards one condition x dataset across N parallel tasks; each scores every Nth
# sample and writes a dense .npz to outputs/neurons/{cond}{_size}/_shards/.
# Merge afterwards with scripts/merge_shards.py --dataset "$DS".
#
# countryrc works here too, which it did not before: upstream's loader refuses
# countryrc without an explicit country list, and the main pass used to hardcode
# target_countries=None, so countryrc could only run through the separate second
# pass that --shard skips. calc_neuron_score now supplies all 81 countries when
# countryrc appears in --dataset-names, so it shards like anything else.
#
#   MODEL=8b COND=tulu3_sft DS=normad YN=1 sbatch --array=0-7 slurm/culnig_sharded_job.sh
#   MODEL=8b COND=tulu3_sft DS=countryrc  sbatch --array=0-7 slurm/culnig_sharded_job.sh
#
# Then:
#   python scripts/merge_shards.py --condition tulu3_sft --model-size 8b --dataset normad_yn

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

: "${COND:?set COND, e.g. COND=tulu3_sft}"
: "${DS:?set DS, e.g. DS=normad}"
MODEL="${MODEL:-8b}"
TD="${TD:-neuron}"

# Fixed, not derived from SLURM_ARRAY_TASK_COUNT: resubmitting one failed shard
# with --array=3 would otherwise set the count to 1 and silently rescore the
# whole dataset into a partial named 3of1.
N="${NSHARDS:-8}"
if (( SLURM_ARRAY_TASK_ID >= N )); then
    echo "shard id $SLURM_ARRAY_TASK_ID >= NSHARDS=$N — set NSHARDS to match --array" >&2
    exit 1
fi

EXTRA=()
[[ "${YN:-0}" == "1" ]] && EXTRA+=(--yn-only)

echo "[culnig_shard] cond=$COND model=$MODEL ds=$DS target_data=$TD shard=$SLURM_ARRAY_TASK_ID/$N"
python culnig/calc_neuron_score.py \
    --condition "$COND" \
    --model-size "$MODEL" \
    --precision matched_bf16 \
    --dataset-names "$DS" \
    --target-data "$TD" \
    --shard "$SLURM_ARRAY_TASK_ID/$N" ${EXTRA[@]+"${EXTRA[@]}"}
