#!/bin/bash
#SBATCH --job-name=ots_probe
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --output=slurm/eval_ots_probe.%A_%a.out
#SBATCH --error=slurm/eval_ots_probe.%A_%a.err

# Cluster-representative probe grid for Llama-3.1-8B-Instruct, all three
# benchmarks. Completes the off-the-shelf arm: the pretrained 8b already has
# all 8 probe files per benchmark, the instruct release has none except BLEnD's
# US swap.
#
# add_probe.py rather than the eval scripts: it reads `pred` from the
# _usprobe.json and runs the swap pass only, which is half the forward passes.
# --probe-country is nargs="+", so all 7 countries share ONE model load -- on a
# one-pass job that load is most of the non-GPU cost, which is why this is 3
# tasks and not 21.
#
# US is not in the list: it IS the base file, so its swap is already recorded
# there as us_pred.
#
# --raw-prompt must match how the base file was made, or `pred` and `us_pred`
# in the same row would come from different prompt formats. add_probe carries
# the _raw tail from the base name into the output name for the same reason.
#
# DEPENDENCY: every task needs eval_offtheshelf_usprobe_job.sh to have written
# the raw-prompted base file first -- BLEnD included, since the BLEnD instruct
# base file already on disk was chat-templated and cannot seed a raw pass.
# Submit as:
#   JID=$(sbatch --parsable slurm/eval_offtheshelf_usprobe_job.sh)
#   sbatch --dependency=afterok:$JID slurm/eval_offtheshelf_probe_job.sh
#
# --skip-existing makes a requeue resume at the first missing country instead
# of redoing the grid, which matters because BLEnD is 23,032 examples x 7
# countries at ~10 it/s, roughly 4.5 h. NormAd and CulturalBench are minutes.
# Full grid, no --sample-frac, so these pair with the existing full-coverage
# probe files rather than a hashed subset of them.

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODEL=8b_instruct
COUNTRIES=(netherlands hungary bosnia_and_herzegovina mexico taiwan philippines iran)

BENCHES=(normad culturalbench blend)
BENCH=${BENCHES[$SLURM_ARRAY_TASK_ID]}

case "$BENCH" in
  normad)        BASE="outputs/behavioral/normad_base_${MODEL}_fs2_yn_usprobe_raw.json" ;;
  culturalbench) BASE="outputs/behavioral/culturalbench_base_${MODEL}_nfs4_usprobe_raw.json" ;;
  blend)         BASE="outputs/behavioral/blend_base_${MODEL}_nfs_usprobe_raw.json" ;;
esac

if [[ ! -f "$BASE" ]]; then
    echo "[ots_probe] missing base file $BASE — run eval_offtheshelf_usprobe_job.sh first" >&2
    exit 1
fi

echo "[ots_probe] benchmark=$BENCH condition=base model=$MODEL base=$BASE"

python evaluate/add_probe.py \
    --base "$BASE" \
    --probe-country "${COUNTRIES[@]}" \
    --condition base \
    --model-size "$MODEL" \
    --precision matched_bf16 \
    --raw-prompt \
    --skip-existing
