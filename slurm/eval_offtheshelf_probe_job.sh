#!/bin/bash
#SBATCH --job-name=ots_probe
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=8:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --array=0-8
#SBATCH --output=slurm/eval_ots_probe.%A_%a.out
#SBATCH --error=slurm/eval_ots_probe.%A_%a.err

# Cluster-representative probe grid for Llama-3.1-8B-Instruct, raw-prompted.
# Completes the off-the-shelf arm: the pretrained 8b already has all 8 probe
# files per benchmark, the instruct release has none.
#
# add_probe.py rather than the eval scripts: it reads `pred` from the
# _usprobe_raw.json and runs the swap pass only, half the forward passes.
#
# TASK LAYOUT is uneven on purpose. --probe-country is nargs="+", so several
# countries can share one model load, which is most of the non-GPU cost of a
# one-pass job. That is worth doing for NormAd (1,697 rows) and CulturalBench
# (4,905), where all 7 finish in minutes. It is NOT viable for BLEnD: 23,032
# rows is ~1.5 h per country on a 12B model, measured from
# eval_blend_nocountry_gemma4_job.sh fitting one pass in 2 h, so 7 in one task
# would exceed any walltime this partition grants. BLEnD therefore gets one
# country per task.
#   task 0   normad         all 7 countries
#   task 1   culturalbench  all 7 countries
#   task 2-8 blend          one country each
#
# US is not in the list: it IS the base file, so its swap is already recorded
# there as us_pred.
#
# --raw-prompt must match how the base file was made, or `pred` and `us_pred`
# in one row would come from different prompt formats. add_probe carries the
# _raw tail from the base name into the output name for the same reason.
#
# --requeue plus --skip-existing is the pair that makes scavenger safe here: a
# preempted task comes back and resumes at the first missing country instead of
# redoing the grid. A walltime TIMEOUT is not requeued, which is the other
# reason BLEnD is one country per task.
#
# DEPENDENCY: every task needs eval_offtheshelf_usprobe_job.sh to have written
# the raw-prompted base file first -- BLEnD included, since the BLEnD instruct
# base file already on disk was chat-templated and cannot seed a raw pass.
# Submit as:
#   JID=$(sbatch --parsable slurm/eval_offtheshelf_usprobe_job.sh)
#   sbatch --dependency=afterok:$JID slurm/eval_offtheshelf_probe_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODEL=8b_instruct
COUNTRIES=(netherlands hungary bosnia_and_herzegovina mexico taiwan philippines iran)

case "$SLURM_ARRAY_TASK_ID" in
  0) BENCH=normad;        BASE="outputs/behavioral/normad_base_${MODEL}_fs2_yn_usprobe_raw.json"
     PROBES=("${COUNTRIES[@]}") ;;
  1) BENCH=culturalbench; BASE="outputs/behavioral/culturalbench_base_${MODEL}_nfs4_usprobe_raw.json"
     PROBES=("${COUNTRIES[@]}") ;;
  *) BENCH=blend;         BASE="outputs/behavioral/blend_base_${MODEL}_nfs_usprobe_raw.json"
     PROBES=("${COUNTRIES[$(( SLURM_ARRAY_TASK_ID - 2 ))]}") ;;
esac

if [[ ! -f "$BASE" ]]; then
    echo "[ots_probe] missing base file $BASE — run eval_offtheshelf_usprobe_job.sh first" >&2
    exit 1
fi

echo "[ots_probe] benchmark=$BENCH model=$MODEL probes=${PROBES[*]} base=$(basename $BASE)"

python evaluate/add_probe.py \
    --base "$BASE" \
    --probe-country "${PROBES[@]}" \
    --condition base \
    --model-size "$MODEL" \
    --precision matched_bf16 \
    --raw-prompt \
    --skip-existing
