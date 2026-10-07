#!/bin/bash
#SBATCH --job-name=probe_grid
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=12:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/probe_grid.%A_%a.out
#SBATCH --error=slurm/probe_grid.%A_%a.err

# Fill the cluster-representative probe grid: 8 IW clusters x 3 benchmarks x 12 models.
#
# Replaces the four per-benchmark probe jobs, which hardcoded different country
# subsets and between them left 169 of 288 cells empty.
#
# One array task per CONDITION, not per (condition, country): add_probe.py takes
# all seven countries and loads the model once for them. Loading is most of the
# cost here, since the probe itself is a single forward pass over predictions that
# already exist — per-country tasks would load the model 7x for the same work.
#
# --skip-existing makes tasks resumable, which matters on scavenger: a preempted
# task is requeued and picks up at the first country it had not finished.
#
# EnglishSpeaking is represented by the US probe. That comes from the eval scripts'
# own --us-probe and is the BASE file here, never a task below.
#
#   BENCH=normad        MODEL=8b     sbatch --array=0-4 slurm/eval_probe_grid_job.sh
#   BENCH=blend         MODEL=gemma4 sbatch --array=0-4 --gres=gpu:rtxa6000:1 slurm/eval_probe_grid_job.sh
#   BENCH=culturalbench MODEL=8b CONDS="tulu3_sft tulu3_dpo" sbatch --array=0-1 slurm/eval_probe_grid_job.sh

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

BENCH="${BENCH:?set BENCH to normad, blend or culturalbench}"
MODEL="${MODEL:-8b}"
read -ra CONDS <<< "${CONDS:-base sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult}"
# The 7 non-US cluster representatives, from CLUSTER_REPS in analysis/cluster_probe_analysis.py.
read -ra COUNTRIES <<< "${COUNTRIES:-netherlands hungary bosnia_and_herzegovina mexico taiwan philippines iran}"

if (( SLURM_ARRAY_TASK_ID >= ${#CONDS[@]} )); then
    echo "[probe_grid] task $SLURM_ARRAY_TASK_ID exceeds ${#CONDS[@]} conditions — use --array=0-$(( ${#CONDS[@]} - 1 ))" >&2
    exit 1
fi
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

# NormAd is scored with --multi-prompt-word, BLEnD and CulturalBench plain 0-shot
# neutral-fewshot; the base filename has to match how each was actually produced.
case "$BENCH" in
    normad)        MIDDLE="_nfs_mpw" ;;
    blend)         MIDDLE="_nfs" ;;
    culturalbench) MIDDLE="_nfs" ;;
    *) echo "BENCH must be normad, blend or culturalbench, got '$BENCH'" >&2; exit 1 ;;
esac

BASE="outputs/behavioral/${BENCH}_${COND}_${MODEL}${MIDDLE}_usprobe.json"
if [[ ! -f "$BASE" ]]; then
    echo "[probe_grid] missing base $BASE" >&2
    echo "[probe_grid] run the benchmark's own eval with --us-probe first" >&2
    exit 1
fi

echo "[probe_grid] bench=$BENCH model=$MODEL condition=$COND countries=${COUNTRIES[*]}"
python evaluate/add_probe.py \
    --base "$BASE" \
    --probe-country "${COUNTRIES[@]}" \
    --condition "$COND" \
    --model-size "$MODEL" \
    --skip-existing
