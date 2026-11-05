#!/bin/bash
#SBATCH --job-name=inst_base
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --output=slurm/eval_inst_baseline.%A_%a.out
#SBATCH --error=slurm/eval_inst_baseline.%A_%a.err
#
# Stage 1 of 2: the BASELINE run. Each example is scored with its own true
# country in the prompt and no probe at all, so the output holds pred/scores
# and nothing else. Stage 2 is slurm/eval_instruct_probe_job.sh, which reads
# these files.
#
# This exists because the old layout entangled the baseline with one probe.
# `--us-probe` wrote pred AND us_pred in a single run, and that file then
# served as add_probe's --base for the other seven countries. So the US swap
# was built by the eval script while every other country was built by
# add_probe, and the two implementations drifted: add_probe hardcoded a
# 4-template prompt ensemble where the base run used one template, and it
# formatted the raw CLI string into the prompt, giving "country:
# bosnia_and_herzegovina" against the US column's "country: United States".
# Every non-US probe was handicapped against the US one.
#
# Splitting the stages costs nothing. Old layout: 2 passes per item for the
# us-probe run plus 7 probe runs = 9. New layout: 1 pass for the baseline plus
# 8 probe runs = 9. The US becomes an ordinary probe country and there is one
# code path for all eight.
#
# WHAT RUNS IS GIVEN ON THE COMMAND LINE, matching the probe job:
#
#   --models       comma list of model_size labels
#   --benchmarks   comma list of normad,culturalbench,blend
#   --condition    override the per-model default (base for the Llama/Gemma
#                  runs, which predate the rename; offshelf for the rest).
#                  The slug goes into the filename and the analysis pairs
#                  probe files against baselines by exact name, so it is not
#                  interchangeable. resolve_condition() treats them
#                  identically -- no adapter of ours either way.
#
# tasks = n_models * n_benchmarks. Run with --layout to print the mapping and
# exit; it needs no GPU and no slurm.
#
#   bash slurm/eval_instruct_baseline_job.sh --layout --benchmarks normad
#   sbatch --array=0-3 slurm/eval_instruct_baseline_job.sh --benchmarks normad
#
# NormAd is ~1,700 items (minutes), CulturalBench ~4,900, BLEnD ~23k (~40 min,
# since the baseline is a single pass rather than the two the old us-probe run
# made). Skips and exits 0 when its output exists, so refilling gaps costs one
# model load per task.

set -euo pipefail

MODELS_ARG="8b_instruct,gemma4_instruct,olmo3_instruct,qwen35_instruct"
BENCHES_ARG="normad,culturalbench,blend"
COND_OVERRIDE=""
LAYOUT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)     MODELS_ARG="$2"; shift 2 ;;
        --benchmarks) BENCHES_ARG="$2"; shift 2 ;;
        --condition)  COND_OVERRIDE="$2"; shift 2 ;;
        --layout)     LAYOUT=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

IFS=',' read -r -a MODELS  <<< "$MODELS_ARG"
IFS=',' read -r -a BENCHES <<< "$BENCHES_ARG"

# Validated up front so a typo fails now rather than after a model load, or --
# worse -- silently in --layout. Model labels are checked by resolve_condition()
# in python; the benchmark list is checked here because it drives the filenames.
for b in "${BENCHES[@]}"; do
    case "$b" in
        normad|culturalbench|blend) ;;
        *) echo "unknown benchmark: $b (expected normad, culturalbench or blend)" >&2
           exit 2 ;;
    esac
done

N_MODELS=${#MODELS[@]}
N_BENCH=${#BENCHES[@]}
N_TASKS=$(( N_MODELS * N_BENCH ))

cond_for() {
    if [[ -n "$COND_OVERRIDE" ]]; then echo "$COND_OVERRIDE"; return; fi
    case "$1" in
        8b_instruct|gemma4_instruct) echo "base" ;;
        *)                           echo "offshelf" ;;
    esac
}

out_for() {
    local bench="$1" cond="$2" model="$3"
    case "$bench" in
        normad)        echo "outputs/behavioral/normad_${cond}_${model}_fs2_yn_raw.json" ;;
        culturalbench) echo "outputs/behavioral/culturalbench_${cond}_${model}_nfs4_raw.json" ;;
        blend)         echo "outputs/behavioral/blend_${cond}_${model}_nfs_raw.json" ;;
        *) echo "unknown benchmark: $bench" >&2; return 1 ;;
    esac
}

if [[ $LAYOUT -eq 1 ]]; then
    echo "tasks: $N_TASKS   (submit with --array=0-$(( N_TASKS - 1 )))"
    for (( t = 0; t < N_TASKS; t++ )); do
        m=${MODELS[$(( t / N_BENCH ))]}; b=${BENCHES[$(( t % N_BENCH ))]}
        c=$(cond_for "$m")
        printf '  %3d  %-9s %-16s %-13s -> %s\n' "$t" "$c" "$m" "$b" "$(out_for "$b" "$c" "$m")"
    done
    exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "no SLURM_ARRAY_TASK_ID — submit with --array=0-$(( N_TASKS - 1 ))," \
         "or pass --layout to inspect the mapping" >&2
    exit 2
fi
if (( SLURM_ARRAY_TASK_ID >= N_TASKS )); then
    echo "task $SLURM_ARRAY_TASK_ID is past the end of this grid ($N_TASKS tasks)" >&2
    exit 2
fi

source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODEL=${MODELS[$(( SLURM_ARRAY_TASK_ID / N_BENCH ))]}
BENCH=${BENCHES[$(( SLURM_ARRAY_TASK_ID % N_BENCH ))]}
COND=$(cond_for "$MODEL")
OUT=$(out_for "$BENCH" "$COND" "$MODEL")

if [[ -f "$OUT" ]]; then echo "[inst_base] $OUT exists - nothing to do"; exit 0; fi

echo "[inst_base] benchmark=$BENCH condition=$COND model=$MODEL"

# Flags match the old us-probe jobs exactly apart from the dropped --us-probe,
# so these baselines are comparable to everything already on disk.
case "$BENCH" in
  normad)
    python evaluate/eval_normad.py --condition "$COND" --model-size "$MODEL" \
        --precision matched_bf16 --few-shot 2 --yn-only --raw-prompt
    ;;
  culturalbench)
    python evaluate/eval_culturalbench.py --condition "$COND" --model-size "$MODEL" \
        --precision matched_bf16 --neutral-shots 4 --raw-prompt
    ;;
  blend)
    python evaluate/eval_blend.py --condition "$COND" --model-size "$MODEL" \
        --precision matched_bf16 --neutral-fewshot --raw-prompt
    ;;
esac
