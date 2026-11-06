#!/bin/bash
#SBATCH --job-name=inst_probe
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --output=slurm/eval_inst_probe.%A_%a.out
#SBATCH --error=slurm/eval_inst_probe.%A_%a.err

# Stage 2 of 2: the probe grid. Replaces the country named in each prompt with
# a probe country and rescores, reading `pred` from the stage-1 baseline file
# so this is one forward pass per row per country rather than two.
#
# The US is an ordinary entry in --countries. It used to be built by the eval
# script's --us-probe in the same run as the baseline, while the other seven
# came from add_probe, so there were two implementations of "swap the country
# and rescore" and they drifted -- see eval_instruct_baseline_job.sh.
#
# WHAT RUNS IS GIVEN ON THE COMMAND LINE, not edited into the script. The array
# indexes the cartesian product of the three lists, so the layout is whatever
# was asked for:
#
#   --models        comma list of model_size labels
#   --benchmarks    comma list of normad,culturalbench,blend
#   --countries     comma list of probe countries (slugs are fine; they are
#                   resolved to each benchmark's own spelling by
#                   evaluate/country_names.py)
#   --split-countries   one country per task instead of all in one task
#   --condition     override the per-model default (base for the Llama/Gemma
#                   runs, which predate the rename; offshelf for the rest)
#
# tasks = n_models * n_benchmarks * (n_countries if --split-countries else 1)
#
# --split-countries is what BLEnD wants: 23,032 rows is ~1 h per country, so
# eight in one task is a serial ~8 h job that is hard to place on a preemptible
# partition and dies with nothing to show if it hits the walltime (--requeue
# covers preemption, not TIMEOUT). NormAd (1,697 rows) and CulturalBench
# (4,905) finish all eight in minutes, so they want the default.
#
# Run with --layout to print the task mapping and exit without scoring
# anything. It needs no GPU and no slurm, so check the array size that way
# before submitting:
#
#   bash slurm/eval_instruct_probe_job.sh --layout --models olmo3_instruct \
#        --benchmarks blend --split-countries
#   sbatch --array=0-7 slurm/eval_instruct_probe_job.sh --models olmo3_instruct \
#        --benchmarks blend --split-countries
#
# Needs eval_instruct_baseline_job.sh to have finished for each (model,
# benchmark) pair, which the missing-base check below enforces.

set -euo pipefail

MODELS_ARG="8b_instruct,gemma4_instruct,olmo3_instruct,qwen35_instruct"
BENCHES_ARG="normad,culturalbench,blend"
# us first so a partial run gets the most-used column done first.
COUNTRIES_ARG="us,netherlands,hungary,bosnia_and_herzegovina,mexico,taiwan,philippines,iran"
SPLIT=0
COND_OVERRIDE=""
LAYOUT=0
# Selects which baseline to probe, and must match the --shots the baseline job
# ran with: the shot count is part of the filename, and add_probe rebuilds the
# base run's prefix from that name so the probe prompt carries the same context.
SHOTS=2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)      MODELS_ARG="$2"; shift 2 ;;
        --benchmarks)  BENCHES_ARG="$2"; shift 2 ;;
        --countries)   COUNTRIES_ARG="$2"; shift 2 ;;
        --condition)   COND_OVERRIDE="$2"; shift 2 ;;
        --shots)       SHOTS="$2"; shift 2 ;;
        --split-countries) SPLIT=1; shift ;;
        --layout)      LAYOUT=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if ! [[ "$SHOTS" =~ ^[0-9]+$ ]]; then
    echo "--shots takes a non-negative integer, got: $SHOTS" >&2; exit 2
fi

IFS=',' read -r -a MODELS    <<< "$MODELS_ARG"
IFS=',' read -r -a BENCHES   <<< "$BENCHES_ARG"
IFS=',' read -r -a COUNTRIES <<< "$COUNTRIES_ARG"

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
# One "country group" per task when splitting, otherwise a single group holding
# every country -- which is what makes the index arithmetic the same either way.
if [[ $SPLIT -eq 1 ]]; then N_GROUPS=${#COUNTRIES[@]}; else N_GROUPS=1; fi
N_TASKS=$(( N_MODELS * N_BENCH * N_GROUPS ))

# The condition slug goes into the filename and the analysis pairs probe files
# against baselines by exact name, so it is per-model rather than global.
cond_for() {
    if [[ -n "$COND_OVERRIDE" ]]; then echo "$COND_OVERRIDE"; return; fi
    case "$1" in
        8b_instruct|gemma4_instruct) echo "base" ;;
        *)                           echo "offshelf" ;;
    esac
}

# Kept identical to out_for() in eval_instruct_baseline_job.sh; the two scripts
# meet on these names and nothing else.
base_for() {
    local bench="$1" cond="$2" model="$3"
    local fs="" nfs=""
    if (( SHOTS > 0 )); then
        fs="_fs${SHOTS}"
        if (( SHOTS == 2 )); then nfs="_nfs"; else nfs="_nfs${SHOTS}"; fi
    fi
    case "$bench" in
        normad)        echo "outputs/behavioral/normad_${cond}_${model}${fs}_yn_wp_raw.json" ;;
        culturalbench) echo "outputs/behavioral/culturalbench_${cond}_${model}${nfs}_raw.json" ;;
        blend)         echo "outputs/behavioral/blend_${cond}_${model}$( (( SHOTS > 0 )) && echo _nfs )_raw.json" ;;
        *) echo "unknown benchmark: $bench" >&2; return 1 ;;
    esac
}

# task id -> (model, benchmark, country group). Decomposed most-significant
# first so a model's tasks stay contiguous.
decompose() {
    local id="$1"
    local per_model=$(( N_BENCH * N_GROUPS ))
    IDX_MODEL=$(( id / per_model ))
    local rem=$(( id % per_model ))
    IDX_BENCH=$(( rem / N_GROUPS ))
    IDX_GROUP=$(( rem % N_GROUPS ))
}

if [[ $LAYOUT -eq 1 ]]; then
    echo "tasks: $N_TASKS   (submit with --array=0-$(( N_TASKS - 1 )))"
    for (( t = 0; t < N_TASKS; t++ )); do
        decompose "$t"
        m=${MODELS[$IDX_MODEL]}; b=${BENCHES[$IDX_BENCH]}
        if [[ $SPLIT -eq 1 ]]; then p=${COUNTRIES[$IDX_GROUP]}; else p="${COUNTRIES[*]}"; fi
        printf '  %3d  %-9s %-16s %-13s %s\n' "$t" "$(cond_for "$m")" "$m" "$b" "$p"
        printf '       base: %s\n' "$(base_for "$b" "$(cond_for "$m")" "$m")"
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

decompose "$SLURM_ARRAY_TASK_ID"
MODEL=${MODELS[$IDX_MODEL]}
BENCH=${BENCHES[$IDX_BENCH]}
COND=$(cond_for "$MODEL")
if [[ $SPLIT -eq 1 ]]; then PROBES=("${COUNTRIES[$IDX_GROUP]}"); else PROBES=("${COUNTRIES[@]}"); fi

BASE=$(base_for "$BENCH" "$COND" "$MODEL")
if [[ ! -f "$BASE" ]]; then
    echo "[inst_probe] missing base $BASE — run eval_instruct_baseline_job.sh first" >&2
    exit 1
fi

echo "[inst_probe] model=$MODEL condition=$COND benchmark=$BENCH probes=${PROBES[*]}"

python evaluate/add_probe.py \
    --base "$BASE" \
    --probe-country "${PROBES[@]}" \
    --condition "$COND" \
    --model-size "$MODEL" \
    --precision matched_bf16 \
    --raw-prompt \
    --skip-existing
