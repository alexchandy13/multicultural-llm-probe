#!/bin/bash
#SBATCH --job-name=inst_probe
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
# A task is now every model for one (benchmark, country group), serially, so
# the ceiling is generous; preemption is covered by --requeue plus
# --skip-existing, TIMEOUT is not.
#SBATCH --time=12:00:00
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
# ONE TASK PER (benchmark, country group): each task loops over every model in
# --models, so the full normad+culturalbench probe pass is a 2-wide array
# rather than 10. The models differ only in which checkpoint is loaded.
#
# tasks = n_benchmarks * (n_countries if --split-countries else 1)
#                      * (n_models if --split-models else 1)
#
# --split-countries is what BLEnD wants: 23,032 rows is ~1 h per country, so
# eight in one task is a serial ~8 h job that is hard to place on a preemptible
# partition and dies with nothing to show if it hits the walltime (--requeue
# covers preemption, not TIMEOUT). NormAd (1,697 rows) and CulturalBench
# (4,905) finish all eight in minutes, so they want the default.
#
# --layout prints the task mapping and the array size, then exits; it needs no
# GPU and no slurm, so check the size that way before submitting:
#
#   bash slurm/eval_instruct_probe_job.sh --layout --benchmarks blend \
#        --split-countries
#   sbatch --array=0-1 slurm/eval_instruct_probe_job.sh \
#        --benchmarks normad,culturalbench
#   sbatch --array=0-7 slurm/eval_instruct_probe_job.sh \
#        --benchmarks blend --split-countries
#
# --split-models additionally gives one task per model, which BLEnD wants when
# the queue is deep enough to place a wide array faster than a long task.
#
# --array is accepted only to requeue part of a finished grid (--array 3,7).
#
# Needs eval_instruct_baseline_job.sh to have finished for each (model,
# benchmark) pair, which the missing-base check below enforces.

set -euo pipefail


# 8b:tulu3_dpo is label:condition -- Tulu is a condition over the 8b base,
# not a model label, and it is the fully post-trained DPO arm; the SFT-only
# arm is no longer part of the table.
MODELS_ARG="8b_instruct,gemma4_instruct,olmo3_instruct,qwen35_instruct,8b:tulu3_dpo"
BENCHES_ARG="normad,culturalbench,blend"
# us first so a partial run gets the most-used column done first.
COUNTRIES_ARG="us,netherlands,hungary,bosnia_and_herzegovina,mexico,taiwan,philippines,iran"
SPLIT=0
COND_OVERRIDE=""
LAYOUT=0
# One task per model as well, instead of looping them inside the task. Off by
# default -- see --split-countries, which is the split that actually matters.
SPLIT_MODELS=0
# Selects which baseline to probe, and must match the --shots the baseline job
# ran with: the shot count is part of the filename, and add_probe rebuilds the
# base run's prefix from that name so the probe prompt carries the same context.
SHOTS=2
# Must match the baseline job's arm. --raw-prompt is the default; --templated
# keeps the chat template and passes --system-prompt neutral, which add_probe
# requires whenever the base file carries a _sys tail -- otherwise the template
# inserts its own default system message on the probe side only.
TEMPLATED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)      MODELS_ARG="$2"; shift 2 ;;
        --benchmarks)  BENCHES_ARG="$2"; shift 2 ;;
        --countries)   COUNTRIES_ARG="$2"; shift 2 ;;
        --condition)   COND_OVERRIDE="$2"; shift 2 ;;
        --shots)       SHOTS="$2"; shift 2 ;;
        --templated)   TEMPLATED=1; shift ;;
        --split-countries) SPLIT=1; shift ;;
        --split-models) SPLIT_MODELS=1; shift ;;
        --layout)      LAYOUT=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if ! [[ "$SHOTS" =~ ^[0-9]+$ ]]; then
    echo "--shots takes a non-negative integer, got: $SHOTS" >&2; exit 2
fi

IFS=',' read -r -a MODEL_SPECS <<< "$MODELS_ARG"

# A --models entry may be "label" or "label:condition". The condition slug is
# part of the output filename and the analysis pairs probe files to baselines by
# exact name, so it cannot be global: tulu3_dpo and tulu3_sft are conditions over
# the same 8b base, not model labels, and they have to sit in the same array as
# the instruct checkpoints to be submitted together.
MODELS=(); MODEL_CONDS=()
for spec in "${MODEL_SPECS[@]}"; do
    MODELS+=("${spec%%:*}")
    if [[ "$spec" == *:* ]]; then MODEL_CONDS+=("${spec##*:}"); else MODEL_CONDS+=(""); fi
done
IFS=',' read -r -a BENCHES   <<< "$BENCHES_ARG"
IFS=',' read -r -a COUNTRIES <<< "$COUNTRIES_ARG"

# Validated up front so a typo fails now rather than after a model load, or --
# worse -- silently in --layout. Model labels are checked by resolve_condition()
# in python; the benchmark list is checked here because it drives the filenames.
for b in "${BENCHES[@]}"; do
    case "$b" in
        normad|culturalbench|cbeasy|blend) ;;
        *) echo "unknown benchmark: $b (expected normad, culturalbench,"\
              "cbeasy or blend)" >&2
           exit 2 ;;
    esac
done

N_MODELS=${#MODELS[@]}
N_BENCH=${#BENCHES[@]}
# One "group" per task when splitting, otherwise a single group holding every
# country / model -- which is what makes the index arithmetic the same either way.
if [[ $SPLIT -eq 1 ]]; then N_GROUPS=${#COUNTRIES[@]}; else N_GROUPS=1; fi
if [[ $SPLIT_MODELS -eq 1 ]]; then N_MGROUPS=$N_MODELS; else N_MGROUPS=1; fi
N_TASKS=$(( N_BENCH * N_GROUPS * N_MGROUPS ))

# Precedence: the entry's own condition, then --condition, then the per-label
# default (base for the Llama/Gemma runs, which predate the rename).
cond_at() {
    local i="$1"
    if [[ -n "${MODEL_CONDS[$i]}" ]]; then echo "${MODEL_CONDS[$i]}"; return; fi
    if [[ -n "$COND_OVERRIDE" ]]; then echo "$COND_OVERRIDE"; return; fi
    case "${MODELS[$i]}" in
        8b_instruct|gemma4_instruct) echo "base" ;;
        *)                           echo "offshelf" ;;
    esac
}

# Kept identical to out_for() in eval_instruct_baseline_job.sh; the two scripts
# meet on these names and nothing else.
# The eval scripts append _raw only when --raw-prompt actually changes
# anything, i.e. when the model label is chat-templated (raw_sfx in
# eval_normad.py). A label like plain "8b" has no chat template, so its output
# carries no _raw tail and predicting one here would make the skip check and the
# probe job's --base both point at a file that is never written. Read from
# evaluate/_common.py rather than restating the set, and without importing it:
# _common pulls in torch and peft, and --layout has to run on a login node.
raw_tail_for() {
    local label="$1"
    if [[ $TEMPLATED -eq 1 ]]; then
        # Unconditional, unlike _raw: sys_sfx in the eval scripts is gated only
        # on --system-prompt being set, not on the label being chat-templated.
        # So a label with no chat template (plain "8b") still gets a _sys name
        # even though the system message has no effect on it.
        echo "_sys"
        return
    fi
    if sed -n '/^CHAT_TEMPLATED_SIZES = {/,/^}/p' evaluate/_common.py \
         | grep -q "\"${label}\","; then
        echo "_raw"
    fi
}

base_for() {
    local bench="$1" cond="$2" model="$3"
    local raw; raw=$(raw_tail_for "$model")
    local fs="" nfs=""
    if (( SHOTS > 0 )); then
        fs="_fs${SHOTS}"
        if (( SHOTS == 2 )); then nfs="_nfs"; else nfs="_nfs${SHOTS}"; fi
    fi
    case "$bench" in
        normad)        echo "outputs/behavioral/normad_${cond}_${model}${fs}_yn_wp${raw}.json" ;;
        culturalbench) echo "outputs/behavioral/culturalbench_${cond}_${model}${nfs}${raw}.json" ;;
        # cbeasy shares BLEnD's prompt and so BLEnD's prefix: four fixed
        # MCQ-format shots, on or off, with no count in the name.
        cbeasy)        echo "outputs/behavioral/cbeasy_${cond}_${model}$( (( SHOTS > 0 )) && echo _nfs )${raw}.json" ;;
        blend)         echo "outputs/behavioral/blend_${cond}_${model}$( (( SHOTS > 0 )) && echo _nfs )${raw}.json" ;;
        *) echo "unknown benchmark: $bench" >&2; return 1 ;;
    esac
}

# task id -> (benchmark, country group, model group). Benchmark most significant
# so a benchmark's tasks stay contiguous, which is what --array subsets index into.
decompose() {
    local id="$1"
    local per_bench=$(( N_GROUPS * N_MGROUPS ))
    IDX_BENCH=$(( id / per_bench ))
    local rem=$(( id % per_bench ))
    IDX_GROUP=$(( rem / N_MGROUPS ))
    IDX_MGROUP=$(( rem % N_MGROUPS ))
}

# The models this task runs: one when splitting, all of them otherwise.
models_for_task() {
    if [[ $SPLIT_MODELS -eq 1 ]]; then echo "$IDX_MGROUP"; return; fi
    seq 0 $(( N_MODELS - 1 ))
}

if [[ $LAYOUT -eq 1 ]]; then
    echo "tasks: $N_TASKS   (submit with --array=0-$(( N_TASKS - 1 )))"
    for (( t = 0; t < N_TASKS; t++ )); do
        decompose "$t"
        b=${BENCHES[$IDX_BENCH]}
        if [[ $SPLIT -eq 1 ]]; then p=${COUNTRIES[$IDX_GROUP]}; else p="${COUNTRIES[*]}"; fi
        printf '  %3d  %-13s %s\n' "$t" "$b" "$p"
        for mi in $(models_for_task); do
            c=$(cond_at "$mi")
            printf '       %-9s %-16s base: %s\n' \
                   "$c" "${MODELS[$mi]}" "$(base_for "$b" "$c" "${MODELS[$mi]}")"
        done
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
BENCH=${BENCHES[$IDX_BENCH]}
if [[ $SPLIT -eq 1 ]]; then PROBES=("${COUNTRIES[$IDX_GROUP]}"); else PROBES=("${COUNTRIES[@]}"); fi

echo "[inst_probe] benchmark=$BENCH probes=${PROBES[*]}"

# Mirrors the baseline arm exactly; add_probe refuses a _sys base without a
# matching --system-prompt, so a mismatched submission fails before it scores.
FMT_FLAG=(--raw-prompt)
[[ $TEMPLATED -eq 1 ]] && FMT_FLAG=(--system-prompt neutral)

# A missing base is this task's fault to report, not a reason to drop the other
# models in it, so it joins the failure list like any other error.
FAILED=()
for mi in $(models_for_task); do
    MODEL=${MODELS[$mi]}
    COND=$(cond_at "$mi")
    BASE=$(base_for "$BENCH" "$COND" "$MODEL")
    if [[ ! -f "$BASE" ]]; then
        echo "[inst_probe] missing base $BASE — run eval_instruct_baseline_job.sh first" >&2
        FAILED+=("$COND/$MODEL:no-base")
        continue
    fi
    echo "[inst_probe] === $BENCH / $COND / $MODEL  base=$BASE"
    if ! python evaluate/add_probe.py \
        --base "$BASE" \
        --probe-country "${PROBES[@]}" \
        --condition "$COND" \
        --model-size "$MODEL" \
        --precision matched_bf16 \
        "${FMT_FLAG[@]}" \
        --skip-existing; then
        echo "[inst_probe] FAILED: $BENCH / $COND / $MODEL" >&2
        FAILED+=("$COND/$MODEL")
    fi
done

if (( ${#FAILED[@]} )); then
    echo "[inst_probe] $BENCH: ${#FAILED[@]} failed: ${FAILED[*]}" >&2
    exit 1
fi
echo "[inst_probe] $BENCH: done"
