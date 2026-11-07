#!/bin/bash
#SBATCH --job-name=inst_base
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
# A task is now every model for one benchmark, serially: BLEnD is ~40 min a
# model, so five is ~3.5 h. Preemption is covered by --requeue plus the
# per-model skip, but TIMEOUT is not, so the ceiling is generous.
#SBATCH --time=12:00:00
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
# ONE TASK PER BENCHMARK: tasks = n_benchmarks, and each task loops over every
# model in --models. The models differ only in which checkpoint gets loaded, so
# a task per (model, benchmark) buys nothing but more queue slots to chase.
# --layout prints the mapping and the array size; it needs no GPU and no slurm.
#
#   bash slurm/eval_instruct_baseline_job.sh --layout
#   sbatch --array=0-2 slurm/eval_instruct_baseline_job.sh
#   sbatch --array=0 slurm/eval_instruct_baseline_job.sh --benchmarks normad
#
# --split-models goes back to one task per (model, benchmark), which is worth it
# only for BLEnD: 23k rows x 5 models is a serial ~3.5 h task, and on a
# preemptible partition a wide array places sooner than a long task.
#
# NormAd is ~1,700 items (minutes), CulturalBench ~4,900, BLEnD ~23k (~40 min,
# since the baseline is a single pass rather than the two the old us-probe run
# made). Each model is skipped when its output already exists, so a requeue
# after preemption resumes at the model it died on rather than redoing the
# task.

set -euo pipefail


# 8b:tulu3_dpo is label:condition -- Tulu is a condition over the 8b base,
# not a model label, and it is the fully post-trained DPO arm; the SFT-only
# arm is no longer part of the table.
MODELS_ARG="8b_instruct,gemma4_instruct,olmo3_instruct,qwen35_instruct,8b:tulu3_dpo"
BENCHES_ARG="normad,culturalbench,blend"
COND_OVERRIDE=""
LAYOUT=0
# One task per (model, benchmark) instead of one per benchmark. Off by default:
# a model is just a different checkpoint in the same loop, so fanning out over
# models multiplies queue slots without shortening the critical path. Worth it
# for BLEnD, where a serial five-model task runs for hours.
SPLIT_MODELS=0
# The country-free arm is the same single pass with the country dropped from the
# prompt, so it belongs here rather than in a second hardcoded script. It is the
# reference for the country-free agreement metric, and it has to be rebuilt
# whenever the template changes -- a no-country file written under one template
# cannot be compared against probes scored under another.
NOCOUNTRY=0
# Number of few-shot demonstrations, 0 for zero-shot. A flag rather than an
# edit because it changes the prompt, so it changes the output filename, so a
# zero-shot run and a few-shot run land side by side instead of on each other.
# Each benchmark spells its own flag differently (--few-shot N for NormAd,
# --neutral-shots N for CulturalBench, --neutral-fewshot for BLEnD, which has
# no count) but the count means the same thing, so one argument drives all three.
SHOTS=2
# --raw-prompt is the default: it bypasses the chat template so an instruct
# checkpoint is scored byte-identically to a base model. --templated is the
# diagnostic arm -- it keeps the chat template, which is how these models are
# actually deployed, and is the way to tell a genuine yes/no bias apart from an
# artifact of scoring an aligned model as a completion model. It pairs with
# --system-prompt neutral, because Olmo-3-Instruct's template otherwise injects
# a function-calling system prompt that no other model in the table sees.
TEMPLATED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)     MODELS_ARG="$2"; shift 2 ;;
        --benchmarks) BENCHES_ARG="$2"; shift 2 ;;
        --condition)  COND_OVERRIDE="$2"; shift 2 ;;
        --shots)      SHOTS="$2"; shift 2 ;;
        --templated)  TEMPLATED=1; shift ;;
        --nocountry)  NOCOUNTRY=1; shift ;;
        --split-models) SPLIT_MODELS=1; shift ;;
        --layout)     LAYOUT=1; shift ;;
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
IFS=',' read -r -a BENCHES <<< "$BENCHES_ARG"

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
# One "model group" per task when splitting, otherwise a single group holding
# every model -- which is what keeps the index arithmetic identical either way.
if [[ $SPLIT_MODELS -eq 1 ]]; then N_GROUPS=$N_MODELS; else N_GROUPS=1; fi
N_TASKS=$(( N_BENCH * N_GROUPS ))

# task id -> (benchmark, model group). Benchmark is most significant so a
# benchmark's tasks stay contiguous, which is what --array subsets index into.
decompose() {
    IDX_BENCH=$(( $1 / N_GROUPS ))
    IDX_GROUP=$(( $1 % N_GROUPS ))
}

# The models this task runs: one when splitting, all of them otherwise.
models_for_task() {
    if [[ $SPLIT_MODELS -eq 1 ]]; then echo "$IDX_GROUP"; return; fi
    seq 0 $(( N_MODELS - 1 ))
}

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

# Must match the suffixes the eval scripts build for the same flags, since the
# existence check below and the probe job's --base both go through this.
# The eval scripts append _raw only when --raw-prompt actually changes
# anything, i.e. when the model label is chat-templated (raw_sfx in
# eval_normad.py). A label like plain "8b" has no chat template, so its output
# carries no _raw tail and predicting one here would make the skip check and the
# probe job's --base both point at a file that is never written. Read from
# evaluate/_common.py rather than restating the set, and without importing it:
# _common pulls in torch and peft, and --layout has to run on a login node.
raw_tail_for() {
    local label="$1" cond="${2:-}"
    # --templated passes no --raw-prompt, so raw_sfx is empty; it passes
    # --system-prompt instead, which appends _sys after where _raw would sit.
    if [[ $TEMPLATED -eq 1 ]]; then
        # Unconditional, unlike _raw: sys_sfx in the eval scripts is gated only
        # on --system-prompt being set, not on the label being chat-templated.
        # So a label with no chat template (plain "8b") still gets a _sys name
        # even though the system message has no effect on it.
        echo "_sys"
        return
    fi
    # Both sets, because a condition can be chat-templated over a base-model
    # label: tulu3_dpo is size "8b", whose own label has no chat template.
    if sed -n '/^CHAT_TEMPLATED_SIZES = {/,/^}/p' evaluate/_common.py \
         | grep -q "\"${label}\","; then
        echo "_raw"
    elif [[ -n "$cond" ]] && \
         sed -n '/^CHAT_TEMPLATED_CONDITIONS = {/,/^}/p' evaluate/_common.py \
           | grep -q "\"${cond}\","; then
        echo "_raw"
    fi
}

out_for() {
    local bench="$1" cond="$2" model="$3"
    local raw; raw=$(raw_tail_for "$model" "$cond")
    # _nocountry sits immediately before the _raw tail, matching the order the
    # eval scripts assemble their own suffixes in.
    local nc=""; [[ $NOCOUNTRY -eq 1 ]] && nc="_nocountry"
    local fs="" nfs=""
    if (( SHOTS > 0 )); then
        fs="_fs${SHOTS}"
        # eval_culturalbench leaves _nfs bare at the historical 2 shots so older
        # filenames keep matching, and only numbers it from 3 up.
        if (( SHOTS == 2 )); then nfs="_nfs"; else nfs="_nfs${SHOTS}"; fi
    fi
    case "$bench" in
        normad)        echo "outputs/behavioral/normad_${cond}_${model}${fs}_yn_wp${nc}${raw}.json" ;;
        culturalbench) echo "outputs/behavioral/culturalbench_${cond}_${model}${nfs}${nc}${raw}.json" ;;
        # BLEnD's neutral prefix has no count in its name: it is a fixed set of
        # four MCQ demos, one per answer letter, so it is on or off.
        # cbeasy shares BLEnD's prompt and so BLEnD's prefix: four fixed
        # MCQ-format shots, on or off, with no count in the name.
        cbeasy)        echo "outputs/behavioral/cbeasy_${cond}_${model}$( (( SHOTS > 0 )) && echo _nfs )${nc}${raw}.json" ;;
        blend)         echo "outputs/behavioral/blend_${cond}_${model}$( (( SHOTS > 0 )) && echo _nfs )${nc}${raw}.json" ;;
        *) echo "unknown benchmark: $bench" >&2; return 1 ;;
    esac
}

if [[ $LAYOUT -eq 1 ]]; then
    echo "tasks: $N_TASKS   (submit with --array=0-$(( N_TASKS - 1 )))"
    for (( t = 0; t < N_TASKS; t++ )); do
        decompose "$t"
        b=${BENCHES[$IDX_BENCH]}
        printf '  %3d  %s\n' "$t" "$b"
        for mi in $(models_for_task); do
            c=$(cond_at "$mi")
            printf '       %-9s %-16s -> %s\n' \
                   "$c" "${MODELS[$mi]}" "$(out_for "$b" "$c" "${MODELS[$mi]}")"
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

echo "[inst_base] benchmark=$BENCH shots=$SHOTS nocountry=$NOCOUNTRY" \
     "templated=$TEMPLATED models=$(for i in $(models_for_task); do \
        printf '%s ' "${MODELS[$i]}"; done)"

# Flag arrays rather than a copy of each command per arm, so the arms cannot
# drift in any other flag -- which is exactly how the probe pass ended up
# scoring a different template from the baseline.
NC_FLAG=()
[[ $NOCOUNTRY -eq 1 ]] && NC_FLAG=(--no-country)
FMT_FLAG=(--raw-prompt)
[[ $TEMPLATED -eq 1 ]] && FMT_FLAG=(--system-prompt neutral)

run_one() {
    local bench="$1" cond="$2" model="$3"
    case "$bench" in
      normad)
        local SHOT_FLAG=(); (( SHOTS > 0 )) && SHOT_FLAG=(--few-shot "$SHOTS")
        python evaluate/eval_normad.py --condition "$cond" --model-size "$model" \
            --precision matched_bf16 --yn-only "${FMT_FLAG[@]}" \
            "${SHOT_FLAG[@]}" "${NC_FLAG[@]}"
        ;;
      culturalbench)
        # --no-fewshot, not "--neutral-shots 0": neutral_fewshot defaults to True
        # in evaluate_one, so the prefix has to be switched off explicitly.
        local SHOT_FLAG=(--no-fewshot)
        (( SHOTS > 0 )) && SHOT_FLAG=(--neutral-shots "$SHOTS")
        python evaluate/eval_culturalbench.py --condition "$cond" --model-size "$model" \
            --precision matched_bf16 "${FMT_FLAG[@]}" \
            "${SHOT_FLAG[@]}" "${NC_FLAG[@]}"
        ;;
      cbeasy)
        local SHOT_FLAG=(); (( SHOTS > 0 )) && SHOT_FLAG=(--neutral-fewshot)
        python evaluate/eval_culturalbench_easy.py --condition "$cond" --model-size "$model" \
            --precision matched_bf16 "${FMT_FLAG[@]}" \
            "${SHOT_FLAG[@]}" "${NC_FLAG[@]}"
        ;;
      blend)
        local SHOT_FLAG=(); (( SHOTS > 0 )) && SHOT_FLAG=(--neutral-fewshot)
        python evaluate/eval_blend.py --condition "$cond" --model-size "$model" \
            --precision matched_bf16 "${FMT_FLAG[@]}" \
            "${SHOT_FLAG[@]}" "${NC_FLAG[@]}"
        ;;
    esac
}

# One model failing must not cost the rest of the task its queue slot, so the
# loop records the failure and carries on, and the task exits non-zero at the
# end so sacct still shows FAILED rather than a silent partial grid.
FAILED=()
for mi in $(models_for_task); do
    MODEL=${MODELS[$mi]}
    COND=$(cond_at "$mi")
    OUT=$(out_for "$BENCH" "$COND" "$MODEL")
    if [[ -f "$OUT" ]]; then
        echo "[inst_base] $OUT exists - skipping"
        continue
    fi
    echo "[inst_base] === $BENCH / $COND / $MODEL -> $OUT"
    if ! run_one "$BENCH" "$COND" "$MODEL"; then
        echo "[inst_base] FAILED: $BENCH / $COND / $MODEL" >&2
        FAILED+=("$COND/$MODEL")
    fi
done

if (( ${#FAILED[@]} )); then
    echo "[inst_base] $BENCH: ${#FAILED[@]} failed: ${FAILED[*]}" >&2
    exit 1
fi
echo "[inst_base] $BENCH: done"
