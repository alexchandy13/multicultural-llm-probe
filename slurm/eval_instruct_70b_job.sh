#!/bin/bash
#SBATCH --job-name=inst_70b
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --time=8:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=slurm/eval_inst_70b.%A_%a.out
#SBATCH --error=slurm/eval_inst_70b.%A_%a.err

# Llama-3.1-70B-Instruct: the "does it survive scale" cell. ~141 GB in bf16, so
# four A6000s; device_map="auto" shards it. A6000 because scavenger has nine
# 8-card A6000 nodes against one B300 and three H200s.
#
# NormAd and CulturalBench ONLY. BLEnD is 23,032 rows and a 70B runs roughly
# 4-6x slower per forward pass than a 12B, putting its two-pass us-probe
# baseline at 12-18 h -- past any walltime here, and a TIMEOUT is not requeued.
# add_probe.py already has --sample-frac for exactly this, but eval_blend.py
# does not, so the baseline cannot be subsampled yet. BLEnD at 70B is therefore
# out of scope on purpose rather than silently timing out.
#
# All three stages live in one array so the dependency chain stays short:
#   task 0,1  no-country        normad, culturalbench
#   task 2,3  own + US swap     normad, culturalbench
#   task 4,5  7-country probes  normad, culturalbench
# Tasks 4,5 need 2,3 to have finished, so submit as two steps -- see below.
#
# Sizes here are small: NormAd 1,697 rows and CulturalBench 4,905, so even at
# 4-6x slower these are well inside 8 h.
#
# --raw-prompt on every task: the chat template is bypassed so each model sees a
# flat string ending in "Answer:", byte-identical across models. Two reasons.
# It matches the Llama/Gemma cells already running, so every cell in the table
# was queried the same way. And it sidesteps the thinking templates -- Qwen3
# enables a reasoning mode by default and gpt-oss emits a reasoning channel, so
# under the chat template the token after "Answer:" can be a thought opener
# rather than yes/no, and scoring would read the wrong position.
#
# The cost is that an instruct model on a flat prompt is off-distribution, so
# accuracy is understated. Agreement rates are the metric here, not accuracy.
# Outputs gain a _raw tail, so a templated rerun lands beside them, not on them.
#
# Submit with:
#   J=$(sbatch --parsable --array=0-3 slurm/eval_instruct_70b_job.sh)
#   sbatch --dependency=afterok:$J --array=4-5 slurm/eval_instruct_70b_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODEL=llama70b
COUNTRIES=(netherlands hungary bosnia_and_herzegovina mexico taiwan philippines iran)

STAGE=$(( SLURM_ARRAY_TASK_ID / 2 ))     # 0 nocountry, 1 usprobe, 2 probe
WHICH=$(( SLURM_ARRAY_TASK_ID % 2 ))     # 0 normad, 1 culturalbench
BENCH=$([ "$WHICH" -eq 0 ] && echo normad || echo culturalbench)

if [[ "$BENCH" == "normad" ]]; then
    BASE="outputs/behavioral/normad_base_${MODEL}_fs2_yn_usprobe.json"
else
    BASE="outputs/behavioral/culturalbench_base_${MODEL}_nfs4_usprobe.json"
fi

echo "[inst_70b] stage=$STAGE benchmark=$BENCH model=$MODEL"

case "$STAGE" in
  0)
    if [[ "$BENCH" == "normad" ]]; then
        python evaluate/eval_normad.py --condition offshelf --model-size "$MODEL" \
            --precision matched_bf16 --few-shot 2 --yn-only --no-country --raw-prompt
    else
        python evaluate/eval_culturalbench.py --condition offshelf --model-size "$MODEL" \
            --precision matched_bf16 --neutral-shots 4 --no-country --raw-prompt
    fi
    ;;
  1)
    if [[ -f "$BASE" ]]; then echo "[inst_70b] $BASE exists - nothing to do"; exit 0; fi
    if [[ "$BENCH" == "normad" ]]; then
        python evaluate/eval_normad.py --condition offshelf --model-size "$MODEL" \
            --precision matched_bf16 --few-shot 2 --yn-only --us-probe --raw-prompt
    else
        python evaluate/eval_culturalbench.py --condition offshelf --model-size "$MODEL" \
            --precision matched_bf16 --neutral-shots 4 --us-probe --raw-prompt
    fi
    ;;
  2)
    if [[ ! -f "$BASE" ]]; then
        echo "[inst_70b] missing base $BASE - run tasks 2-3 first" >&2; exit 1
    fi
    python evaluate/add_probe.py --base "$BASE" \
        --probe-country "${COUNTRIES[@]}" \
        --condition offshelf --model-size "$MODEL" \
        --precision matched_bf16 --raw-prompt --skip-existing
    ;;
esac
