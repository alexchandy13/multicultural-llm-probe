#!/bin/bash
#SBATCH --job-name=inst_nocountry
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --array=0-8
#SBATCH --output=slurm/eval_inst_nocountry.%A_%a.out
#SBATCH --error=slurm/eval_inst_nocountry.%A_%a.err

# Country ablation for the released instruct models, all three benchmarks.
# condition=offshelf throughout -- no adapter of ours is loaded; the arm is carried
# entirely by model_size.
#
# Tier: models whose bf16 weights fit two A6000s (96 GB).
#   olmo3_instruct   allenai/Olmo-3-7B-Instruct        ~14 GB
#   qwen35_instruct  Qwen/Qwen3.5-9B-Instruct          ~18 GB
#   gptoss20b        openai/gpt-oss-20b                ~42 GB once bf16-upcast
#
# This is the PILOT. Llama-3.1-8B-Instruct and gemma-4-12b-it are absent on
# purpose: the eval_offtheshelf_* jobs already cover them raw-prompted, so
# repeating them here would differ only in the condition slug in the
# filename. The analysis reads both slugs.
# These three are the smallest member of each remaining family, so unresolvable model
# ids, degenerate output and chat-template surprises all surface cheaply before
# the 27-70B tier. To scale up, swap MODELS for the larger siblings
# (qwen38_27b gemma4_31b olmo3_32b) and refit --array to 3 or 9 per model.
# device_map="auto" shards over whatever cards the task gets, so one array
# covers all five. Llama-70B needs ~141 GB and has its own 4-card variant.
#
# rtxa6000 rather than h200/b300: nine 8-card A6000 nodes on scavenger against
# one B300 and three H200s, so this schedules instead of queueing. B300 is also
# sm_100, which an older torch build cannot target at all.
#
# 3 models x 3 benchmarks = 9 tasks. BLEnD's 23k rows set the walltime; at 32B
# one pass is ~3-4 h against ~1.5 h for a 12B.
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
# Submit with: sbatch slurm/eval_instruct_nocountry_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODELS=(olmo3_instruct qwen35_instruct gptoss20b)
BENCHES=(normad culturalbench blend)

N_BENCH=${#BENCHES[@]}
BENCH=${BENCHES[$(( SLURM_ARRAY_TASK_ID % N_BENCH ))]}
MODEL=${MODELS[$(( SLURM_ARRAY_TASK_ID / N_BENCH ))]}

# The eval scripts have no checkpointing, so a preempted task restarts from
# zero. This at least stops a requeue redoing a task that already finished --
# which matters because BLEnD is ~1.3-2.5 h and scavenger preempts.
case "$BENCH" in
  normad)        OUT="outputs/behavioral/normad_offshelf_${MODEL}_fs2_yn_nocountry_raw.json" ;;
  culturalbench) OUT="outputs/behavioral/culturalbench_offshelf_${MODEL}_nfs4_nocountry_raw.json" ;;
  blend)         OUT="outputs/behavioral/blend_offshelf_${MODEL}_nfs_nocountry_raw.json" ;;
esac
if [[ -f "$OUT" ]]; then echo "[inst_nocountry] $OUT exists - nothing to do"; exit 0; fi

echo "[inst_nocountry] benchmark=$BENCH condition=offshelf model=$MODEL"

case "$BENCH" in
  normad)
    python evaluate/eval_normad.py --condition offshelf --model-size "$MODEL" \
        --precision matched_bf16 --few-shot 2 --yn-only --no-country --raw-prompt
    ;;
  culturalbench)
    python evaluate/eval_culturalbench.py --condition offshelf --model-size "$MODEL" \
        --precision matched_bf16 --neutral-shots 4 --no-country --raw-prompt
    ;;
  blend)
    python evaluate/eval_blend.py --condition offshelf --model-size "$MODEL" \
        --precision matched_bf16 --neutral-fewshot --no-country --raw-prompt
    ;;
esac
