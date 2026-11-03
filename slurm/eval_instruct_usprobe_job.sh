#!/bin/bash
#SBATCH --job-name=inst_usprobe
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --array=0-8
#SBATCH --output=slurm/eval_inst_usprobe.%A_%a.out
#SBATCH --error=slurm/eval_inst_usprobe.%A_%a.err

# Own-country + US-swap pass for the released instruct models.
#
# Step 1 of 2 for the probe grid: add_probe.py reads `pred` (the own-country
# answer) out of a _usprobe.json and runs only the swap pass, so every probe
# country needs this file first. Two passes per row here against one per row
# per country there, which is why this asks for 12 h and the probe grid 8.
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
# Skips and exits 0 when its output already exists, so re-submitting to fill
# gaps costs one model load per task.
#
# 3 models x 3 benchmarks = 9 tasks. The BLEnD tasks are the tight ones: two
# passes over 23k rows at 32B is ~6-8 h. If sbatch rejects 12 h the partition
# cap is lower, and BLEnD needs --sample-frac support in eval_blend.py.
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

case "$BENCH" in
  normad)        OUT="outputs/behavioral/normad_offshelf_${MODEL}_fs2_yn_usprobe_raw.json" ;;
  culturalbench) OUT="outputs/behavioral/culturalbench_offshelf_${MODEL}_nfs4_usprobe_raw.json" ;;
  blend)         OUT="outputs/behavioral/blend_offshelf_${MODEL}_nfs_usprobe_raw.json" ;;
esac
if [[ -f "$OUT" ]]; then echo "[inst_usprobe] $OUT exists - nothing to do"; exit 0; fi

echo "[inst_usprobe] benchmark=$BENCH condition=offshelf model=$MODEL"

case "$BENCH" in
  normad)
    python evaluate/eval_normad.py --condition offshelf --model-size "$MODEL" \
        --precision matched_bf16 --few-shot 2 --yn-only --us-probe --raw-prompt
    ;;
  culturalbench)
    python evaluate/eval_culturalbench.py --condition offshelf --model-size "$MODEL" \
        --precision matched_bf16 --neutral-shots 4 --us-probe --raw-prompt
    ;;
  blend)
    python evaluate/eval_blend.py --condition offshelf --model-size "$MODEL" \
        --precision matched_bf16 --neutral-fewshot --us-probe --raw-prompt
    ;;
esac
