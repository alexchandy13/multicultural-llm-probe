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
#SBATCH --array=0-14
#SBATCH --output=slurm/eval_inst_nocountry.%A_%a.out
#SBATCH --error=slurm/eval_inst_nocountry.%A_%a.err

# Country ablation for the released instruct models, all three benchmarks.
# condition=offshelf throughout -- no adapter of ours is loaded; the arm is carried
# entirely by model_size.
#
# Tier: models whose bf16 weights fit two A6000s (96 GB).
#   olmo3_instruct   allenai/Olmo-3-7B-Instruct        ~14 GB
#   qwen35_instruct  Qwen/Qwen3.5-9B-Instruct          ~18 GB
#   8b_instruct      meta-llama/Llama-3.1-8B-Instruct  ~16 GB  (cached)
#   gemma4_instruct  google/gemma-4-12b-it             ~24 GB  (cached)
#   gptoss20b        openai/gpt-oss-20b                ~42 GB once bf16-upcast
#
# This is the PILOT: the smallest member of each family, so unresolvable model
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
# NO --raw-prompt. These are measured through their chat templates, which is how
# they ship and how anyone uses them. The flag remains for the separate question
# of how much of a base-vs-instruct gap is prompt format rather than weights.
#
# 5 models x 3 benchmarks = 15 tasks. BLEnD's 23k rows set the walltime; at 32B
# one pass is ~3-4 h against ~1.5 h for a 12B.
#
# Submit with: sbatch slurm/eval_instruct_nocountry_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODELS=(olmo3_instruct qwen35_instruct 8b_instruct gemma4_instruct gptoss20b)
BENCHES=(normad culturalbench blend)

N_BENCH=${#BENCHES[@]}
BENCH=${BENCHES[$(( SLURM_ARRAY_TASK_ID % N_BENCH ))]}
MODEL=${MODELS[$(( SLURM_ARRAY_TASK_ID / N_BENCH ))]}

echo "[inst_nocountry] benchmark=$BENCH condition=offshelf model=$MODEL"

case "$BENCH" in
  normad)
    python evaluate/eval_normad.py --condition offshelf --model-size "$MODEL" \
        --precision matched_bf16 --few-shot 2 --yn-only --no-country
    ;;
  culturalbench)
    python evaluate/eval_culturalbench.py --condition offshelf --model-size "$MODEL" \
        --precision matched_bf16 --neutral-shots 4 --no-country
    ;;
  blend)
    python evaluate/eval_blend.py --condition offshelf --model-size "$MODEL" \
        --precision matched_bf16 --neutral-fewshot --no-country
    ;;
esac
