#!/bin/bash
#SBATCH --job-name=ots_nc_g4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --time=2:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5
#SBATCH --output=slurm/eval_ots_nc_g4.%A_%a.out
#SBATCH --error=slurm/eval_ots_nc_g4.%A_%a.err

# Country ablation for the OFF-THE-SHELF Gemma releases, on all three
# benchmarks. condition=base means no adapter of ours is loaded at all:
#   gemma4           google/gemma-4-12B      (pretrained only)
#   gemma4_instruct  google/gemma-4-12b-it   (Google's own post-training)
#
# The pretrained arm already has all 8 probe-country files for each benchmark
# (run Oct 8); the country-free prediction they pair against was never run, and
# per_probe_country() cannot be computed without it. The instruct arm needs the
# probe grid too -- see eval_offtheshelf_usprobe_job.sh and
# eval_offtheshelf_probe_job.sh, which must run in that order.
#
# --raw-prompt on every task: it bypasses the chat template on the *_instruct
# sizes so their prompts are byte-identical to the pretrained ones'. This is
# what makes the arm comparable to the rest of the results table, where
# sftdpo_aya_* (LoRA on pretrained Llama) and tulu3_sft/tulu3_dpo (full
# post-trained releases) are ALL scored as flat strings, because is_instruct()
# keys off the size label and both run at model_size=8b. Without the flag the
# vendor releases would be the only chat-templated cells in the table.
# The flag is guarded by is_instruct(), so it is a no-op -- in behaviour and in
# filename -- on the pretrained sizes, and instruct outputs gain a _raw tail.
#
# Flags per benchmark match the suffix the existing probe grid was run at, so
# the files pair by row order: _fs2_yn, _nfs4, _nfs. Getting these wrong writes
# a file whose name looks right but whose prompts differ from the probe files.
#
# With --raw-prompt the only thing differing between the pretrained and
# instruct cells is the weights, which is the point of the comparison. To
# measure the prompt-format effect instead, rerun without the flag: the
# instruct outputs then land at the same names minus _raw.
#
# 3 benchmarks x 2 sizes = 6 tasks. Gemma-4-12B is ~24 GB in bf16, too tight
# for one A5000, so every task requests two cards. BLEnD's 23k examples
# dominate the walltime.
#
# Submit with: sbatch slurm/eval_offtheshelf_nocountry_gemma4_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

BENCHES=(normad culturalbench blend)
SIZES=(gemma4 gemma4_instruct)

N_BENCH=${#BENCHES[@]}
BENCH=${BENCHES[$(( SLURM_ARRAY_TASK_ID % N_BENCH ))]}
MODEL=${SIZES[$(( SLURM_ARRAY_TASK_ID / N_BENCH ))]}

echo "[ots_nc_g4] benchmark=$BENCH condition=base model=$MODEL"

case "$BENCH" in
  normad)
    python evaluate/eval_normad.py \
        --condition base --model-size "$MODEL" --precision matched_bf16 \
        --few-shot 2 --yn-only --no-country --raw-prompt
    ;;
  culturalbench)
    python evaluate/eval_culturalbench.py \
        --condition base --model-size "$MODEL" --precision matched_bf16 \
        --neutral-shots 4 --no-country --raw-prompt
    ;;
  blend)
    python evaluate/eval_blend.py \
        --condition base --model-size "$MODEL" --precision matched_bf16 \
        --neutral-fewshot --no-country --raw-prompt
    ;;
esac
