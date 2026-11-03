#!/bin/bash
#SBATCH --job-name=inst_probe
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --array=0-26
#SBATCH --output=slurm/eval_inst_probe.%A_%a.out
#SBATCH --error=slurm/eval_inst_probe.%A_%a.err

# Cluster-representative probe grid for the released instruct models.
#
# add_probe.py reads `pred` from the _usprobe.json and runs the swap pass only,
# so this is one forward pass per row per country rather than two.
#
# TASK LAYOUT is 9 tasks per model, uneven on purpose. --probe-country takes
# nargs="+", and several countries sharing one model load is most of the saving
# on a one-pass job -- worth it for NormAd (1,697 rows) and CulturalBench
# (4,905), where all 7 finish in minutes. Not viable for BLEnD: 23,032 rows runs
# ~3-4 h per country at 32B, so 7 in one task would blow any walltime this
# partition grants, and a TIMEOUT is not requeued the way a preemption is.
#   offset 0    normad         all 7 countries
#   offset 1    culturalbench  all 7 countries
#   offset 2-8  blend          one country each
#
# US is absent from COUNTRIES: it IS the base file, recorded there as us_pred.
#
# --requeue plus --skip-existing means a preempted task resumes at the first
# missing country instead of redoing the grid.
#
# DEPENDENCY: needs eval_instruct_usprobe_job.sh to have written the base file.
#   J=$(sbatch --parsable slurm/eval_instruct_usprobe_job.sh)
#   sbatch --dependency=afterok:$J slurm/eval_instruct_probe_job.sh
#
# 3 models x 9 = 27 tasks.
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

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODELS=(olmo3_instruct qwen35_instruct gptoss20b)
COUNTRIES=(netherlands hungary bosnia_and_herzegovina mexico taiwan philippines iran)

PER_MODEL=9
MODEL=${MODELS[$(( SLURM_ARRAY_TASK_ID / PER_MODEL ))]}
OFFSET=$(( SLURM_ARRAY_TASK_ID % PER_MODEL ))

case "$OFFSET" in
  0) BENCH=normad;        BASE="outputs/behavioral/normad_base_${MODEL}_fs2_yn_usprobe.json"
     PROBES=("${COUNTRIES[@]}") ;;
  1) BENCH=culturalbench; BASE="outputs/behavioral/culturalbench_base_${MODEL}_nfs4_usprobe.json"
     PROBES=("${COUNTRIES[@]}") ;;
  *) BENCH=blend;         BASE="outputs/behavioral/blend_base_${MODEL}_nfs_usprobe.json"
     PROBES=("${COUNTRIES[$(( OFFSET - 2 ))]}") ;;
esac

if [[ ! -f "$BASE" ]]; then
    echo "[inst_probe] missing base $BASE — run eval_instruct_usprobe_job.sh first" >&2
    exit 1
fi

echo "[inst_probe] model=$MODEL benchmark=$BENCH probes=${PROBES[*]}"

python evaluate/add_probe.py \
    --base "$BASE" \
    --probe-country "${PROBES[@]}" \
    --condition offshelf \
    --model-size "$MODEL" \
    --precision matched_bf16 \
    --raw-prompt \
    --skip-existing
