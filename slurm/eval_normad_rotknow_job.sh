#!/bin/bash
#SBATCH --job-name=normad_rotknow
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=2:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-23
#SBATCH --output=slurm/eval_normad_rotknow.%A_%a.out
#SBATCH --error=slurm/eval_normad_rotknow.%A_%a.err

# Ask about the Rule-of-Thumb with no scenario, to compare against the
# standard scenario eval.
#
#   APPLY   country + scenario   -> yes/no    already run (_fs2_yn)
#   KNOW    country + norm       -> yes/no    this job
#
# Same template with "Scenario:" swapped for "Norm:", same 1,697 examples in
# the same order, so the two cross per example.
#
# Plus the country swap: the same norm asked again with the country replaced,
# stored in us_pred exactly as the scenario arm stores it. Same 8 probe
# countries, one per IW cluster, so the grids line up cell for cell.
#
# 2 passes per example, 3,394 per task, ~6 min on 8b.
#
# 3 conditions x 8 probe countries = 24 tasks. Tulu-3 is 8b-only, so the Gemma job runs the other two.
#
# Submit with: sbatch slurm/eval_normad_rotknow_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

CONDS=(sftdpo_aya_cult sftdpo_aya_nocult tulu3_dpo)
PROBES=(us netherlands hungary bosnia_and_herzegovina mexico taiwan philippines iran)
N_PROBES=${#PROBES[@]}

COND=${CONDS[$(( SLURM_ARRAY_TASK_ID / N_PROBES ))]}
PROBE=${PROBES[$(( SLURM_ARRAY_TASK_ID % N_PROBES ))]}

if [ "$PROBE" = "us" ]; then PROBE_ARG=(--us-probe)
else PROBE_ARG=(--probe-country "$PROBE"); fi

echo "[normad_rotknow] condition=$COND probe=$PROBE model=8b"

python evaluate/eval_normad_rot_knowledge.py \
    --condition "$COND" \
    --model-size 8b \
    --precision matched_bf16 \
    "${PROBE_ARG[@]}"
