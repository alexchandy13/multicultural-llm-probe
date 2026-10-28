#!/bin/bash
#SBATCH --job-name=normad_regrid
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=2:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-15
#SBATCH --output=slurm/eval_normad_regrid.%A_%a.out
#SBATCH --error=slurm/eval_normad_regrid.%A_%a.err

# Re-run the NormAd _fs2_yn grid under the revised YN_PROMPT_TEMPLATE, which
# drops the duplicated "Question: Is this action socially acceptable in this
# culture?" line (the story already asks it) and adds an instruction paragraph.
#
# 2 conditions (sftdpo_aya_nocult, tulu3_dpo) x 8 probe countries = 16
# tasks. Tulu-3 is 8b-only, so the Gemma job runs the one condition.
# The "us" probe uses --us-probe; the other seven use --probe-country, which
# stores its swapped prediction in us_pred for compatibility.
#
# These OVERWRITE the existing _fs2_yn files, which were produced with the old
# template. Back them up first if the old numbers are still wanted:
#   mkdir -p outputs/behavioral/_fs2yn_oldtemplate
#   cp outputs/behavioral/normad_*_fs2_yn_*probe.json \
#      outputs/behavioral/_fs2yn_oldtemplate/
#
# 1,697 examples x 2 passes (real + probe) per run, ~4 min on 8b.
#
# Submit with: sbatch slurm/eval_normad_fs2yn_regrid_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

CONDS=(sftdpo_aya_nocult tulu3_dpo)
PROBES=(us netherlands hungary bosnia_and_herzegovina mexico taiwan philippines iran)

N_PROBES=${#PROBES[@]}
COND=${CONDS[$(( SLURM_ARRAY_TASK_ID / N_PROBES ))]}
PROBE=${PROBES[$(( SLURM_ARRAY_TASK_ID % N_PROBES ))]}

echo "[normad_regrid] condition=$COND probe=$PROBE model=8b"

if [ "$PROBE" = "us" ]; then
    PROBE_ARG=(--us-probe)
else
    PROBE_ARG=(--probe-country "$PROBE")
fi

python evaluate/eval_normad.py \
    --condition "$COND" \
    --model-size 8b \
    --precision matched_bf16 \
    --few-shot 2 \
    --yn-only \
    "${PROBE_ARG[@]}"
