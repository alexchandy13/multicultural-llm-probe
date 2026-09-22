#!/bin/bash
#SBATCH --job-name=normy_probe_g4
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --output=slurm/eval_culturalbench_normy_probe_gemma4.%A_%a.out
#SBATCH --error=slurm/eval_culturalbench_normy_probe_gemma4.%A_%a.err

# CulturalBench-Normy cluster-representative probe eval for Gemma 4 12B.
# Uses add_probe.py — reads pred from existing usprobe file, runs probe pass only.
# Array: 5 conditions × 2 countries = 10 tasks (0-9)
# Depends on eval_culturalbench_normy_gemma4_job.sh completing first.
# Submit with: sbatch slurm/eval_culturalbench_normy_probe_gemma4_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

CONDS=(base sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult)
COUNTRIES=(taiwan iran)

N_COUNTRIES=${#COUNTRIES[@]}
COND_IDX=$(( SLURM_ARRAY_TASK_ID / N_COUNTRIES ))
COUNTRY_IDX=$(( SLURM_ARRAY_TASK_ID % N_COUNTRIES ))

COND=${CONDS[$COND_IDX]}
COUNTRY=${COUNTRIES[$COUNTRY_IDX]}

BASE="outputs/behavioral/culturalbench_normy_${COND}_gemma4_nfs_mpw_usprobe.json"

echo "[culturalbench_normy_probe_gemma4] condition=$COND  probe_country=$COUNTRY"

python evaluate/add_probe.py \
    --base "$BASE" \
    --probe-country "$COUNTRY" \
    --condition "$COND" \
    --model-size gemma4
