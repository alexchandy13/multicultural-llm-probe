#!/bin/bash
#SBATCH --job-name=cult_probe_8b
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa4000:2
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --output=slurm/eval_cluster_probe_8b.%A_%a.out
#SBATCH --error=slurm/eval_cluster_probe_8b.%A_%a.err

# Cluster-representative probe eval for LLaMA 3.1 8B on NormAd.
# Uses add_probe.py — reads pred from existing usprobe file, runs probe pass only.
# Array: 5 conditions × 2 countries = 10 tasks (0-9)
# Submit with: sbatch slurm/eval_cluster_probe_8b_job.sh

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

BASE="outputs/behavioral/normad_${COND}_8b_nfs_mpw_usprobe.json"

echo "[cluster_probe_8b] condition=$COND  probe_country=$COUNTRY"

python evaluate/add_probe.py \
    --base "$BASE" \
    --probe-country "$COUNTRY" \
    --condition "$COND" \
    --model-size 8b
