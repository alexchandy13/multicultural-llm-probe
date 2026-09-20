#!/bin/bash
#SBATCH --job-name=cult_probe_blend_8b
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-34
#SBATCH --output=slurm/eval_cluster_probe_blend_8b.%A_%a.out
#SBATCH --error=slurm/eval_cluster_probe_blend_8b.%A_%a.err

# Cluster-representative probe eval for LLaMA 3.1 8B on BLEnD.
# Array: 5 conditions × 7 countries = 35 tasks (0-34)
# Submit with: sbatch slurm/eval_cluster_probe_blend_8b_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

CONDS=(base sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult)
COUNTRIES=(netherlands hungary bosnia_and_herzegovina mexico taiwan philippines iran)

N_COUNTRIES=${#COUNTRIES[@]}

COND_IDX=$(( SLURM_ARRAY_TASK_ID / N_COUNTRIES ))
COUNTRY_IDX=$(( SLURM_ARRAY_TASK_ID % N_COUNTRIES ))

COND=${CONDS[$COND_IDX]}
COUNTRY=${COUNTRIES[$COUNTRY_IDX]}

echo "[cluster_probe_blend_8b] condition=$COND  probe_country=$COUNTRY"

python evaluate/eval_blend.py \
    --condition "$COND" \
    --model-size 8b \
    --neutral-fewshot \
    --probe-country "$COUNTRY"
