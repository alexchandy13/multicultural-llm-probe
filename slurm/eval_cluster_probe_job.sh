#!/bin/bash
#SBATCH --job-name=culture_eval_cluster_probe
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --time=20:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-34
#SBATCH --output=slurm/eval_cluster_probe.%A_%a.out
#SBATCH --error=slurm/eval_cluster_probe.%A_%a.err

# Cluster-representative probe eval for Llama 3.1 8B.
# For each of 8 IW cluster centroid-representative countries, re-scores every
# NormAd example with that country substituted in as the probe. Produces files:
#   normad_{cond}_8b_nfs_mpw_{country}probe.json
#
# Array: 5 conditions × 8 countries = 40 tasks (0-39)
# Submit with: sbatch slurm/eval_cluster_probe_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

CONDS=(base sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult)
COUNTRIES=(netherlands hungary bosnia_and_herzegovina mexico taiwan philippines iran)

N_CONDS=${#CONDS[@]}
N_COUNTRIES=${#COUNTRIES[@]}

COND_IDX=$(( SLURM_ARRAY_TASK_ID / N_COUNTRIES ))
COUNTRY_IDX=$(( SLURM_ARRAY_TASK_ID % N_COUNTRIES ))

COND=${CONDS[$COND_IDX]}
COUNTRY=${COUNTRIES[$COUNTRY_IDX]}

echo "[cluster_probe] condition=$COND  probe_country=$COUNTRY"

python evaluate/eval_normad.py \
    --condition "$COND" \
    --model-size 8b \
    --neutral-fewshot \
    --multi-prompt-word \
    --probe-country "$COUNTRY"
