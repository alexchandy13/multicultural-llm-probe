#!/bin/bash
#SBATCH --job-name=cult_probe_sftdpo
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --exclusive
#SBATCH --cpus-per-task=4
#SBATCH --array=0-13
#SBATCH --output=slurm/eval_cluster_probe_sftdpo.%A_%a.out
#SBATCH --error=slurm/eval_cluster_probe_sftdpo.%A_%a.err

# Cluster-representative probe for sftdpo conditions, using add_probe.py.
# Reads existing pred/scores from the usprobe file and only runs the probe-country
# pass — half the GPU memory of a full re-eval.
#
# Array: 2 conditions × 7 countries = 14 tasks (0-13)
# Submit with: sbatch slurm/eval_cluster_probe_sftdpo_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONDS=(sftdpo_aya_cult sftdpo_aya_nocult)
COUNTRIES=(netherlands hungary bosnia_and_herzegovina mexico taiwan philippines iran)

N_CONDS=${#CONDS[@]}
N_COUNTRIES=${#COUNTRIES[@]}

COND_IDX=$(( SLURM_ARRAY_TASK_ID / N_COUNTRIES ))
COUNTRY_IDX=$(( SLURM_ARRAY_TASK_ID % N_COUNTRIES ))

COND=${CONDS[$COND_IDX]}
COUNTRY=${COUNTRIES[$COUNTRY_IDX]}

BASE="outputs/behavioral/normad_${COND}_8b_nfs_mpw_usprobe.json"

echo "[cluster_probe_sftdpo] condition=$COND  probe_country=$COUNTRY"
echo "  base=$BASE"

python evaluate/add_probe.py \
    --base "$BASE" \
    --probe-country "$COUNTRY" \
    --condition "$COND" \
    --model-size 8b
