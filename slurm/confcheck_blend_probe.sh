#!/bin/bash
#SBATCH --job-name=blend_confchk
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=00:45:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --output=slurm/blend_confcheck.%A_%a.out
#SBATCH --error=slurm/blend_confcheck.%A_%a.err
#SBATCH --array=0-31

# One-off confidence check: does the model's confidence on the swapped prompt
# (us_scores, now persisted by add_probe.py's run_blend_probe) differ between
# the US probe and the other 7 cluster-representative probes? 4 instruct
# models x 8 probe countries = 32 tasks, one GPU each, all submitted at once
# so the scheduler runs as many in parallel as the partition allows -- no task
# waits on another.
#
# --sample-frac 0.15 (~3,450 of BLEnD's 23,032 rows/country) keeps each task to
# roughly a model load plus ~5-6 min of scoring. Output goes to a distinct
# _confcheck.json tail so this never collides with or overwrites the real,
# full probe grid at *_<country>probe_raw.json.
#
#   sbatch slurm/confcheck_blend_probe.sh
#   sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed --array

set -euo pipefail

MODELS=(8b_instruct gemma4_instruct olmo3_instruct qwen35_instruct)
CONDS=(base         base             offshelf      offshelf)
COUNTRIES=(us netherlands hungary bosnia_and_herzegovina mexico taiwan philippines iran)

N_COUNTRIES=${#COUNTRIES[@]}
MI=$(( SLURM_ARRAY_TASK_ID / N_COUNTRIES ))
CI=$(( SLURM_ARRAY_TASK_ID % N_COUNTRIES ))
MODEL=${MODELS[$MI]}
COND=${CONDS[$MI]}
COUNTRY=${COUNTRIES[$CI]}

BASE="outputs/behavioral/blend_${COND}_${MODEL}_nfs_raw.json"
OUT="outputs/behavioral/blend_${COND}_${MODEL}_nfs_${COUNTRY}probe_confcheck.json"

echo "[confcheck] task=$SLURM_ARRAY_TASK_ID model=$MODEL cond=$COND country=$COUNTRY"
echo "[confcheck] base=$BASE out=$OUT"

if [[ ! -f "$BASE" ]]; then
    echo "[confcheck] missing base $BASE" >&2
    exit 1
fi

source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python evaluate/add_probe.py \
    --base "$BASE" \
    --probe-country "$COUNTRY" \
    --condition "$COND" --model-size "$MODEL" --precision matched_bf16 \
    --raw-prompt --sample-frac 0.15 \
    --out "$OUT"
