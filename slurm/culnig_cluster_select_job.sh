#!/bin/bash
#SBATCH --job-name=clust_sel
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=2:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=slurm/clust_sel.%A_%a.out
#SBATCH --error=slurm/clust_sel.%A_%a.err
#
# Cluster-specific culture neuron selection. No GPU — this only reads score files
# that calc_neuron_score.py already wrote.
#
# Not run on the login node because it is memory-hungry rather than slow: a
# gemma4 pass streams a ~5.9 GB countryrc file and holds 2.95M neurons x 8
# clusters as float64, so a few GB resident. The submission-node policy forbids
# exactly that.
#
#   MODEL=8b     DS=culturalbench sbatch --array=0-4 slurm/culnig_cluster_select_job.sh
#   MODEL=gemma4 DS=normad YN=1 --mem=48G sbatch --array=0-4 slurm/culnig_cluster_select_job.sh
#
# DS=normad needs YN=1 to read normad_yn_max_scores.json rather than the 3-way file.

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODEL="${MODEL:-8b}"
DS="${DS:-culturalbench}"
read -ra CONDS <<< "${CONDS:-base sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult}"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

EXTRA=()
[[ "${YN:-0}" == "1" ]] && EXTRA+=(--yn-only)
[[ -n "${MIN_SPREAD:-}" ]] && EXTRA+=(--min-spread "$MIN_SPREAD")

echo "[clust_sel] condition=$COND model=$MODEL dataset=$DS"
python culnig/decide_cluster_neurons.py \
    --condition "$COND" --model-size "$MODEL" \
    --dataset-names "$DS" ${EXTRA[@]+"${EXTRA[@]}"}
