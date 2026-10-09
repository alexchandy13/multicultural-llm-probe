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
# Defaults match the culture-general selection: 5% of MLP, 1% of attention, 1%
# countryrc from each pool. Override only to test sensitivity.
[[ -n "${MLP_PROP:-}" ]] && EXTRA+=(--mlp-proportion "$MLP_PROP")
[[ -n "${ATTN_PROP:-}" ]] && EXTRA+=(--attn-proportion "$ATTN_PROP")
[[ -n "${CRC_PROP:-}" ]] && EXTRA+=(--countryrc-proportion "$CRC_PROP")
# BY_COUNTRY=1 groups by individual country rather than IW cluster, so the
# grouping is measured rather than assumed. NormAd has 68 countries against 8
# clusters, so the per-neuron group vector is 8.5x wider: budget ~12G at 8b and
# ~32G at gemma4, well above the 24G default here.
[[ "${BY_COUNTRY:-0}" == "1" ]] && EXTRA+=(--by-country)

echo "[clust_sel] condition=$COND model=$MODEL dataset=$DS"
python culnig/decide_cluster_neurons.py \
    --condition "$COND" --model-size "$MODEL" \
    --dataset-names "$DS" ${EXTRA[@]+"${EXTRA[@]}"}
