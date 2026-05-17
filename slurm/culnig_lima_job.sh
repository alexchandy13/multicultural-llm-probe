#!/bin/bash
#SBATCH --job-name=culture_culnig_lima
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=10:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=slurm/culnig_lima.%A_%a.out
#SBATCH --error=slurm/culnig_lima.%A_%a.err

# CULNIG (Step 5) for the LIMA-variant conditions only (C2b, C4b).
# Same structure as slurm/culnig_alpaca_job.sh — 5b NormAd path only
# (5a BLEnD baseline remains disabled across all CULNIG scripts).

set -euo pipefail
source env.sh

CONDS=(sft_lima sftdpo_lima)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[culnig_lima] condition=$COND"

# 5b — NormAd novel extension
python3.12 culnig/calc_neuron_score_normad.py --condition "$COND" --dataset-names normad
python3.12 culnig/calc_neuron_score_normad.py --condition "$COND" --dataset-names normadcontrol
python3.12 culnig/decide_culture_neurons_normad.py --condition "$COND" --dataset-names normad

# 5a — BLEnD baseline disabled here for symmetry with the other CULNIG scripts.
# python3.12 culnig/calc_neuron_score_normad.py --condition "$COND" --dataset-names blend
# python3.12 culnig/calc_neuron_score_normad.py --condition "$COND" --dataset-names blendcontrol
# python3.12 culnig/decide_culture_neurons_normad.py --condition "$COND" --dataset-names blend
