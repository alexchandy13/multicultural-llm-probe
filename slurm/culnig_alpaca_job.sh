#!/bin/bash
#SBATCH --job-name=culture_culnig_alpaca
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=slurm/culnig_alpaca.%A_%a.out
#SBATCH --error=slurm/culnig_alpaca.%A_%a.err

# CULNIG (Step 5) for the Alpaca-variant conditions only (C2a, C4a).
# Same structure as slurm/culnig_job.sh — runs both 5b (NormAd) and 5a (BLEnD)
# per condition. Two array tasks: 0 -> sft_alpaca, 1 -> sftdpo_alpaca.
#
# Condition list is hard-coded here for the same reason as eval_alpaca_job.sh:
# the primary CULNIG job continues to operate on the env.sh `CONDITIONS` set.

set -euo pipefail
source env.sh

CONDS=(sft_alpaca sftdpo_alpaca)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[culnig_alpaca] condition=$COND"

# 5b — NormAd novel extension
python3.12 culnig/calc_neuron_score_normad.py --condition "$COND" --dataset-names normad
python3.12 culnig/calc_neuron_score_normad.py --condition "$COND" --dataset-names normadcontrol
python3.12 culnig/decide_culture_neurons_normad.py --condition "$COND" --dataset-names normad

# 5a — BLEnD baseline disabled here for symmetry with slurm/culnig_job.sh.
# See that file's comment for re-enable instructions.
# python3.12 culnig/calc_neuron_score_normad.py --condition "$COND" --dataset-names blend
# python3.12 culnig/calc_neuron_score_normad.py --condition "$COND" --dataset-names blendcontrol
# python3.12 culnig/decide_culture_neurons_normad.py --condition "$COND" --dataset-names blend
