#!/bin/bash
#SBATCH --job-name=culture_culnig
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=10:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --output=slurm/culnig.%A_%a.out
#SBATCH --error=slurm/culnig.%A_%a.err

# Array job: one task per condition. Runs Step 5b (NormAd, the novel extension)
# and Step 5a (BLEnD baseline) — both via our culnig/ forked scripts so they
# share QLoRA loading and Llama-3.2-3B model whitelisting.
#
# Output layout:
#   outputs/neurons/{condition}/
#     normad_max_scores.json
#     normadcontrol_max_scores.json
#     countryrc_max_scores.json
#     all_neurons_normad_max.json
#     blend_max_scores.json
#     blendcontrol_max_scores.json
#     all_neurons_blend_max.json

set -euo pipefail
source env.sh

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[culnig] condition=$COND"

# 5b — NormAd novel extension (primary)
python3.12 culnig/calc_neuron_score_normad.py --condition "$COND" --dataset-names normad
python3.12 culnig/calc_neuron_score_normad.py --condition "$COND" --dataset-names normadcontrol
python3.12 culnig/decide_culture_neurons_normad.py --condition "$COND" --dataset-names normad

# 5a — BLEnD baseline disabled. Upstream's BLEnD loader requires local metadata
# (US_questions.csv etc.) that we never installed. To re-enable: clone BLEnD into
# culnig/_upstream/data/BLEnD/, then uncomment the three lines below.
# python3.12 culnig/calc_neuron_score_normad.py --condition "$COND" --dataset-names blend
# python3.12 culnig/calc_neuron_score_normad.py --condition "$COND" --dataset-names blendcontrol
# python3.12 culnig/decide_culture_neurons_normad.py --condition "$COND" --dataset-names blend
