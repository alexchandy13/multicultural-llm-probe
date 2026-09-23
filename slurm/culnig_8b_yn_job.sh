#!/bin/bash
#SBATCH --job-name=culture_culnig_8b_yn
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --output=slurm/culnig_8b_yn.%A_%a.out
#SBATCH --error=slurm/culnig_8b_yn.%A_%a.err

# CULNIG gradient scoring for 8B across NormAd and CulturalBench.
# Outputs land in outputs/neurons/{cond}_8b/:
#   normad_yn_max_scores.json / normadcontrol_max_scores.json / all_neurons_normad_yn_max.json
#   culturalbench_max_scores.json / culturalbenchcontrol_max_scores.json / all_neurons_culturalbench_max.json

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[culnig_8b_yn] condition=$COND"

# NormAd (yes/no only)
python culnig/calc_neuron_score.py --condition "$COND" --model-size 8b --precision matched_bf16 --dataset-names normad --yn-only
python culnig/calc_neuron_score.py --condition "$COND" --model-size 8b --precision matched_bf16 --dataset-names normadcontrol
python culnig/decide_culture_neurons.py --condition "$COND" --model-size 8b --dataset-names normad --yn-only

# CulturalBench
python culnig/calc_neuron_score.py --condition "$COND" --model-size 8b --precision matched_bf16 --dataset-names culturalbench
python culnig/calc_neuron_score.py --condition "$COND" --model-size 8b --precision matched_bf16 --dataset-names culturalbenchcontrol
python culnig/decide_culture_neurons.py --condition "$COND" --model-size 8b --dataset-names culturalbench
