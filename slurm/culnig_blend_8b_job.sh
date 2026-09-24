#!/bin/bash
#SBATCH --job-name=culture_culnig_blend_8b
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --output=slurm/culnig_blend_8b.%A_%a.out
#SBATCH --error=slurm/culnig_blend_8b.%A_%a.err

# CULNIG gradient scoring for 8B on BLEnD. No train/test split needed since
# BLEnD has only a test split and we run no downstream intervention eval.
# PREREQUISITE: culnig/_upstream/data/BLEnD/US_questions.csv must exist on the cluster
# (the upstream BLEnD loader needs it for control prompt construction).
# Outputs land in outputs/neurons/{cond}_8b/:
#   blend_max_scores.json / blendcontrol_max_scores.json / all_neurons_blend_max.json

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[culnig_blend_8b] condition=$COND"

python culnig/calc_neuron_score.py --condition "$COND" --model-size 8b --precision matched_bf16 --dataset-names blend --target-data all
python culnig/calc_neuron_score.py --condition "$COND" --model-size 8b --precision matched_bf16 --dataset-names blendcontrol --target-data all
python culnig/decide_culture_neurons.py --condition "$COND" --model-size 8b --dataset-names blend
