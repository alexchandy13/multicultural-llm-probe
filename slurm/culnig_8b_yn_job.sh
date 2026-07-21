#!/bin/bash
#SBATCH --job-name=culture_culnig_8b_yn
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/culnig_8b_yn.%A_%a.out
#SBATCH --error=slurm/culnig_8b_yn.%A_%a.err

# CULNIG gradient scoring with yn-only NormAd and updated normadcontrol (full
# content removal). Outputs land in outputs/neurons/{cond}_8b/ with _yn suffix:
#   normad_yn_max_scores.json        (from --yn-only scoring pass)
#   normadcontrol_max_scores.json    (overwritten with full-content-removal version)
#   all_neurons_normad_yn_max.json   (from decide step)

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[culnig_8b_yn] condition=$COND"

python culnig/calc_neuron_score_normad.py --condition "$COND" --model-size 8b --precision matched_bf16 --dataset-names normad --yn-only
python culnig/calc_neuron_score_normad.py --condition "$COND" --model-size 8b --precision matched_bf16 --dataset-names normadcontrol
python culnig/decide_culture_neurons_normad.py --condition "$COND" --model-size 8b --dataset-names normad --yn-only
