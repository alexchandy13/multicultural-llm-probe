#!/bin/bash
#SBATCH --job-name=culture_culnig_qwen35
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/culnig_qwen35.%A_%a.out
#SBATCH --error=slurm/culnig_qwen35.%A_%a.err

# CULNIG gradient scoring at qwen35. Output dirs are outputs/neurons/{cond}_qwen35/.

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[culnig_qwen35] condition=$COND"

python culnig/calc_neuron_score_normad.py --condition "$COND" --model-size qwen35 --precision matched_bf16 --dataset-names normad
python culnig/calc_neuron_score_normad.py --condition "$COND" --model-size qwen35 --precision matched_bf16 --dataset-names normadcontrol
python culnig/decide_culture_neurons_normad.py --condition "$COND" --model-size qwen35 --dataset-names normad
