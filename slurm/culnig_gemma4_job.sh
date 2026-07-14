#!/bin/bash
#SBATCH --job-name=culture_culnig_gemma4
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/culnig_gemma4.%A_%a.out
#SBATCH --error=slurm/culnig_gemma4.%A_%a.err

# CULNIG gradient scoring at gemma4 on CLIP A6000 (48GB). Array indices map
# to CONDITIONS in env.sh. Output dirs are outputs/neurons/{cond}_gemma4/.

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[culnig_gemma4] condition=$COND"

# NormAd novel extension — score, control, then select
python culnig/calc_neuron_score_normad.py --condition "$COND" --model-size gemma4 --precision matched_bf16 --dataset-names normad
python culnig/calc_neuron_score_normad.py --condition "$COND" --model-size gemma4 --precision matched_bf16 --dataset-names normadcontrol
python culnig/decide_culture_neurons_normad.py --condition "$COND" --model-size gemma4 --dataset-names normad
