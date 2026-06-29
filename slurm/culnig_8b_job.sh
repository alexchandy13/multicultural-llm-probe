#!/bin/bash
#SBATCH --job-name=culture_culnig_8b
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/culnig_8b.%A_%a.out
#SBATCH --error=slurm/culnig_8b.%A_%a.err

# CULNIG gradient scoring at 8B on CLIP A6000 (48GB — plenty of headroom for
# bf16 base + retained activations + grads). Array indices map to CONDITIONS
# in env.sh. Output dirs are outputs/neurons/{cond}_8b/.

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[culnig_8b] condition=$COND"

# NormAd novel extension — score, control, then select
python culnig/calc_neuron_score_normad.py --condition "$COND" --model-size 8b --precision matched_bf16 --dataset-names normad
python culnig/calc_neuron_score_normad.py --condition "$COND" --model-size 8b --precision matched_bf16 --dataset-names normadcontrol
python culnig/decide_culture_neurons_normad.py --condition "$COND" --model-size 8b --dataset-names normad
