#!/bin/bash
#SBATCH --job-name=culture_culnig_8b
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --nodelist=tron46,tron47,tron48
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/culnig_8b.%A_%a.out
#SBATCH --error=slurm/culnig_8b.%A_%a.err

# See sft_8b_job.sh for why --nodelist=tron47 (only A5000 with python3.12).

# Array job — CULNIG gradient scoring at 8B for all 4 conditions. Indices map
# to CONDITIONS in env.sh. Output dirs are outputs/neurons/{cond}_8b/.
#
# Memory bump (32G → 64G) covers Llama 3.1 8B in bf16 (~16 GB on GPU) plus
# host-side tensor copies during the per-batch backward pass.
# Time bump (10h → 20h) covers ~2-3× slower per-batch time at 8B vs 3B.

set -euo pipefail
source env.sh

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[culnig_8b] condition=$COND"

# 5b — NormAd novel extension
python3.12 culnig/calc_neuron_score_normad.py --condition "$COND" --model-size 8b --dataset-names normad
python3.12 culnig/calc_neuron_score_normad.py --condition "$COND" --model-size 8b --dataset-names normadcontrol
python3.12 culnig/decide_culture_neurons_normad.py --condition "$COND" --model-size 8b --dataset-names normad
