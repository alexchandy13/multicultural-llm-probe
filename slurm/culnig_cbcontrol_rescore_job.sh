#!/bin/bash
#SBATCH --job-name=cbctrl_rescore
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/cbctrl_rescore.%A_%a.out
#SBATCH --error=slurm/cbctrl_rescore.%A_%a.err

# Rescore culturalbenchcontrol after changing how the control strips the country.
#
# The old control blanked the name only, leaving "In , is it customary..." —
# ungrammatical. Attribution is gradient-based and gradients grow where the model
# is uncertain, so a malformed prompt inflates the control. Both sftdpo conditions
# ended up with the control out-scoring the real prompt at every layer (negative
# deltas at 32/32 layers), which neither NormAd's control (blank fields, structure
# intact) nor BLEnD's (question deleted, instruction intact) produce. The control
# now drops the whole clause: "Is it customary...".
#
# Only the control changes; culturalbench itself is untouched. But the selection is
# ranked on (dataset - control), so every culturalbench selection must be
# regenerated afterwards.
#
#   MODEL=8b sbatch --array=0-4 slurm/culnig_cbcontrol_rescore_job.sh
#
# gemma4 needs 2 GPUs and is better run through the sharded job:
#   sbatch --export=ALL,COND=<cond>,DS=culturalbenchcontrol --array=0-7 \
#     --partition=clip --account=clip --qos=high --nodelist=clip13 \
#     --gres=gpu:rtxa6000:2 --mem=10G --cpus-per-task=2 --time=2:00:00 \
#     slurm/culnig_sharded_gemma4_job.sh

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODEL="${MODEL:-8b}"
read -ra CONDS <<< "${CONDS:-base sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult}"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

echo "[cbctrl_rescore] condition=$COND model=$MODEL"
python culnig/calc_neuron_score.py \
    --condition "$COND" --model-size "$MODEL" \
    --precision matched_bf16 --dataset-names culturalbenchcontrol
