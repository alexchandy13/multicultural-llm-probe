#!/bin/bash
#SBATCH --job-name=normad_rotknow_nc
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=1:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --output=slurm/eval_normad_rotknow_nc.%A_%a.out
#SBATCH --error=slurm/eval_normad_rotknow_nc.%A_%a.err

# The norm question with no country at all, beside the same question with the
# example's own country:
#
#   own country   Country: nepal / Norm: ... / Is this a social norm in this country?
#   no country                     Norm: ... / Is this a social norm?
#
# The country-free prompt is hand-written, not the own-country one with the
# "Country:" line deleted. Deleting the line alone leaves "in this country"
# pointing at nothing, which is the incoherent control culnig/dataset_ext.py
# records on CulturalBench -- a dangling reference inflated it until it
# outscored the real prompt. So the clause goes too.
#
# This is the baseline the country swap does not have. us_pred says what the
# model answers when told a country; nc_pred says what it answers when told
# none. Without it, a high swap agreement cannot be told apart from the model
# ignoring the country line entirely.
#
# 2 passes per example, 3,394 per task, ~6 min on 8b. No probe countries here:
# the swap grid is the other job, and the own-country answers in both are the
# same prompt.
#
# 3 conditions = 3 tasks. Tulu-3 is 8b-only, so the Gemma job runs the other two.
#
# Submit with: sbatch slurm/eval_normad_rotknow_nocountry_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

CONDS=(sftdpo_aya_cult sftdpo_aya_nocult tulu3_dpo)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

echo "[normad_rotknow_nc] condition=$COND model=8b"

python evaluate/eval_normad_rot_knowledge.py \
    --condition "$COND" \
    --model-size 8b \
    --precision matched_bf16 \
    --no-country
