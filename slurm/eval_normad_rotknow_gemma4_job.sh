#!/bin/bash
#SBATCH --job-name=normad_rotknow
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=slurm/eval_normad_rotknow_gemma4.%A_%a.out
#SBATCH --error=slurm/eval_normad_rotknow_gemma4.%A_%a.err

# Ask about the Rule-of-Thumb with no scenario, to compare against the
# standard scenario eval.
#
#   APPLY   country + scenario   -> yes/no    already run (_fs2_yn)
#   KNOW    country + norm       -> yes/no    this job
#
# Same template with "Scenario:" swapped for "Norm:", same 1,697 examples in
# the same order, so the two cross per example.
#
# Plus the country swap: the same norm asked again with the country replaced,
# stored in us_pred exactly as the scenario arm stores it. Same 8 probe
# countries, one per IW cluster, so the grids line up cell for cell.
#
# Plus the coherent country-free drop (--no-country): the same question with
# the "Country:" line and "in this country" both removed, so nothing dangles.
# That answer is the model's own normative default, and comparing it to each
# of the 8 countries is what says whether the default is US-like.
#
# All of it runs in ONE task per condition. Neither the own-country answer nor
# the country-free answer depends on the probe country, so scoring each once
# covers all 8 probes: 1 + 1 + 8 = 10 passes per example rather than 8 x 2 for
# the probes plus 2 more for a separate country-free job, which would score
# the own-country prompt twice. 2 tasks instead of 16 + 2.
#
# Output is still one file per probe country, each carrying pred (own
# country), nc_pred (no country) and us_pred (swapped), so nothing downstream
# needs to know this was one job.
#
# 10 passes x 1,697 = 16,970 per task, ~56 min on Gemma-4-12B, which needs two cards in bf16. Walltime is 4h because a
# scavenger preemption now costs the whole condition rather than one probe.
#
# 2 conditions = 2 tasks (no Gemma Tulu-3 release).
#
# Submit with: sbatch slurm/eval_normad_rotknow_gemma4_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

CONDS=(sftdpo_aya_cult sftdpo_aya_nocult)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

echo "[normad_rotknow] condition=$COND model=gemma4"

python evaluate/eval_normad_rot_knowledge.py \
    --condition "$COND" \
    --model-size gemma4 \
    --precision matched_bf16 \
    --no-country \
    --probe-countries us netherlands hungary bosnia_and_herzegovina \
                      mexico taiwan philippines iran
