#!/bin/bash
#SBATCH --job-name=cb_nocountry
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=2:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --output=slurm/eval_culturalbench_nocountry.%A_%a.out
#SBATCH --error=slurm/eval_culturalbench_nocountry.%A_%a.err

# CulturalBench with the country removed from every question, matching the
# NormAd and BLEnD country ablations.
#
# The reformatted questions open with a prepositional phrase naming the
# country -- "In the Netherlands, is it customary to remove your shoes
# indoors?" -- so the coherent drop deletes the whole phrase and recapitalises:
# "Is it customary to remove your shoes indoors?". Blanking the name instead
# would leave "In , is it customary ...", which is the defect
# culnig/dataset_ext.py records on the CulturalBench control: the malformed
# prompt scored HIGHER than the real one at every layer, because gradients
# grow with uncertainty.
#
# Each prediction gains nc_pred beside pred and us_pred, so the country-free
# answer can be compared against the own-country answer and, by joining on row
# order, against each of the 8 probe-country files that already exist.
#
# --neutral-shots 4 matches the _nfs4 files the probe grid was run with. The
# shots are country-free, so they do not reintroduce a country.
#
# The run also reports how many rows the strip applied to. Any row it misses
# is one where make_probe_prompt() silently left the country in place too, so
# the count is a check on the existing probe runs as well as this one.
#
# 1 pass per question on top of the baseline pass, ~8 min on 8b.
#
# 3 conditions = 3 tasks. Tulu-3 is 8b-only, so the Gemma job runs the other two.
#
# Submit with: sbatch slurm/eval_culturalbench_nocountry_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

CONDS=(sftdpo_aya_cult sftdpo_aya_nocult tulu3_dpo)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

echo "[cb_nocountry] condition=$COND model=8b"

python evaluate/eval_culturalbench.py \
    --condition "$COND" \
    --model-size 8b \
    --precision matched_bf16 \
    --neutral-shots 4 \
    --no-country
