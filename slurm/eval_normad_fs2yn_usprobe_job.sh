#!/bin/bash
#SBATCH --job-name=normad_fs2yn
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=6:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/normad_fs2yn.%A_%a.out
#SBATCH --error=slurm/normad_fs2yn.%A_%a.err

# NormAd fs2 + yes/no + US probe, which is the base file the cluster-representative
# probe grid reads for the _fs2_yn strategy.
#
# fs2 rather than nfs because it is the prefix that actually fixes NormAd's
# calibration: skew runs +4.0 to +8.8 at fs2 versus -34.6 to +48.0 at 0-shot and
# -16.6 to +25.1 at nfs+mpw. Probe deltas measured on nfs are contaminated by
# threshold shifts — several conditions score *higher* with the wrong country
# substituted in, which is impossible if the delta tracked cultural knowledge.
#
#   MODEL=8b     sbatch --array=0-6 slurm/eval_normad_fs2yn_usprobe_job.sh
#   MODEL=gemma4 sbatch --array=0-4 --gres=gpu:rtxa6000:1 slurm/eval_normad_fs2yn_usprobe_job.sh
#
# base already has this file at both sizes, so it is last in the list and a
# --array that stops short of it simply skips it.

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODEL="${MODEL:-8b}"
SHOTS="${SHOTS:-2}"
read -ra CONDS <<< "${CONDS:-sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult tulu3_sft tulu3_dpo base}"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

if [[ "$MODEL" != "8b" && "$COND" == tulu3_* ]]; then
    echo "[normad_fs2yn] $COND exists only at 8b — nothing to do for $MODEL"
    exit 0
fi

OUT="outputs/behavioral/normad_${COND}_${MODEL}_fs${SHOTS}_yn_usprobe.json"
if [[ -f "$OUT" ]]; then
    echo "[normad_fs2yn] $OUT already exists — nothing to do"
    exit 0
fi

echo "[normad_fs2yn] condition=$COND model=$MODEL shots=$SHOTS"
python evaluate/eval_normad.py \
    --condition "$COND" --model-size "$MODEL" \
    --few-shot "$SHOTS" --yn-only --us-probe
