#!/bin/bash
#SBATCH --job-name=nlu_nfs4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=3:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-6
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/nlu_nfs4.%A_%a.out
#SBATCH --error=slurm/nlu_nfs4.%A_%a.err

# QNLI and MRPC with a gold-matched 4-shot neutral prefix.
#
# The existing _nfs runs use 1 yes / 1 no, which signals a 50% yes-rate. QNLI's
# gold is 49.5% yes so that was already fine, but MRPC's is 68.4% — an 18-point
# under-signal that drove tulu3_dpo from +3.4 skew at 0-shot to -30.6 with nfs,
# costing 12 accuracy points. At 4 shots the matched splits are 2/2 and 3/1.
#
# BoolQ (-12.2 mismatch) and CSQA (uniform shots, uniform gold, no skew) are
# deliberately excluded and will refuse --neutral-shots 4.
#
#   MODEL=8b     sbatch slurm/eval_nlu_nfs4_job.sh
#   MODEL=gemma4 sbatch --array=0-4 slurm/eval_nlu_nfs4_job.sh   # tulu3 is 8b-only
#
# QNLI is 5463 examples and MRPC 408, so ~5-10 min per condition after load.

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODEL="${MODEL:-8b}"
SHOTS="${SHOTS:-4}"
read -ra CONDS <<< "${CONDS:-base sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult tulu3_sft tulu3_dpo}"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

if [[ "$MODEL" != "8b" && "$COND" == tulu3_* ]]; then
    echo "[nlu_nfs4] $COND exists only at 8b — nothing to do for $MODEL"
    exit 0
fi

for DS in qnli mrpc; do
    echo "[nlu_nfs4] condition=$COND model=$MODEL dataset=$DS shots=$SHOTS"
    python evaluate/eval_nlu.py \
        --condition "$COND" --model-size "$MODEL" --dataset "$DS" \
        --neutral-shots "$SHOTS"
done
