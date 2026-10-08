#!/bin/bash
#SBATCH --job-name=cb_nfs4_us
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=6:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/cb_nfs4_us.%A_%a.out
#SBATCH --error=slurm/cb_nfs4_us.%A_%a.err

# CulturalBench with the gold-matched 4-shot neutral prefix plus the US probe,
# which is the base file the cluster-representative probe grid reads for _nfs4.
#
# nfs4 rather than nfs because CulturalBench gold is 27% yes and the hardcoded
# 2-shot prefix is 1 yes / 1 no, signalling 50%. That mis-signal leaves skew at
# +7.8 to +26.2 (8b) and +15.7 to +35.9 (gemma4), and it shows up in the probe
# deltas as conditions scoring *higher* with a wrong country substituted in.
# The 1 yes / 3 no prefix brings skew to -6.1..+15.9 and raises accuracy on every
# condition.
#
#   MODEL=8b     sbatch --array=0-6 slurm/eval_culturalbench_nfs4_usprobe_job.sh
#   MODEL=gemma4 sbatch --array=0-4 --gres=gpu:rtxa6000:1 slurm/eval_culturalbench_nfs4_usprobe_job.sh
#
# HF_DATASETS_OFFLINE=1 is worth setting at submit time: some compute nodes have
# an NFS mount without working flock, and the datasets builder lock then dies with
# "OSError: [Errno 37] No locks available".

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

MODEL="${MODEL:-8b}"
SHOTS="${SHOTS:-4}"
read -ra CONDS <<< "${CONDS:-base sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult tulu3_sft tulu3_dpo}"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

if [[ "$MODEL" != "8b" && "$COND" == tulu3_* ]]; then
    echo "[cb_nfs4_us] $COND exists only at 8b — nothing to do for $MODEL"
    exit 0
fi

OUT="outputs/behavioral/culturalbench_${COND}_${MODEL}_nfs${SHOTS}_usprobe.json"
if [[ -f "$OUT" ]]; then
    echo "[cb_nfs4_us] $OUT already exists — nothing to do"
    exit 0
fi

echo "[cb_nfs4_us] condition=$COND model=$MODEL shots=$SHOTS"
python evaluate/eval_culturalbench.py \
    --condition "$COND" --model-size "$MODEL" \
    --neutral-shots "$SHOTS" --us-probe
