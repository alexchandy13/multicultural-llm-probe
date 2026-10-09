#!/bin/bash
#SBATCH --job-name=olmo3_suite
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/olmo3_suite.%A_%a.out
#SBATCH --error=slurm/olmo3_suite.%A_%a.err

# Full eval suite for the Olmo 3 7B ladder, at each benchmark's calibrated prefix.
# One array task per checkpoint: base -> SFT -> DPO.
#
#   normad         fs2 + yn    skew +4 to +9, vs -35..+48 at 0-shot
#   culturalbench  nfs4        1 yes / 3 no, matching the 27% gold rate
#   blend          nfs         0-shot neutral shots, no multi-prompt
#   nlu            nfs         capability control
#
# --us-probe on the three cultural benchmarks writes the base file the probe grid
# needs for Taiwan and Iran. Scores are persisted by all current eval code, so
# AUROC is available afterwards.
#
#   sbatch --array=0-2 slurm/eval_olmo3_suite_job.sh
#
# HF_DATASETS_OFFLINE=1 is worth setting at submit time once the datasets are
# cached: some nodes have an NFS mount without working flock and the datasets
# builder lock dies with "OSError: [Errno 37] No locks available".

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra SIZES <<< "${SIZES:-olmo3 olmo3_sft olmo3_dpo}"
SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}
echo "[olmo3_suite] model_size=$SIZE"

echo "--- normad (fs2 + yn) ---"
python evaluate/eval_normad.py --condition base --model-size "$SIZE" \
    --few-shot 2 --yn-only --us-probe

echo "--- culturalbench (nfs4) ---"
python evaluate/eval_culturalbench.py --condition base --model-size "$SIZE" \
    --neutral-shots 4 --us-probe

echo "--- blend (nfs) ---"
python evaluate/eval_blend.py --condition base --model-size "$SIZE" \
    --neutral-fewshot --us-probe

echo "--- general NLU (nfs) ---"
for DS in boolq csqa qnli mrpc; do
    python evaluate/eval_nlu.py --condition base --model-size "$SIZE" \
        --dataset "$DS" --neutral-fewshot
done
