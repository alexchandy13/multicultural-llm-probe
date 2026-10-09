#!/bin/bash
#SBATCH --job-name=nlu_olmo3_zs
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/nlu_olmo3_zs.%A_%a.out
#SBATCH --error=slurm/nlu_olmo3_zs.%A_%a.err
#
# Zero-shot general NLU for the OLMo 3 ladder. eval_olmo3_suite_job.sh ran these
# four with --neutral-fewshot only, so olmo3 has no 0-shot NLU.
#
# Unlike the cultural benchmarks, this one has a complete comparison set already:
# every aya condition plus tulu3_sft/dpo has bare nlu_{ds}_{cond}{_size}.json at
# both sizes. So these 12 runs finish the grid rather than creating an orphan row.
#
# Zero-shot is the absence of both prefix flags — omitting --neutral-fewshot and
# --few-shot leaves nfs_sfx empty, giving nlu_{ds}_base_{size}.json.
#
#   sbatch --array=0-2 slurm/eval_nlu_olmo3_zeroshot_job.sh

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra SIZES <<< "${SIZES:-olmo3 olmo3_sft olmo3_dpo}"
SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}
read -ra DATASETS <<< "${DATASETS:-boolq csqa qnli mrpc}"

echo "[nlu_olmo3_zs] model_size=$SIZE datasets=${DATASETS[*]}"
for DS in "${DATASETS[@]}"; do
    OUT="outputs/behavioral/nlu_${DS}_base_${SIZE}.json"
    if [[ -f "$OUT" ]]; then
        echo "--- $DS: $OUT exists, skipping ---"
        continue
    fi
    echo "--- $DS (0-shot) ---"
    python evaluate/eval_nlu.py --condition base --model-size "$SIZE" --dataset "$DS"
done
