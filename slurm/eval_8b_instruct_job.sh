#!/bin/bash
#SBATCH --job-name=culture_eval_8b_inst
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/eval_8b_instruct.%j.out
#SBATCH --error=slurm/eval_8b_instruct.%j.err

# Eval Llama 3.1 8B Instruct (Meta's official instruction-tuned model).
# Comparison point: if instruct scores higher than our SFT/DPO conditions,
# the issue is the specific alignment datasets (Alpaca/HH-RLHF), not the
# alignment process itself. Only "base" condition makes sense here —
# the instruct model is already aligned; we don't apply custom adapters.

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

echo "[eval_8b_instruct] running base condition (instruct model, no adapter)"

python evaluate/eval_normad.py --condition base --model-size 8b_instruct
python evaluate/eval_normad.py --condition base --model-size 8b_instruct --calibrate
python evaluate/eval_normad.py --condition base --model-size 8b_instruct --few-shot 3
python evaluate/eval_normad.py --condition base --model-size 8b_instruct --mc-format
