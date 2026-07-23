#!/bin/bash
#SBATCH --job-name=culture_eval_8b_blend
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/eval_8b_blend.%A_%a.out
#SBATCH --error=slurm/eval_8b_blend.%A_%a.err

# BLEnD MCQ eval for Llama 3.1 8B.
# Scores A/B/C/D log-probs after '{"answer_choice":"'; averages 4 instruction
# prefix variants for multi-prompt mode. US-probe replaces country with 'US'.
# Outputs: blend_{cond}_8b[_fs2][_mp][_usprobe].json
# Submit with: CONDITIONS="base sft dpo sftdpo" sbatch slurm/eval_8b_blend_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval_8b_blend] condition=$COND"

# 0-shot
python evaluate/eval_blend.py --condition "$COND" --model-size 8b --us-probe
# 0-shot multi-prompt
python evaluate/eval_blend.py --condition "$COND" --model-size 8b --multi-prompt --us-probe
# 2-shot
python evaluate/eval_blend.py --condition "$COND" --model-size 8b --few-shot 2 --us-probe
# 2-shot multi-prompt
python evaluate/eval_blend.py --condition "$COND" --model-size 8b --few-shot 2 --multi-prompt --us-probe
