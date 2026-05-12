#!/bin/bash
#SBATCH --job-name=culture_culnig
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=10:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm/culnig.%A_%a.out
#SBATCH --error=slurm/culnig.%A_%a.err

# Array job: one task per condition. Runs both 5a (BLEnD baseline, upstream scripts)
# and 5b (NormAd extension, our scripts), then decides culture neurons for both.

set -euo pipefail
source env.sh

read -ra CONDS <<< "$CONDITIONS"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "[culnig] condition=$COND"

# 5a — Original CULNIG / BLEnD path. Run only if upstream clone is present.
if [[ -d culnig/_upstream ]]; then
    python culnig/_upstream/calc_neuron_score.py --condition "$COND"
    python culnig/_upstream/decide_culture_general_neurons.py --condition "$COND"
else
    echo "[skip] culnig/_upstream not found — skipping BLEnD baseline"
fi

# 5b — Extended CULNIG / NormAd path.
python culnig/calc_neuron_score_normad.py     --condition "$COND"
python culnig/decide_culture_neurons_normad.py --condition "$COND"
