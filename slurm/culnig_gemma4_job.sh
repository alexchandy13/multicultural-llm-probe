#!/bin/bash
#SBATCH --job-name=culture_culnig_gemma4
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --output=slurm/culnig_gemma4.%A_%a.out
#SBATCH --error=slurm/culnig_gemma4.%A_%a.err

# CULNIG gradient scoring at gemma4 on CLIP A6000 (48GB). Array indices map
# to CONDITIONS in env.sh, or to CONDS if you set it. Output dirs are
# outputs/neurons/{cond}_gemma4/.
#
# STAGE picks which halves to run, so a condition whose normad is already correct
# does not spend ~10h rescoring it:
#   STAGE=all            (default) normad + culturalbench
#   STAGE=normad         normad only
#   STAGE=culturalbench  culturalbench only
#
#   CONDS="sftdpo_aya_cult sftdpo_aya_nocult" sbatch --array=0-1 slurm/culnig_gemma4_job.sh
#   CONDS="base sft_aya_cult" STAGE=culturalbench sbatch --array=0-1 slurm/culnig_gemma4_job.sh
#
# The decide_culture_neurons steps need countryrc_max_scores.json(.gz) in the
# condition dir. It is not kept on the cluster (47 GB across conditions), so when
# it is absent this job scores only and skips the selection, which you then run
# locally — it streams the files now and peaks around 750 MB:
#
#   python culnig/decide_culture_neurons.py --condition <c> --model-size gemma4 \
#       --dataset-names culturalbench

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

read -ra CONDS <<< "${CONDS:-$CONDITIONS}"
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
STAGE="${STAGE:-all}"
case "$STAGE" in all|normad|culturalbench) ;; *) echo "bad STAGE '$STAGE'" >&2; exit 1 ;; esac
echo "[culnig_gemma4] condition=$COND stage=$STAGE"

# Selection needs countryrc; score-only is a valid outcome when it is not here.
# Checked at point of use, not at job start, so a file still being uploaded while
# the hours of scoring run is picked up by the time selection is reached.
CRC_DIR="outputs/neurons/${COND}_gemma4"

decide() {
    if [[ -f "$CRC_DIR/countryrc_max_scores.json" || -f "$CRC_DIR/countryrc_max_scores.json.gz" ]]; then
        python culnig/decide_culture_neurons.py --condition "$COND" --model-size gemma4 "$@"
    else
        echo "[culnig_gemma4] no countryrc in $CRC_DIR — skipping selection for $*;" \
             "run it locally: python culnig/decide_culture_neurons.py --condition $COND --model-size gemma4 $*"
    fi
}

if [[ "$STAGE" == all || "$STAGE" == normad ]]; then
    python culnig/calc_neuron_score.py --condition "$COND" --model-size gemma4 --precision matched_bf16 --dataset-names normad --yn-only
    python culnig/calc_neuron_score.py --condition "$COND" --model-size gemma4 --precision matched_bf16 --dataset-names normadcontrol
    decide --dataset-names normad --yn-only
fi

if [[ "$STAGE" == all || "$STAGE" == culturalbench ]]; then
    python culnig/calc_neuron_score.py --condition "$COND" --model-size gemma4 --precision matched_bf16 --dataset-names culturalbench
    python culnig/calc_neuron_score.py --condition "$COND" --model-size gemma4 --precision matched_bf16 --dataset-names culturalbenchcontrol
    decide --dataset-names culturalbench
fi
