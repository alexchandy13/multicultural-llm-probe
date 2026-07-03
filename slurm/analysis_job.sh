#!/bin/bash
#SBATCH --job-name=culture_analysis
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm/analysis.%j.out
#SBATCH --error=slurm/analysis.%j.err

# Build all paper tables + figures from outputs/. No GPU needed.
#
# Run after Step 5 (eval_job.sh) and Step 6 (culnig_job.sh) have completed.
# All scripts are CPU-only; this batch wrapper just bundles them so a fresh
# analysis pass is one command.

set -euo pipefail
source env.sh
PYTHON=$(command -v python3.12 2>/dev/null || command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python)

MODEL_SIZE="3b"
for arg in "$@"; do
    if [[ "$arg" == "--model-size" ]]; then
        shift; MODEL_SIZE="$1"
    elif [[ "$arg" == --model-size=* ]]; then
        MODEL_SIZE="${arg#--model-size=}"
    fi
done

# --- Prerequisites: I-W coordinates table (needed by every culture-split script).
# Idempotent — skips the heavy WVS pass if data/iw_coordinates.csv already exists.
if [[ ! -f data/iw_coordinates.csv ]]; then
    $PYTHON analysis/culturemapping/compute_iw_coords.py
fi
$PYTHON analysis/culturemapping/validate_iw_clusters.py   # sanity-check + summary CSV

# --- Cross-condition comparison tables
$PYTHON analysis/compare_conditions.py
$PYTHON analysis/neuron_attribution.py
$PYTHON analysis/neuron_overlap.py --model-size "$MODEL_SIZE"
$PYTHON analysis/significance_tests.py --model-size "$MODEL_SIZE"

# --- Headline figures (each writes to outputs/figures/)
$PYTHON analysis/cluster_accuracy_bars.py --model-size "$MODEL_SIZE"
$PYTHON analysis/accuracy_deltas_bars.py --model-size "$MODEL_SIZE"
$PYTHON analysis/accuracy_pipeline_lines.py --model-size "$MODEL_SIZE"
$PYTHON analysis/cluster_accuracy_scatter.py --model-size "$MODEL_SIZE"

# --- CULNIG heatmaps (all four primary variants)
$PYTHON analysis/heatmaps.py --figures layer_count --model-size "$MODEL_SIZE"
$PYTHON analysis/heatmaps.py --figures group_attribution_per_condition --subtract-control --model-size "$MODEL_SIZE"
$PYTHON analysis/heatmaps.py --figures group_attribution_per_condition_by_module --subtract-control --model-size "$MODEL_SIZE"
$PYTHON analysis/heatmaps.py --figures asymmetry_attribution --subtract-control --model-size "$MODEL_SIZE"

echo "[done] analysis complete — outputs in outputs/{behavioral,neurons,figures}/"
