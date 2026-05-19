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

# --- Prerequisites: I-W coordinates table (needed by every culture-split script).
# Idempotent — skips the heavy WVS pass if data/iw_coordinates.csv already exists.
if [[ ! -f data/iw_coordinates.csv ]]; then
    python3.12 analysis/culturemapping/compute_iw_coords.py
fi
python3.12 analysis/culturemapping/validate_iw_clusters.py   # sanity-check + summary CSV

# --- Cross-condition comparison tables
python3.12 analysis/compare_conditions.py
python3.12 analysis/neuron_attribution.py

# --- Headline figures (each writes to outputs/figures/)
python3.12 analysis/cluster_accuracy_bars.py
python3.12 analysis/accuracy_deltas_bars.py
python3.12 analysis/accuracy_pipeline_lines.py
python3.12 analysis/cluster_accuracy_scatter.py

# --- CULNIG heatmaps (all four primary variants)
python3.12 analysis/heatmaps.py --figures layer_count
python3.12 analysis/heatmaps.py --figures group_attribution_per_condition --subtract-control
python3.12 analysis/heatmaps.py --figures group_attribution_per_condition_by_module --subtract-control
python3.12 analysis/heatmaps.py --figures asymmetry_attribution --subtract-control

echo "[done] analysis complete — outputs in outputs/{behavioral,neurons,figures}/"
