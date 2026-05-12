#!/bin/bash
#SBATCH --job-name=culture_analysis
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm/analysis.%j.out
#SBATCH --error=slurm/analysis.%j.err

# Builds Table 1 + three figures from outputs/. No GPU needed.

set -euo pipefail
source env.sh

python analysis/compare_conditions.py
python analysis/neuron_attribution.py
python analysis/visualize.py
