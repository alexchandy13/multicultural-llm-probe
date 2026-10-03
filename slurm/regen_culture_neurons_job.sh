#!/bin/bash
#SBATCH --job-name=regen_neurons
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=8:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --output=slurm/regen_neurons.%j.out
#SBATCH --error=slurm/regen_neurons.%j.err

# Re-run decide_culture_neurons.py for every condition x dataset, against the
# extended countryrc (81 countries). CPU only — no --gres, so this schedules easily.
#
# Memory is the binding constraint, not time. decide_culture_neurons.py json.loads
# three files and holds them live; parsing costs ~2x file size, so gemma4 normad
# peaks near 34 GB (5.4 + 5.4 + 6.3 GB of JSON). 96 GB leaves headroom.
#
# Narrow to one model size by passing a glob:
#   sbatch slurm/regen_culture_neurons_job.sh '*_8b'

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

echo "[regen_neurons] pattern=${1:-*}"
bash scripts/regen_culture_neurons.sh "${1:-*}"
