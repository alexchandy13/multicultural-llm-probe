#!/bin/bash
#SBATCH --job-name=culture_sft_8b
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --exclude=tron[49-61]
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/sft_8b.%j.out
#SBATCH --error=slurm/sft_8b.%j.err

# SFT on Alpaca for Llama 3.1 8B (C2 at 8B). Mirrors sft_job.sh but on the 8B
# config. Memory bumped to 48G and wall-time bumped to 24h to account for the
# ~3-4× slower per-step time at 8B vs 3B.
#
# NB on --nodelist=tron47: the Tron A5000 fleet (tron46-tron61) is heterogeneous
# on OS image. tron51-58+ ship python3.9 only, but our pip-installed packages
# live under python3.12 (~/.local/lib/python3.12/...). tron47 is the only A5000
# we've confirmed has python3.12 at /usr/bin/python3.12. Run
# `slurm/check_python_on_nodes.sh rtxa5000 46 61` to re-probe; update this
# nodelist if more good nodes appear. tron47 contention is the main downside.

set -euo pipefail
source env.sh

python3.12 finetune/sft_train.py --config finetune/configs/sft_8b_config.yaml
