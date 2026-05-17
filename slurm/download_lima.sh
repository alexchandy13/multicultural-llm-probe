#!/bin/bash
#SBATCH --job-name=culture_download_lima
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm/download_lima.%j.out
#SBATCH --error=slurm/download_lima.%j.err

# Standalone LIMA downloader. Useful when slurm/download_data.sh already ran
# successfully for the other datasets and only LIMA needs to be added.
#
# Background: GAIR/lima ships a legacy `lima.py` loading script which newer
# `datasets` versions (>=3.0) refuse to execute. We bypass that by pulling
# train.jsonl directly via huggingface_hub and saving it as a HF Dataset in
# the layout finetune/sft_train.py:_load_lima expects.

set -euo pipefail
source env.sh

python3.12 - <<'PY'
import json
from pathlib import Path
from datasets import Dataset
from huggingface_hub import hf_hub_download

target = Path("data/lima")
target.mkdir(parents=True, exist_ok=True)

raw = hf_hub_download(
    repo_id="GAIR/lima", filename="train.jsonl", repo_type="dataset",
)
print(f"Downloaded raw file to: {raw}")

with open(raw) as f:
    rows = [json.loads(line) for line in f]
print(f"Parsed {len(rows)} LIMA examples. First row keys: {list(rows[0].keys())}")

Dataset.from_list(rows).save_to_disk(str(target))
print(f"Saved as HF Dataset at: {target}")
PY
