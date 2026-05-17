#!/bin/bash
#SBATCH --job-name=culture_download
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/download_data.%j.out
#SBATCH --error=slurm/download_data.%j.err

# Pre-pull all models and datasets to the login-node cache before submitting GPU jobs —
# compute nodes don't reliably have outbound internet on Nexus.

set -euo pipefail

source env.sh

python3.12 - <<'PY'
import os, sys
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA = Path("data")

specs = [
    # HH-RLHF: source for the primary SFT (chosen only) and DPO (full pairs).
    ("Anthropic/hh-rlhf",           DATA / "hh-rlhf"),
    # Alpaca: source for the C2a robustness variant. SFT was trained on this
    # offline from Nexus (checkpoint at checkpoints/sft_alpaca/), but we keep
    # the data downloadable so the config is reproducible from this repo.
    ("tatsu-lab/alpaca",            DATA / "alpaca"),
    # LIMA: source for the C2b robustness variant. 1000 examples; trained on Nexus.
    ("GAIR/lima",                   DATA / "lima"),
    ("akhilayerukola/NormAd",       DATA / "NormAd"),
    ("Taise228/CountryRC",          DATA / "CountryRC"),
]

for repo, target in specs:
    target.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo} -> {target}")
    ds = load_dataset(repo)
    ds.save_to_disk(str(target))

# BLEnD per its own download instructions — handled by CULNIG upstream; placeholder.
print("[note] BLEnD: follow ynklab/CULNIG download instructions; expected at data/BLEnD")

# Models
for name in ("meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"):
    print(f"[download] {name}")
    AutoTokenizer.from_pretrained(name)
    AutoModelForCausalLM.from_pretrained(name, low_cpu_mem_usage=True)

print("[done] downloads complete")
PY
