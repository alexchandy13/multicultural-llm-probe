# UMIACS Nexus setup guide

Step-by-step setup for running the 3B and 8B pipelines on UMIACS Nexus with a
class account (30 GB home, ~200 GB scratch).

## Storage layout

The total quota is tight relative to what 8B training and CULNIG scoring
produce, so this matters. The guiding rule:

| Lives where | What goes there | Why |
|---|---|---|
| `~` (home, 30 GB, backed up) | Code, configs, slurm scripts, small analysis outputs | Snapshots + backups; small enough to fit |
| `/fs/nexus-scratch/<USERNAME>` or `/scratch0/<USERNAME>` (scratch, ~200 GB, no backups) | Datasets (HH-RLHF ≈ 5 GB; WVS ≈ 200 MB), LoRA adapter checkpoints, CULNIG score JSONs (~1.7 GB × 4 files × 4 conditions × N model sizes), HF model cache | Big, regenerable from huggingface + training scripts |

**Class-account caveat.** The Nexus wiki notes that *"class accounts do not
have network scratch directories"* by default. Your 200 GB allocation may
therefore be local `/scratch0/<USERNAME>` on compute nodes (with 90-day
auto-delete) rather than network `/fs/nexus-scratch/<USERNAME>`. **Verify with
`df -h ~` and `ls /fs/nexus-scratch/$USER /scratch0/$USER` after first login**;
if `/fs/nexus-scratch/$USER` doesn't exist, you're on local scratch and need to
treat the 200 GB as job-local, not persistent. The setup below assumes
network scratch is available; substitute `/scratch0/$USER` everywhere if not,
and add a job-end rsync step to copy outputs back to home before the
allocation auto-deletes.

## Budgeting your storage

| Item | 3B | 8B | Notes |
|---|---:|---:|---|
| HF model cache: Llama-3.2-3B | ~6 GB | — | bf16 weights |
| HF model cache: Llama-3.1-8B | — | ~16 GB | bf16 weights |
| Datasets (HH-RLHF + Alpaca + NormAd + CountryRC) | ~6 GB | (shared) | One copy across sizes |
| WVS Wave 7 raw CSV | ~200 MB | (shared) | One copy |
| LoRA adapters (3 conditions × ~300 MB) | ~1 GB | ~2.5 GB | Trained on Nexus |
| CULNIG score files (4 files × 1.7 GB × 4 conditions) | ~27 GB | ~30 GB | Per size; primary disk hog |
| Behavioral eval JSONs | <100 MB | <100 MB | Tiny |
| **Total per model size** | **~40 GB** | **~48 GB** | |

Running both 3B and 8B end-to-end fits in 200 GB scratch with ~110 GB to
spare. Running just 8B (after deleting 3B artifacts) leaves ~150 GB.

## Initial setup (do once)

```bash
# 1. SSH in
ssh <username>@nexus.umiacs.umd.edu

# 2. Confirm storage
df -h ~                              # home directory quota — should show 30G
ls /fs/nexus-scratch/$USER 2>/dev/null && echo "have network scratch" \
    || echo "no network scratch — use /scratch0 instead"
df -h /fs/nexus-scratch/$USER 2>/dev/null || df -h /scratch0

# 3. Pick a scratch location and create your work dir there
SCRATCH=/fs/nexus-scratch/$USER     # or /scratch0/$USER if no network scratch
mkdir -p "$SCRATCH/multicultural-llm-probe-data"

# 4. Clone the repo into home (small — fine for 30 GB quota)
cd ~
git clone https://github.com/alexchandy13/multicultural-llm-probe.git
cd multicultural-llm-probe
```

## Linking large artifacts to scratch

The repo expects three big directories (`data/`, `checkpoints/`, `outputs/`)
at the project root. We symlink them to scratch so the bytes live there but
the code paths don't change:

```bash
cd ~/multicultural-llm-probe

# Remove the existing (empty) dirs and replace with symlinks to scratch.
rm -rf data checkpoints outputs

ln -s "$SCRATCH/multicultural-llm-probe-data/data"        data
ln -s "$SCRATCH/multicultural-llm-probe-data/checkpoints" checkpoints
ln -s "$SCRATCH/multicultural-llm-probe-data/outputs"     outputs

# Create the targets
mkdir -p "$SCRATCH/multicultural-llm-probe-data"/{data,checkpoints,outputs}
```

After this, anything the code writes to `data/`, `checkpoints/`, or
`outputs/` actually lands in scratch, but you can still `cd
~/multicultural-llm-probe` and `git status` cleanly without scratch pollution.

## Redirecting the Hugging Face cache

The repo's `env.sh` already sets `HF_HOME=$HOME/.cache/huggingface`. On
Nexus that path eats your home quota fast (Llama 3.1 8B alone is 16 GB
download). **Override it to scratch:**

```bash
# Edit env.sh and change the HF_HOME block. The cleanest patch:
sed -i 's|export HF_HOME="$HOME/.cache/huggingface"|export HF_HOME="$SCRATCH/multicultural-llm-probe-data/hf_cache"|' env.sh
```

Or add a one-line override at the top of `env.sh` for Nexus only:

```bash
# Nexus override: HF cache lives on scratch
[ -d /fs/nexus-scratch/$USER ] && export HF_HOME=/fs/nexus-scratch/$USER/hf_cache
```

This is the single most common quota-killer for class accounts — fix it
before the first download attempt.

## Downloading data and models (one job)

```bash
# Authenticate to HF for the gated Llama models
huggingface-cli login   # paste your token; needs read scope

# Submit the download job (runs on a CPU node, no GPU needed)
source env.sh
sbatch slurm/download_data.sh
```

`slurm/download_data.sh` already handles HH-RLHF, Alpaca, NormAd, CountryRC,
and Llama-3.2-3B. To also pre-fetch Llama-3.1-8B, edit the for-loop at the
bottom of `slurm/download_data.sh`:

```python
# Change this line:
for name in ("meta-llama/Llama-3.2-3B",):
# To this:
for name in ("meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B"):
```

Or, simpler, just download the 8B base ad-hoc on the login node:

```bash
python3.12 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
for m in ['meta-llama/Llama-3.1-8B']:
    AutoTokenizer.from_pretrained(m)
    AutoModelForCausalLM.from_pretrained(m, low_cpu_mem_usage=True)
"
```

## WVS Wave 7 CSV (manual download — not on HF)

```bash
# Get the WVS Wave 7 cross-national CSV from worldvaluessurvey.org
# (the dataset is freely available but requires email registration). Place it at:
mkdir -p data/wvs
# Then either:
#   - scp the CSV from your laptop:
#     scp WVS_Cross-National_Wave_7_csv_v6_0.csv \
#         <user>@nexus.umiacs.umd.edu:~/multicultural-llm-probe/data/wvs/
#   - or wget if you have a direct URL from WVS:
#     wget -O data/wvs/WVS_Cross-National_Wave_7_csv_v6_0.csv "<url>"
```

After it's in place, build the IW coordinates table once on the login node:

```bash
python3.12 analysis/culturemapping/compute_iw_coords.py
```

(That writes `data/iw_coordinates.csv`, ~100 KB.)

## First-run validation

Before kicking off long jobs, verify all four pieces work:

```bash
source env.sh

# 1. Eval (5 min on A5000 at 3B — confirms inference path works)
sbatch slurm/eval_job.sh
# wait, then check outputs/behavioral/normad_base.json exists and has predictions

# 2. CULNIG smoke test (5 min — just check it doesn't crash)
# Edit slurm/culnig_job.sh temporarily to add `--dataset-names normad` only
# and skip the normadcontrol + decide steps, then submit. Or just run it on
# the login node briefly:
python3.12 culnig/calc_neuron_score_normad.py --condition base --dataset-names normad
# Ctrl-C after a few batches print; you just want to confirm the model loads
# and dataloader iterates. Delete the partial output.
```

## Running the 8B pipeline

```bash
# 1. SFT training (24h)
SFT8B_ID=$(sbatch --parsable slurm/sft_8b_job.sh)

# 2. DPO training in parallel with SFT (24h)
sbatch slurm/dpo_8b_job.sh

# 3. SFT+DPO chained on SFT completion (24h, depends on SFT)
sbatch --dependency=afterok:$SFT8B_ID slurm/sftdpo_8b_job.sh

# 4. After all three training jobs finish, eval (CPU-cheap, ~3h × 4 conds)
sbatch slurm/eval_8b_job.sh

# 5. CULNIG scoring (~20h × 4 conds — heavy)
sbatch slurm/culnig_8b_job.sh

# 6. Analysis (CPU-only, runs on login node)
python3.12 analysis/heatmaps.py
python3.12 analysis/accuracy_deltas_bars.py
# ...etc. The existing analysis scripts read from outputs/behavioral and
# outputs/neurons; pass model-size-aware paths if you've added that flag.
```

Total wall-clock for an 8B pipeline run: roughly **5-6 days end-to-end**
(SFT 24h → SFT+DPO 24h serial; DPO 24h in parallel; eval 3h; CULNIG 20h; some
queue wait time on the class partition).

## Running the precision-matched fix on existing 3B checkpoints

The new `--precision matched_bf16` flag (default) loads all four conditions in
bf16 instead of mixing 4-bit and bf16. To rerun the 3B evaluation under
matched precision without retraining:

```bash
# Eval all 4 conditions at 3B in matched bf16 (default for new code path)
sbatch slurm/eval_job.sh
# Outputs land at outputs/behavioral/normad_{cond}.json — same path as before,
# so they'll overwrite the prior 3B QLoRA results. Back them up first if you
# want to keep both:
mkdir -p outputs/behavioral_qlora4bit_legacy
cp outputs/behavioral/normad_*.json outputs/behavioral_qlora4bit_legacy/
```

Same idea for CULNIG — the new default precision regime is matched-bf16; pass
`--precision qlora_4bit` explicitly if you want to reproduce the legacy
result set instead.

## Cleaning up between runs

Storage will fill fast. After confirming a result set is good and backed up:

```bash
# Delete the per-(neuron, country) score files for one model size
# (these are the 1.7 GB hogs, regenerable by rerunning CULNIG)
rm outputs/neurons/*_8b/normad_max_scores.json
rm outputs/neurons/*_8b/normadcontrol_max_scores.json
rm outputs/neurons/*_8b/countryrc_max_scores.json
# Keep all_neurons_normad_max.json — those are the small selected-neuron sets

# Or wholesale clear scratch HF cache after a run is done
rm -rf $HF_HOME/hub/models--meta-llama--Llama-3.1-8B
```

## Common gotchas

1. **HF cache filling home directory.** Single most common quota issue. Fix
   with `HF_HOME` override (see above) before *any* download.
2. **Network scratch not available for class accounts.** Use `/scratch0/$USER`
   if `/fs/nexus-scratch/$USER` doesn't exist. Local scratch has 90-day
   auto-delete — rsync outputs back to home or to a shared dir before they
   expire.
3. **Wall-time limits.** Class partition's `high` QoS caps at 24h; 8B SFT or
   DPO can occasionally hit this. If a job times out, the trainer's
   epoch-checkpointing means you can resume; pass `--resume_from_checkpoint`
   to the trainer by editing the relevant config or job script.
4. **Bitsandbytes + Llama 3.1 8B compat.** Some bitsandbytes versions choke on
   4-bit Llama 3.1 if `attn_implementation` defaults to flash. The CULNIG
   loader pins `attn_implementation="sdpa"`; if you see CUDA crashes during
   training, add the same to the trainer's `model = AutoModelForCausalLM...`
   call in `finetune/sft_train.py`.
5. **First inference at 8B is slow.** The Llama-3.1-8B repo has separate
   safetensor shards; first load on a fresh cache takes ~5-10 min while it
   downloads + verifies. Don't kill the job if it appears stuck on
   "Loading checkpoint shards" — it's downloading.
