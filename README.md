# Cultural Norm Drift in LLMs: Disentangling the Effects of SFT and DPO

Cultural drift in LLMs: disentangling the effects of SFT vs. DPO with a values-based extension of CULNIG.

## Four conditions

| ID | Internal name | Model | Training | Purpose |
|----|---------------|-------|----------|---------|
| C1 | `base`          | Llama 3.2 3B base | — | Pre-alignment baseline |
| C2 | `sft`    | Llama 3.2 3B base + SFT on Alpaca, 3 ep | SFT | Isolates SFT effect |
| C3 | `dpo`           | Llama 3.2 3B base + DPO on HH-RLHF, 2 ep | DPO | Isolates DPO effect |
| C4 | `sftdpo` | C2 adapter merged into base, then DPO on HH-RLHF, 2 ep | Sequential | Real-world SFT→DPO recipe |

SFT (C2) trains on `tatsu-lab/alpaca` (52K English instruction-response pairs). 
DPO (C3) and the DPO stage of C4 share the same HH-RLHF preference data (same 30K subsample, same seed)

## Layout

```
data/         HF datasets (HH-RLHF, NormAd, Alpaca, CountryRC) + constructed NormAdctrl
models/       Base model weights
checkpoints/  LoRA adapters per condition + epoch
finetune/     SFT (Alpaca), DPO (HH-RLHF), SFT+DPO training scripts and configs
evaluate/     NormAd eval (plus all-conditions wrapper)
culnig/       CULNIG core (clone of ynklab/CULNIG) + NormAd extensions
analysis/     Cross-condition comparison + neuron attribution + figures
slurm/        Job scripts for download/train/eval/CULNIG on UMIACS Nexus
outputs/      Behavioral JSON results, neuron files, paper figures
health/       Random checks used for debugging, checking validity of training
```

## Pipeline

```
Step 0  source env.sh                        Login node, no GPU
Step 1  slurm/download_data.sh               Pre-fetch HH-RLHF, Alpaca, NormAd, CountryRC, base model
Step 2  slurm/sft_job.sh                     SFT on Alpaca (52K), 3 ep, ~3-5h on A5000
Step 3  slurm/dpo_job.sh                     DPO on HH-RLHF (30K), 2 ep, ~7-9h — parallel with Step 2
Step 4  slurm/sftdpo_job.sh                  DPO on top of merged SFT adapter, 2 ep, ~7-9h — depends on Step 2
Step 5  slurm/eval_job.sh                    NormAd eval, 4-task array, one per condition
Step 6  slurm/culnig_job.sh                  CULNIG gradient scoring on (NormAd, NormAdctrl) per condition
Step 7  slurm/analysis_job.sh                Build IW coords + comparison tables + paper figures (CPU-only)
```

The NormAdctrl control set is **not** a separate build step — `culnig/dataset_ext.py` constructs it in-flight during Step 6 by reusing every NormAd item but stripping the country tag from the prompt scaffold (see CULNIG extension below).

`slurm/analysis_job.sh` is idempotent and bundles everything from the IW-coords build through every paper figure — re-run it any time the underlying outputs change. To regenerate a single figure ad hoc instead, call the corresponding script directly (e.g. `python3.12 analysis/heatmaps.py --figures asymmetry_attribution --subtract-control`).

## Quickstart on UMIACS Nexus

```bash
source env.sh
sbatch slurm/download_data.sh

# SFT and DPO in parallel; SFT+DPO chained on SFT's completion
SFT_ID=$(sbatch --parsable slurm/sft_job.sh)
sbatch slurm/dpo_job.sh
sbatch --dependency=afterok:$SFT_ID slurm/sftdpo_job.sh

# wait for the training jobs, then:
sbatch slurm/eval_job.sh
sbatch slurm/culnig_job.sh
sbatch slurm/analysis_job.sh
```

## Hardware

UMIACS Nexus, RTX A5000 (24GB), SLURM. QLoRA 4-bit + adapters only — four LoRA adapters total fit in well under 1 GB of disk.
