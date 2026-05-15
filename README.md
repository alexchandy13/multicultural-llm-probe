# Multicultural LLM Probe

Cultural norm drift in LLMs: disentangling the effects of SFT vs. DPO with a values-based extension of CULNIG.

## Research question

Does SFT vs. DPO differentially erode non-Western cultural norms and values in LLMs, and can this be explained at the neuron level using a values-based extension of CULNIG?

## Five conditions

| ID | Model | Training | Purpose |
|----|-------|----------|---------|
| C1 | Llama 3.2 3B base | — | Pre-alignment baseline |
| C2 | Llama 3.2 3B base + SFT on HH-RLHF chosen, 3 ep | SFT | Isolates SFT effect |
| C3 | Llama 3.2 3B base + DPO on HH-RLHF, 2 ep | DPO | Isolates DPO effect |
| C4 | Llama 3.2 3B base → SFT → DPO on HH-RLHF | sequential | Controlled SFT+DPO (real-world recipe) |
| C5 | Llama 3.2 3B Instruct (Meta) | proprietary SFT+DPO+RLHF | Real-world reference |

C2 and C3 share **the same HH-RLHF source data** (same 30k subsample, same seed) — so any contrast between them isolates the training method, not the data. C4 chains the two stages on top of each other on the same data, controlling what Meta's official Instruct release does with proprietary data.

### Alpaca robustness variant

To check robustness to the SFT dataset choice, we additionally train an
Alpaca-based variant of SFT and SFT+DPO. The DPO-only condition is unchanged
between setups since it always uses HH-RLHF.

| Condition | Model | Training | Purpose |
|----|-------|----------|---------|
| C2a | Llama 3.2 3B + Alpaca, 3 ep | SFT on Alpaca | Robustness check for SFT data |
| C4a | Llama 3.2 3B + Alpaca SFT + HH-RLHF DPO | Sequential, mixed-data | Robustness check for sequential pipeline |

The Alpaca SFT checkpoint was trained on a prior run; only the SFT+DPO stage
needs to run on Nexus:

```bash
sbatch slurm/sftdpo_alpaca_job.sh
# wait for completion
sbatch slurm/eval_alpaca_job.sh
sbatch slurm/culnig_alpaca_job.sh
python analysis/compare_conditions.py --setup both
```

The alpaca jobs (`*_alpaca_job.sh`) target the C2a/C4a conditions only — the
primary `eval_job.sh` and `culnig_job.sh` continue to operate on the HH-RLHF
conditions (C1–C5) unchanged. Analysis scripts accept `--setup {hhrlhf,alpaca,both}`,
defaulting to `hhrlhf` so prior figures and tables are reproduced byte-identically.

## Layout

```
data/         HF datasets (HH-RLHF, NormAd, BLEnD, CountryRC) + constructed NormAdctrl
models/       Base + Instruct model weights
checkpoints/  LoRA adapters per condition + epoch
finetune/     SFT, DPO, SFT+DPO training scripts and configs
evaluate/     NormAd eval (plus all-conditions wrapper). CARE dropped — see eval_all_conditions.py docstring.
culnig/       CULNIG core (clone of ynklab/CULNIG) + NormAd extensions
analysis/     Cross-condition comparison + neuron attribution + figures
slurm/        Job scripts for download/train/eval/CULNIG on UMIACS Nexus
outputs/      Behavioral JSON results, neuron files, paper figures
```

## Pipeline

```
Step 0  env_setup            login node, no GPU
Step 1  sft_job.sh           SFT on HH-RLHF chosen (30k), 3 ep, ~3-5h on A5000
Step 2  construct_normad_ctrl.py    strip cultural prefixes (rule-based)
Step 3  dpo_job.sh           DPO on HH-RLHF (30k), 2 ep, ~7-9h on A5000, parallel with Step 1
Step 3b sftdpo_job.sh        DPO on top of SFT, 2 ep, ~7-9h on A5000 — depends on Step 1
Step 4  eval_job.sh          NormAd on all five conditions (array 0-4)
Step 5  culnig_job.sh        gradient scoring on BLEnD (baseline) and NormAd (novel)
Step 6  analysis/*.py        results table + 3 figures
```

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

## CULNIG extension

The novel methodological contribution is `NormAdctrl`: NormAd scenarios with cultural context stripped. Without this, gradient-identified neurons bleed into general language understanding rather than culture-specific representation. Construction is rule-based regex on country/culture prefixes, with manual verification on 20-30 examples.

Two CULNIG runs are launched per condition:
- **5a:** Original BLEnD pipeline, unchanged. Baseline / fallback.
- **5b:** Extended NormAd pipeline. Same gradient scoring logic; only the dataset and control set are swapped.

## Hardware

UMIACS Nexus, RTX A5000 (24GB), SLURM. QLoRA 4-bit + adapters only — five LoRA adapters total fit in well under 1 GB of disk.

## Fallback

If fine-tuning collapses, the paper pivots to a C1-vs-C5-only comparison using stock models + original CULNIG. See plan §"Fallback Plan".
