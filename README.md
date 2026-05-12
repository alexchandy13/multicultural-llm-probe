# Multicultural LLM Probe

Cultural norm drift in LLMs: disentangling the effects of SFT vs. DPO with a values-based extension of CULNIG.

## Research question

Does SFT vs. DPO differentially erode non-Western cultural norms and values in LLMs, and can this be explained at the neuron level using a values-based extension of CULNIG?

## Four conditions

| ID | Model | Training | Purpose |
|----|-------|----------|---------|
| C1 | Llama 3.2 3B base | — | Pre-alignment baseline |
| C2 | Llama 3.2 3B base + Alpaca, 3 ep | SFT | Isolates SFT effect |
| C3 | Llama 3.2 3B base + HH-RLHF, 2 ep | DPO | Isolates DPO effect |
| C4 | Llama 3.2 3B Instruct (Meta) | SFT+DPO+RLHF | Real-world reference |

## Layout

```
data/         HF + constructed datasets (NormAd, CARE, BLEnD, CountryRC, Alpaca, HH-RLHF, NormAdctrl)
models/       Base + Instruct model weights
checkpoints/  LoRA adapters per condition + epoch
finetune/     SFT + DPO training scripts and configs
evaluate/     NormAd + CARE eval, plus all-conditions wrapper
culnig/       CULNIG core (clone of ynklab/CULNIG) + NormAd extensions
analysis/     Cross-condition comparison + neuron attribution + figures
slurm/        Job scripts for download/train/eval/CULNIG on UMIACS Nexus
outputs/      Behavioral JSON results, neuron files, paper figures
```

## Pipeline

```
Step 0  env_setup            login node, no GPU
Step 1  sft_job.sh           SFT on Alpaca, 3 ep, ~6-8h on A5000
Step 2  construct_normad_ctrl.py    strip cultural prefixes (rule-based)
Step 3  dpo_job.sh           DPO on HH-RLHF, 2 ep, parallel with Step 1 if quota allows
Step 4  eval_job.sh          NormAd + CARE on all four conditions
Step 5  culnig_job.sh        gradient scoring on BLEnD (baseline) and NormAd (novel)
Step 6  analysis/*.py        results table + 3 figures
```

## Quickstart on UMIACS Nexus

```bash
source env.sh
sbatch slurm/download_data.sh
sbatch slurm/sft_job.sh
sbatch slurm/dpo_job.sh
# wait for both to finish, then:
sbatch slurm/eval_job.sh
sbatch slurm/culnig_job.sh
python analysis/compare_conditions.py
python analysis/neuron_attribution.py
python analysis/visualize.py
```

## CULNIG extension

The novel methodological contribution is `NormAdctrl`: NormAd scenarios with cultural context stripped. Without this, gradient-identified neurons bleed into general language understanding rather than culture-specific representation. Construction is rule-based regex on country/culture prefixes, with manual verification on 20-30 examples.

Two CULNIG runs are launched per condition:
- **5a:** Original BLEnD pipeline, unchanged. Baseline / fallback.
- **5b:** Extended NormAd pipeline. Same gradient scoring logic; only the dataset and control set are swapped.

## Hardware

UMIACS Nexus, RTX A5000 (24GB), SLURM. QLoRA 4-bit + adapters only stays well under the 30GB storage quota.

## Fallback

If fine-tuning collapses, the paper pivots to a C1-vs-C4-only comparison using stock models + original CULNIG. See plan §"Fallback Plan".
