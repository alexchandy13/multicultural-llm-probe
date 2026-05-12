# CULNIG extension

Upstream [ynklab/CULNIG](https://github.com/ynklab/CULNIG) is cloned into
[culnig/_upstream/](_upstream/) and left **pristine** — no in-place edits. The
extension lives entirely in the four files in this directory and re-uses
upstream's `calculate_scores` loop unchanged, per the plan's directive: "keep
gradient scoring logic completely unchanged — no modifications to the core
algorithm."

## What's here

| File | Purpose |
|---|---|
| [construct_normad_ctrl.py](construct_normad_ctrl.py) | Regex utilities for stripping country/demonym mentions from NormAd story text. Runs standalone to produce a manual-verification sample. |
| [dataset_ext.py](dataset_ext.py) | Monkey-patches upstream's `load_dataset_neuron_scores` to add a `normadcontrol` block (NormAd story stripped + country dropped from prompt). |
| [calc_neuron_score_normad.py](calc_neuron_score_normad.py) | Drives upstream's `calculate_scores` against a given condition: QLoRA-loads the (base + optional LoRA) Llama 3.2 3B, extends the model whitelist for Llama 3.2, writes scores under `outputs/neurons/{condition}/`. |
| [decide_culture_neurons_normad.py](decide_culture_neurons_normad.py) | Selection step: top-t% per module on (NormAd − NormAdctrl), minus top-r% on CountryRC. Output dirs adjusted to our `outputs/neurons/` tree. |

The same `calc_neuron_score_normad.py` runs both 5a (BLEnD) and 5b (NormAd) by
varying `--dataset-names`. This keeps the model loading, quantization, and
whitelist patches in one place.

## Why no in-place edits to upstream?

Two things need to change for our setup:
1. Llama-3.2-3B added to upstream's model whitelists (only Llama-3.1-8B-Instruct
   is currently listed for that branch — the architectures are otherwise compatible).
2. A `normadcontrol` block added to `dataset.py` (upstream stubs out the name in
   its `__main__` test but never implements the block).

Both are handled at import time by `dataset_ext.py` and `calc_neuron_score_normad.py`
via monkey-patching, so `culnig/_upstream/` can be re-pulled from git without
losing local modifications.

## Pipeline (one condition)

```bash
python culnig/calc_neuron_score_normad.py --condition sft --dataset-names normad
python culnig/calc_neuron_score_normad.py --condition sft --dataset-names normadcontrol
python culnig/decide_culture_neurons_normad.py --condition sft --dataset-names normad
```

Then the BLEnD baseline (Step 5a) for cross-source overlap:

```bash
python culnig/calc_neuron_score_normad.py --condition sft --dataset-names blend
python culnig/calc_neuron_score_normad.py --condition sft --dataset-names blendcontrol
python culnig/decide_culture_neurons_normad.py --condition sft --dataset-names blend
```

The SLURM wrapper [slurm/culnig_job.sh](../slurm/culnig_job.sh) runs all six commands
as a 4-task array, one per condition.

## Output layout

```
outputs/neurons/{condition}/
  normad_max_scores.json        # per-neuron scores on NormAd
  normadcontrol_max_scores.json # per-neuron scores on NormAdctrl
  blend_max_scores.json         # 5a baseline
  blendcontrol_max_scores.json  # 5a baseline
  countryrc_max_scores.json     # exclusion filter
  all_neurons_normad_max.json   # selected culture neurons (5b)
  all_neurons_blend_max.json    # selected culture neurons (5a)
```

## Fallback

If `dataset_ext.py`'s monkey-patching breaks (e.g., upstream refactors
`load_dataset_neuron_scores`), the BLEnD-only path still works — just drop
`normadcontrol` from the pipeline and run BLEnD/CountryRC. That matches the
plan's §"Fallback Plan".
