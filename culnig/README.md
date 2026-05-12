# CULNIG extension

Original CULNIG (Culture Neuron Identification via Gradient) lives upstream at
[ynklab/CULNIG](https://github.com/ynklab/CULNIG). Clone it into `culnig/_upstream/`:

```bash
git clone https://github.com/ynklab/CULNIG culnig/_upstream
```

The three scripts in this directory are the novel extension described in the research
plan §"Step 5b — Extended CULNIG with NormAd":

| Script | Purpose |
|---|---|
| `construct_normad_ctrl.py` | Strip cultural prefixes from NormAd scenarios to create the control set (NormAdctrl), analogous to BLEnDctrl. |
| `calc_neuron_score_normad.py` | Gradient-score MLP neurons on NormAd vs. NormAdctrl per condition. Mirrors upstream `calc_neuron_score.py` — only dataset loading and output dir change. |
| `decide_culture_neurons_normad.py` | Apply CULNIG's thresholding to the NormAd scores to pick culture-norm neurons. Mirrors upstream `decide_culture_general_neurons.py`. |

The BLEnD pipeline (`5a`) runs the upstream scripts unchanged; only `5b` lives here.

**Important:** keep the core gradient-scoring logic identical to upstream. The only
swaps are (a) NormAd in place of BLEnD and (b) NormAdctrl in place of BLEnDctrl. Anything
else — CountryRC filtering, thresholding policy, layer aggregation — stays as-is so
results are directly comparable.
