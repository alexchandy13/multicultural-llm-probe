"""Jaccard overlap of culture-neuron sets across the four conditions.

For each pair (A, B) in {C1, C2, C3, C4}, compute:
    J(A, B) = |A ∩ B| / |A ∪ B|
where A, B are sets of (module_name, layer_idx, neuron_idx) triples taken from
the condition's all_neurons_normad_max.json file.

Answers the reviewer question raised by §5.3 of the paper: when we compare layer-
attribution heatmaps across conditions, are we tracking the same neurons or are
the selected populations themselves different? High Jaccard (≥0.5) supports the
"shared circuit, different amplification" interpretation; low Jaccard (≤0.2)
would mean the apparent C4 amplification is partly an artifact of which neurons
each condition's selection pipeline retains.

Reads:
  outputs/neurons/{base,sft,dpo,sftdpo}/all_neurons_normad_max.json

Outputs:
  - Set-size summary
  - 4×4 Jaccard matrix (markdown, pasteable into appendix)
  - Pairwise detail table (|A|, |B|, |A∩B|, |A∪B|, Jaccard)
  - Per-module-family breakdown (MLP vs attention) — flags whether overlap is
    uniformly high or concentrated in one architectural component

Usage:
    python3 analysis/neuron_overlap.py
    python3 analysis/neuron_overlap.py --model-size 8b
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEURONS_DIR = PROJECT_ROOT / "outputs" / "neurons"

CONDITIONS = ["base", "sft", "dpo", "sftdpo"]
COND_LABELS = {
    "base":   "C1 Base",
    "sft":    "C2 SFT",
    "dpo":    "C3 DPO",
    "sftdpo": "C4 SFT+DPO",
}

MLP_MODULES = {"mlp.gate_proj"}
ATTN_MODULES = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}


def load_neuron_set(cond: str, size_suffix: str = "") -> set[tuple[str, int, int]] | None:
    """Return {(module_name, layer_idx, neuron_idx)} for one condition, or None
    if the all_neurons_normad_max.json file is missing."""
    path = NEURONS_DIR / f"{cond}{size_suffix}" / "all_neurons_normad_max.json"
    if not path.exists():
        return None
    neurons = json.loads(path.read_text()).get("top_neurons", [])
    return {(n["module_name"], n["layer_idx"], n["neuron_idx"]) for n in neurons}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["3b", "8b"], default="3b")
    args = parser.parse_args()
    size_suffix = "_8b" if args.model_size == "8b" else ""

    sets = {}
    for cond in CONDITIONS:
        s = load_neuron_set(cond, size_suffix)
        if s is None:
            print(f"# WARN: missing all_neurons_normad_max.json for {cond}", file=sys.stderr)
            continue
        sets[cond] = s

    if len(sets) < 2:
        sys.exit("Need at least 2 condition neuron sets to compute overlap.")

    # === Set sizes ===
    print(f"\n# Culture-neuron set sizes\n")
    print(f"| Condition | # neurons | # MLP | # attention |")
    print(f"|---|---:|---:|---:|")
    for cond in CONDITIONS:
        if cond not in sets:
            continue
        s = sets[cond]
        mlp = sum(1 for m, _, _ in s if m in MLP_MODULES)
        attn = sum(1 for m, _, _ in s if m in ATTN_MODULES)
        print(f"| {COND_LABELS[cond]} | {len(s):,} | {mlp:,} | {attn:,} |")

    # === Jaccard matrix ===
    print(f"\n# Jaccard overlap matrix\n")
    header = "| | " + " | ".join(COND_LABELS[c] for c in CONDITIONS if c in sets) + " |"
    sep = "|---|" + "|".join("---:" for c in CONDITIONS if c in sets) + "|"
    print(header)
    print(sep)
    for c1 in CONDITIONS:
        if c1 not in sets:
            continue
        row = [COND_LABELS[c1]]
        for c2 in CONDITIONS:
            if c2 not in sets:
                continue
            if c1 == c2:
                row.append("1.000")
            else:
                row.append(f"{jaccard(sets[c1], sets[c2]):.3f}")
        print(f"| {row[0]} | {' | '.join(row[1:])} |")

    # === Pairwise detail ===
    print(f"\n# Pairwise detail\n")
    print(f"| Pair | |A| | |B| | |A∩B| | |A∪B| | Jaccard |")
    print(f"|---|---:|---:|---:|---:|---:|")
    available = [c for c in CONDITIONS if c in sets]
    for c1, c2 in combinations(available, 2):
        a, b = sets[c1], sets[c2]
        inter, union = len(a & b), len(a | b)
        j = inter / union if union > 0 else 0.0
        print(f"| {COND_LABELS[c1]} vs {COND_LABELS[c2]} | {len(a):,} | {len(b):,} | "
              f"{inter:,} | {union:,} | {j:.3f} |")

    # === Per-module-family Jaccard ===
    # Tells us whether overlap is uniform across MLP and attention or concentrated
    # in one. The paper's §5.3 amplification claim is grounded if both families
    # show high overlap.
    print(f"\n# Per-module-family Jaccard (overlap restricted to each module class)\n")
    print(f"| Pair | MLP Jaccard | Attention Jaccard |")
    print(f"|---|---:|---:|")
    for c1, c2 in combinations(available, 2):
        a, b = sets[c1], sets[c2]
        a_mlp = {t for t in a if t[0] in MLP_MODULES}
        b_mlp = {t for t in b if t[0] in MLP_MODULES}
        a_attn = {t for t in a if t[0] in ATTN_MODULES}
        b_attn = {t for t in b if t[0] in ATTN_MODULES}
        print(f"| {COND_LABELS[c1]} vs {COND_LABELS[c2]} | "
              f"{jaccard(a_mlp, b_mlp):.3f} | {jaccard(a_attn, b_attn):.3f} |")

    print(f"\nInterpretation:")
    print(f"  J ≥ 0.5 → mostly shared circuit; attribution differences are amplification, not")
    print(f"           re-selection. Supports the paper's §5.3 'shared shape, different")
    print(f"           magnitude' interpretation of Figure 2.")
    print(f"  J ≤ 0.2 → mostly disjoint sets; cross-condition heatmap comparisons need a")
    print(f"           caveat that the selected populations themselves differ.")
    print(f"  Between → mixed interpretation; pair-by-pair detail above is the key table.")


if __name__ == "__main__":
    main()
