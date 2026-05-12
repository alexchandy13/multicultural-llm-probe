"""Neuron-level analysis across the four conditions.

Reads CULNIG-format JSON from outputs/neurons/{condition}/all_neurons_{source}_max.json
where source is 'normad' (5b) or 'blend' (5a).

Computes:
  - Total culture-neuron count per (source, condition)
  - Mean attribution score
  - Per-layer distribution (count + mean score)
  - Pairwise Jaccard vs. base condition
  - Cross-source overlap (NormAd vs. BLEnD per condition) — validates the extension

Outputs:
  outputs/neurons/attribution_summary.json
  outputs/neurons/attribution_per_layer.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEURONS_DIR = PROJECT_ROOT / "outputs" / "neurons"

CONDITIONS = ["base", "sft", "dpo", "instruct"]


def load_neurons(source: str, cond: str) -> list[dict] | None:
    """Return the list of (module, layer, neuron, score) tuples for a (source, cond) run."""
    path = NEURONS_DIR / cond / f"all_neurons_{source}_max.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("top_neurons", [])


def neuron_key(n: dict) -> tuple[str, int, int]:
    return (n["module_name"], n["layer_idx"], n["neuron_idx"])


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def per_layer(neurons: list[dict]) -> dict[int, dict]:
    by_layer: dict[int, list[float]] = defaultdict(list)
    for n in neurons:
        by_layer[n["layer_idx"]].append(n["attribute_score"])
    return {
        L: {"count": len(scores), "mean_score": sum(scores) / len(scores)}
        for L, scores in by_layer.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["normad", "blend"])
    args = parser.parse_args()

    summary = {"per_source": {}, "cross_source_overlap": {}}
    layer_rows = []

    keys_by_source_cond: dict[tuple[str, str], set] = {}

    for source in args.sources:
        per_cond = {}
        for cond in CONDITIONS:
            neurons = load_neurons(source, cond)
            if neurons is None:
                per_cond[cond] = {"missing": True}
                continue
            keys = {neuron_key(n) for n in neurons}
            keys_by_source_cond[(source, cond)] = keys
            scores = [n["attribute_score"] for n in neurons]
            layer_stats = per_layer(neurons)

            per_cond[cond] = {
                "total": len(neurons),
                "mean_score": sum(scores) / len(scores) if scores else 0.0,
                "per_layer": layer_stats,
            }
            for L, st in layer_stats.items():
                layer_rows.append([source, cond, L, st["count"], st["mean_score"]])

        base_keys = keys_by_source_cond.get((source, "base"))
        if base_keys is not None:
            per_cond["jaccard_vs_base"] = {
                cond: jaccard(keys_by_source_cond[(source, cond)], base_keys)
                for cond in CONDITIONS
                if cond != "base" and (source, cond) in keys_by_source_cond
            }
        summary["per_source"][source] = per_cond

    # Cross-source overlap (NormAd vs. BLEnD per condition)
    if "normad" in args.sources and "blend" in args.sources:
        for cond in CONDITIONS:
            n_keys = keys_by_source_cond.get(("normad", cond))
            b_keys = keys_by_source_cond.get(("blend", cond))
            if n_keys is not None and b_keys is not None:
                summary["cross_source_overlap"][cond] = {
                    "jaccard_normad_vs_blend": jaccard(n_keys, b_keys),
                    "normad_only": len(n_keys - b_keys),
                    "blend_only": len(b_keys - n_keys),
                    "shared": len(n_keys & b_keys),
                }

    out_json = NEURONS_DIR / "attribution_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    out_csv = NEURONS_DIR / "attribution_per_layer.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "condition", "layer", "count", "mean_score"])
        w.writerows(layer_rows)

    print(f"Wrote {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
