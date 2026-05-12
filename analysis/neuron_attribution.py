"""Neuron-level analysis across the four conditions.

For each of (BLEnD-identified, NormAd-identified) sets, per condition:
  - Total culture-neuron count
  - Mean attribution score (delta tensor masked to culture neurons)
  - Per-layer distribution
  - Pairwise overlap with the base-condition set (Jaccard)

Cross-source comparison: overlap of BLEnD-identified vs. NormAd-identified neurons
per condition — validates that the NormAd extension is not measuring random noise.

Outputs:
  outputs/neurons/attribution_summary.json
  outputs/neurons/attribution_per_layer.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEURONS_DIR = PROJECT_ROOT / "outputs" / "neurons"

CONDITIONS = ["base", "sft", "dpo", "instruct"]


def load_set(source: str, cond: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return (culture_mask [L, I], delta_scores [L, I]) for a given (source, condition)."""
    base_dir = NEURONS_DIR / f"{source}_{cond}"
    mask_path = base_dir / "culture_mask.pt"
    delta_path = base_dir / "scores_delta.pt"
    if not mask_path.exists() or not delta_path.exists():
        return None, None
    return torch.load(mask_path, map_location="cpu"), torch.load(delta_path, map_location="cpu")


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return float("nan")
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union else 0.0


def per_layer_counts(mask: torch.Tensor) -> list[int]:
    return [int(mask[L].sum().item()) for L in range(mask.shape[0])]


def per_layer_mean_score(mask: torch.Tensor, delta: torch.Tensor) -> list[float]:
    out = []
    for L in range(mask.shape[0]):
        sel = mask[L]
        out.append(float(delta[L, sel].mean().item()) if sel.any() else 0.0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["normad", "blend"])
    args = parser.parse_args()

    summary = {"per_source": {}, "cross_source_overlap": {}}

    layer_rows = []
    for source in args.sources:
        per_cond = {}
        masks = {}
        for cond in CONDITIONS:
            mask, delta = load_set(source, cond)
            if mask is None:
                per_cond[cond] = {"missing": True}
                continue
            masks[cond] = mask
            per_cond[cond] = {
                "total": int(mask.sum().item()),
                "mean_score": float(delta[mask].mean().item()) if mask.any() else 0.0,
                "per_layer_counts": per_layer_counts(mask),
                "per_layer_mean_score": per_layer_mean_score(mask, delta),
            }
            for L, (cnt, score) in enumerate(zip(per_cond[cond]["per_layer_counts"],
                                                  per_cond[cond]["per_layer_mean_score"])):
                layer_rows.append([source, cond, L, cnt, score])

        # Jaccard vs. base, per source
        overlaps = {}
        base_mask = masks.get("base")
        for cond in CONDITIONS:
            if cond == "base" or cond not in masks or base_mask is None:
                continue
            overlaps[f"{cond}_vs_base"] = jaccard(masks[cond], base_mask)
        per_cond["jaccard_vs_base"] = overlaps
        summary["per_source"][source] = per_cond

    # Cross-source overlap: NormAd vs. BLEnD, per condition
    if "normad" in args.sources and "blend" in args.sources:
        for cond in CONDITIONS:
            n_mask, _ = load_set("normad", cond)
            b_mask, _ = load_set("blend", cond)
            if n_mask is not None and b_mask is not None:
                summary["cross_source_overlap"][cond] = {
                    "jaccard_normad_vs_blend": jaccard(n_mask, b_mask),
                    "normad_only": int((n_mask & ~b_mask).sum().item()),
                    "blend_only": int((b_mask & ~n_mask).sum().item()),
                    "shared": int((n_mask & b_mask).sum().item()),
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
