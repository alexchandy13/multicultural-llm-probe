"""Decide culture-norm neurons from NormAd score deltas.

Reads the scores_delta tensor from calc_neuron_score_normad.py and applies CULNIG's
thresholding policy (top-k per layer, then CountryRC-filter) to emit a binary mask
of culture-norm neurons.

CountryRC filtering: a neuron is kept only if its activation profile correlates with
country identity in the CountryRC probe set. We do not re-derive that filter here —
we expect upstream CULNIG to have produced a CountryRC neuron-id list per layer that
both BLEnD and NormAd runs share.

Usage:
    python culnig/decide_culture_neurons_normad.py --condition sft
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_country_rc_mask(path: Path, shape: tuple[int, int]) -> torch.Tensor:
    """Load the CountryRC-derived eligible-neuron mask, or fall back to "all eligible"."""
    if not path.exists():
        print(f"[warn] CountryRC mask not found at {path}; treating all neurons as eligible.")
        return torch.ones(shape, dtype=torch.bool)
    mask = torch.load(path, map_location="cpu")
    if mask.shape != shape:
        raise ValueError(f"CountryRC mask shape {mask.shape} != scores shape {shape}")
    return mask.bool()


def topk_per_layer(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Keep top-k neurons per layer by absolute score; return [layers, intermediate] bool mask."""
    mask = torch.zeros_like(scores, dtype=torch.bool)
    n_layers = scores.shape[0]
    for L in range(n_layers):
        if k >= scores.shape[1]:
            mask[L] = True
            continue
        _, idx = torch.topk(scores[L].abs(), k)
        mask[L, idx] = True
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["base", "sft", "dpo", "instruct"])
    parser.add_argument("--scores-dir", default=None,
                        help="Defaults to outputs/neurons/normad_{condition}/")
    parser.add_argument("--country-rc-mask", default=None,
                        help="Path to CountryRC eligibility mask (.pt). "
                             "Defaults to outputs/neurons/country_rc_mask.pt")
    parser.add_argument("--top-k-per-layer", type=int, default=200,
                        help="CULNIG default; tune downstream if needed.")
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir) if args.scores_dir else (
        PROJECT_ROOT / "outputs" / "neurons" / f"normad_{args.condition}"
    )
    delta = torch.load(scores_dir / "scores_delta.pt", map_location="cpu")

    rc_mask_path = Path(args.country_rc_mask) if args.country_rc_mask else (
        PROJECT_ROOT / "outputs" / "neurons" / "country_rc_mask.pt"
    )
    rc_mask = load_country_rc_mask(rc_mask_path, tuple(delta.shape))

    topk = topk_per_layer(delta, args.top_k_per_layer)
    culture_mask = topk & rc_mask

    # Per-layer counts and IDs
    n_layers, inter = culture_mask.shape
    per_layer = []
    for L in range(n_layers):
        ids = culture_mask[L].nonzero(as_tuple=True)[0].tolist()
        per_layer.append({
            "layer": L,
            "count": int(culture_mask[L].sum().item()),
            "neuron_ids": ids,
            "mean_score": float(delta[L, culture_mask[L]].mean().item()) if ids else 0.0,
        })

    torch.save(culture_mask, scores_dir / "culture_mask.pt")
    summary = {
        "condition": args.condition,
        "top_k_per_layer": args.top_k_per_layer,
        "total_culture_neurons": int(culture_mask.sum().item()),
        "per_layer": per_layer,
    }
    (scores_dir / "culture_neurons.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {scores_dir/'culture_neurons.json'} "
          f"(total = {summary['total_culture_neurons']})")


if __name__ == "__main__":
    main()
