"""Cluster-representative probe analysis.

For each IW cluster, runs the same statistics as us_probe_analysis.py but using
the cluster's centroid-representative country as the probe country instead of
the US. For each example:

  correct          — pred == gold
  wrong_probe_match   — pred != gold AND pred == us_pred  (cluster-defaulting error)
  wrong_probe_diverge — pred != gold AND pred != us_pred  (other error)

Requires probe output files named:
  normad_{condition}_{model}_nfs_mpw_{country}probe.json
e.g.
  normad_base_8b_nfs_mpw_australiaprobe.json

These are produced by eval_normad.py with --neutral-fewshot --multi-prompt-word
--probe-country <country>.

The us_pred field in each prediction entry holds the probe country's prediction
(field name kept for compatibility with eval_normad.py).

Usage:
    python analysis/cluster_probe_analysis.py
    python analysis/cluster_probe_analysis.py --model 8b
    python analysis/cluster_probe_analysis.py --model gemma4 --condition base
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL = PROJECT_ROOT / "outputs" / "behavioral"

# Cluster representatives from centroid analysis (NormAd countries, real WVS data only)
CLUSTER_REPS = {
    "ProtestantEurope": "netherlands",
    "CatholicEurope":   "hungary",
    "Orthodox":         "bosnia_and_herzegovina",
    "LatinAmerica":     "mexico",
    "Confucian":        "taiwan",
    "SouthAsia":        "philippines",
    "AfricanIslamic":   "iran",
}

# Cluster display order
CLUSTER_ORDER = [
    "ProtestantEurope", "CatholicEurope",
    "Orthodox", "LatinAmerica", "Confucian",
    "SouthAsia", "AfricanIslamic",
]

CONDITIONS = ["base", "sft_aya_cult", "sft_aya_nocult", "sftdpo_aya_cult", "sftdpo_aya_nocult"]


def probe_slug(country: str) -> str:
    return country.lower().replace(" ", "_")


def load_probe_file(condition: str, model: str, country: str) -> list[dict] | None:
    slug = probe_slug(country)
    path = BEHAVIORAL / f"normad_{condition}_{model}_nfs_mpw_{slug}probe.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["predictions"]


def accuracy(preds: list[dict]) -> float:
    if not preds:
        return float("nan")
    return sum(1 for p in preds if p["pred"] == p["gold"]) / len(preds)


def probe_accuracy(preds: list[dict], probe_country_name: str) -> float:
    """Fraction where us_pred == gold (how often the probe country's answer is correct)."""
    eligible = [p for p in preds if p.get("us_pred") is not None and p["country"] != probe_country_name]
    if not eligible:
        return float("nan")
    return sum(1 for p in eligible if p["us_pred"] == p["gold"]) / len(eligible)


def default_rate_among_errors(preds: list[dict], probe_country_name: str) -> tuple[float, int, int, int]:
    """Returns (default_rate, n_correct, n_probe_match, n_other) for non-probe-country examples."""
    correct = wrong_match = wrong_diverge = 0
    for p in preds:
        if p.get("us_pred") is None or p["country"] == probe_country_name:
            continue
        if p["pred"] == p["gold"]:
            correct += 1
        elif p["pred"] == p["us_pred"]:
            wrong_match += 1
        else:
            wrong_diverge += 1
    wrong = wrong_match + wrong_diverge
    rate = wrong_match / wrong if wrong > 0 else float("nan")
    return rate, correct, wrong_match, wrong_diverge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="8b", choices=["8b", "gemma4"])
    parser.add_argument("--condition", default=None, help="Single condition to show (default: all)")
    args = parser.parse_args()

    conditions = [args.condition] if args.condition else CONDITIONS

    print(f"\nCluster-representative probe analysis  (model={args.model})")
    print("=" * 90)
    print(f"\n{'Condition':<22}  {'Cluster':>18}  {'Rep country':<26}  "
          f"{'Own acc':>8}  {'Probe acc':>9}  {'Default rate':>12}")
    print("-" * 90)

    for cond in conditions:
        first = True
        for cluster in CLUSTER_ORDER:
            rep = CLUSTER_REPS[cluster]
            preds = load_probe_file(cond, args.model, rep)
            if preds is None:
                label = cond if first else ""
                print(f"  {label:<20}  {cluster:>18}  {rep:<26}  [missing file]")
                first = False
                continue

            own_acc = accuracy(preds)
            pr_acc  = probe_accuracy(preds, rep)
            dr, n_c, n_m, n_d = default_rate_among_errors(preds, rep)

            label = cond if first else ""
            own_s  = f"{own_acc:.3f}" if own_acc == own_acc else "—"
            pr_s   = f"{pr_acc:.3f}"  if pr_acc  == pr_acc  else "—"
            dr_s   = f"{dr:.1%}"      if dr      == dr      else "—"
            print(f"  {label:<20}  {cluster:>18}  {rep:<26}  "
                  f"{own_s:>8}  {pr_s:>9}  {dr_s:>12}  "
                  f"(correct={n_c}, probe_match={n_m}, other={n_d})")
            first = False
        print()


if __name__ == "__main__":
    main()
