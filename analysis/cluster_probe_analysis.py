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


def load_probe_file(condition: str, model: str, country: str,
                    benchmark: str = "normad") -> list[dict] | None:
    slug = probe_slug(country)
    if benchmark == "normad":
        path = BEHAVIORAL / f"normad_{condition}_{model}_nfs_mpw_{slug}probe.json"
    elif benchmark == "culturalbench":
        path = BEHAVIORAL / f"culturalbench_{condition}_{model}_nfs_{slug}probe.json"
    else:
        path = BEHAVIORAL / f"blend_{condition}_{model}_nfs_{slug}probe.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["predictions"]


def accuracy(preds: list[dict]) -> float:
    if not preds:
        return float("nan")
    return sum(1 for p in preds if p["pred"] == p["gold"]) / len(preds)


def same_country(a: str, b: str) -> bool:
    return a.lower().replace(" ", "_") == b.lower().replace(" ", "_")


def probe_accuracy(preds: list[dict], probe_country_name: str) -> float:
    """Fraction where us_pred == gold (how often the probe country's answer is correct)."""
    eligible = [p for p in preds if p.get("us_pred") is not None and not same_country(p["country"], probe_country_name)]
    if not eligible:
        return float("nan")
    return sum(1 for p in eligible if p["us_pred"] == p["gold"]) / len(eligible)


def default_rate_among_errors(preds: list[dict], probe_country_name: str) -> tuple[float, int, int, int]:
    """Returns (default_rate, n_correct, n_probe_match, n_other) for non-probe-country examples."""
    correct = wrong_match = wrong_diverge = 0
    for p in preds:
        if p.get("us_pred") is None or same_country(p["country"], probe_country_name):
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


def load_us_probe_file(condition: str, model: str,
                       benchmark: str = "normad") -> list[dict] | None:
    if benchmark == "normad":
        path = BEHAVIORAL / f"normad_{condition}_{model}_nfs_mpw_usprobe.json"
    elif benchmark == "culturalbench":
        path = BEHAVIORAL / f"culturalbench_{condition}_{model}_nfs_usprobe.json"
    else:
        path = BEHAVIORAL / f"blend_{condition}_{model}_nfs_usprobe.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["predictions"]


def us_default_rate_overall(preds: list[dict]) -> float | None:
    """Overall US default rate among errors (all non-US examples)."""
    match = diverge = 0
    for p in preds:
        if p.get("us_pred") is None or p["country"] == "US":
            continue
        if p["pred"] != p["gold"]:
            if p["pred"] == p["us_pred"]:
                match += 1
            else:
                diverge += 1
    wrong = match + diverge
    return match / wrong if wrong > 0 else None


def us_probe_accuracy(preds: list[dict]) -> float:
    """Fraction of non-US examples where us_pred == gold."""
    eligible = [p for p in preds if p.get("us_pred") is not None and p["country"] != "US"]
    if not eligible:
        return float("nan")
    return sum(1 for p in eligible if p["us_pred"] == p["gold"]) / len(eligible)


def fmt_diff(val: float, pct: bool = False) -> str:
    if val != val:
        return "—"
    s = f"{val:+.1%}" if pct else f"{val:+.3f}"
    return s


def print_table(conditions: list[str], model: str, benchmark: str) -> None:
    print(f"\nCluster-representative probe analysis  (benchmark={benchmark}, model={model})")
    print("=" * 105)
    print(f"\n{'Condition':<22}  {'Cluster':>18}  {'Rep country':<26}  "
          f"{'Probe-Own':>10}  {'Probe-US':>9}  {'ClustDR-USDR':>13}")
    print("-" * 105)

    for cond in conditions:
        us_preds = load_us_probe_file(cond, model, benchmark)
        us_dr  = us_default_rate_overall(us_preds) if us_preds else float("nan")
        us_acc = us_probe_accuracy(us_preds)        if us_preds else float("nan")

        first = True
        for cluster in CLUSTER_ORDER:
            rep = CLUSTER_REPS[cluster]
            preds = load_probe_file(cond, model, rep, benchmark)
            if preds is None:
                continue

            own_acc = accuracy(preds)
            pr_acc  = probe_accuracy(preds, rep)
            dr, *_  = default_rate_among_errors(preds, rep)

            label = cond if first else ""
            d_own = fmt_diff(pr_acc - own_acc)
            d_us  = fmt_diff(pr_acc - us_acc)
            d_dr  = fmt_diff(dr - us_dr, pct=True)
            print(f"  {label:<20}  {cluster:>18}  {rep:<26}  "
                  f"{d_own:>10}  {d_us:>9}  {d_dr:>13}")
            first = False
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="8b", choices=["8b", "gemma4"])
    parser.add_argument("--benchmark", default="normad", choices=["normad", "blend", "culturalbench", "both", "all"])
    parser.add_argument("--condition", default=None, help="Single condition to show (default: all)")
    args = parser.parse_args()

    conditions = [args.condition] if args.condition else CONDITIONS
    if args.benchmark == "both":
        benchmarks = ["normad", "blend"]
    elif args.benchmark == "all":
        benchmarks = ["normad", "blend", "culturalbench"]
    else:
        benchmarks = [args.benchmark]

    for bm in benchmarks:
        print_table(conditions, args.model, bm)


if __name__ == "__main__":
    main()
