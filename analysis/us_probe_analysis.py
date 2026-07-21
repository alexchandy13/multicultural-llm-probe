"""US-probe matching analysis.

For each model/condition, loads the *_usprobe.json result files and asks:
among examples the model got wrong, what fraction did it predict the same
thing it would have predicted for "United States"?

Three-way split per example (non-US only):
  correct          — pred == gold
  wrong_us_match   — pred != gold AND pred == us_pred  (US-defaulting error)
  wrong_us_diverge — pred != gold AND pred != us_pred  (other error)

The ratio wrong_us_match / (wrong_us_match + wrong_us_diverge) is the
headline "US-default rate among errors". A high rate means the model is
wrong in the same direction as the US prediction.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL = PROJECT_ROOT / "outputs" / "behavioral"
IW_CSV = PROJECT_ROOT / "data" / "iw_coordinates.csv"


def load_iw_clusters() -> dict[str, str]:
    clusters: dict[str, str] = {}
    for row in csv.DictReader(open(IW_CSV)):
        if row["normad_country"]:
            clusters[row["normad_country"].lower().replace(" ", "_")] = row["cluster"]
    return clusters


def normalize(s: str) -> str:
    return s.lower().replace(" ", "_").replace("-", "_")


def cluster_of(country: str, clusters: dict[str, str]) -> str:
    return clusters.get(normalize(country), "Other")


def analyze(preds: list[dict], clusters: dict[str, str]) -> dict:
    by_cluster: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)

    for p in preds:
        if p["country"] == "US":
            continue
        us_pred = p.get("us_pred")
        if us_pred is None:
            continue
        gold = p["gold"]
        pred = p["pred"]
        cl = cluster_of(p["country"], clusters)

        if pred == gold:
            key = "correct"
        elif pred == us_pred:
            key = "wrong_us_match"
        else:
            key = "wrong_us_diverge"

        totals[key] += 1
        by_cluster[cl][key] += 1

    return {"totals": dict(totals), "by_cluster": {k: dict(v) for k, v in by_cluster.items()}}


def us_default_rate(counts: dict[str, int]) -> float | None:
    wrong = counts.get("wrong_us_match", 0) + counts.get("wrong_us_diverge", 0)
    if wrong == 0:
        return None
    return counts.get("wrong_us_match", 0) / wrong


def print_summary(label: str, counts: dict[str, int]) -> None:
    n = sum(counts.values())
    if n == 0:
        print(f"  {label}: no data")
        return
    correct = counts.get("correct", 0)
    wum = counts.get("wrong_us_match", 0)
    wud = counts.get("wrong_us_diverge", 0)
    wrong = wum + wud
    udr = us_default_rate(counts)
    acc = correct / n
    udr_str = f"{udr:.1%}" if udr is not None else "—"
    print(f"  {label:<22s}  n={n:4d}  acc={acc:.3f}  wrong={wrong:4d}  "
          f"us_match={wum:4d}  us_diverge={wud:4d}  us_default_rate={udr_str}")


def main():
    clusters = load_iw_clusters()

    models = [("8b", "Llama 8B"), ("gemma4", "Gemma4")]
    conditions = ["base", "sft", "dpo", "sftdpo"]
    shots = [("yn_usprobe", "0-shot"), ("fs2_yn_usprobe", "fs2")]

    for shot_sfx, shot_label in shots:
        print(f"\n{'='*80}")
        print(f"  {shot_label}  —  US-default rate among errors")
        print(f"{'='*80}")

        for mkey, mlabel in models:
            print(f"\n{mlabel}")
            print("-" * 70)

            for cond in conditions:
                p = BEHAVIORAL / f"normad_{cond}_{mkey}_{shot_sfx}.json"
                if not p.exists():
                    print(f"  {cond:<10s}  [missing: {p.name}]")
                    continue

                data = json.loads(p.read_text())
                preds = data["predictions"]
                result = analyze(preds, clusters)
                totals = result["totals"]
                by_cluster = result["by_cluster"]

                print(f"\n  condition={cond}")
                print_summary("overall", totals)

                cluster_order = [
                    "EnglishSpeaking", "ProtestantEurope", "CatholicEurope",
                    "Orthodox", "LatinAmerica", "Confucian",
                    "SouthAsia", "AfricanIslamic", "Other",
                ]
                for cl in cluster_order:
                    if cl in by_cluster:
                        print_summary(cl, by_cluster[cl])


if __name__ == "__main__":
    main()
