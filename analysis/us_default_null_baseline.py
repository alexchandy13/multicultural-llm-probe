"""Compare the US-default match rate to a per-scenario null baseline.

For each NormAd scenario the same situation is asked about multiple countries, each
with its own gold answer. The null baseline for a given prediction is: if you picked a
random OTHER country's gold answer for that same scenario, how often would it match
the model's prediction? If the US-default rate is well above this null, the model is
genuinely US-biased rather than just matching the majority answer pattern.

Aligns predictions to NormAd source rows by index (eval_normad.py iterates in dataset
order; with --neutral-fewshot all non-neutral examples are included without exclusions).

Usage:
    python analysis/us_default_null_baseline.py
    python analysis/us_default_null_baseline.py --model-size 8b
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL_DIR = PROJECT_ROOT / "outputs" / "behavioral"

HOLDOUT_COUNTRIES = {
    "Syria", "Indonesia", "Colombia",
    "Austria", "north_macedonia", "Sweden",
}

CONDITIONS = {
    "base":              "Base",
    "sft_aya_cult":      "SFT-Cult",
    "sft_aya_nocult":    "SFT-NoCult",
    "sftdpo_aya_cult":   "SFT+DPO-Cult",
    "sftdpo_aya_nocult": "SFT+DPO-NoCult",
}


def _gold_label(ex: dict) -> str:
    for key in ("Gold Label", "gold_label", "label", "answer", "normative"):
        if key in ex:
            v = str(ex[key]).strip().lower()
            return {"yes": "yes", "no": "no", "neutral": "neutral",
                    "1": "yes", "0": "no", "neither": "neutral"}.get(v, v)
    return ""


def _country(ex: dict) -> str:
    ALIASES = {
        "united_states_of_america": "US", "usa": "US",
        "south_korea": "South_Korea", "north_korea": "North_Korea",
    }
    for key in ("country", "Country", "nation", "culture"):
        if key in ex:
            raw = str(ex[key]).strip()
            return ALIASES.get(raw.lower(), raw)
    return ""


def _scenario_text(ex: dict) -> str:
    for key in ("Story", "story", "scenario", "situation", "text"):
        if key in ex:
            return str(ex[key]).strip()
    return ""


def load_normad_dataset():
    from datasets import load_dataset
    ds = load_dataset("akhilayerukola/NormAd")
    for split in ("test", "validation", "train"):
        if split in ds:
            return ds[split]
    raise ValueError(f"No usable split in {list(ds.keys())}")


def build_scenario_golds(ds) -> dict[str, dict[str, str]]:
    """Build {scenario_text: {country: gold}} for all non-neutral examples."""
    sg: dict[str, dict[str, str]] = defaultdict(dict)
    for ex in ds:
        gold = _gold_label(ex)
        if gold == "neutral":
            continue
        scenario = _scenario_text(ex)
        country  = _country(ex)
        if scenario and country and gold in ("yes", "no"):
            sg[scenario][country] = gold
    return sg


def align_predictions_to_dataset(ds, pred_file: Path) -> list[tuple[dict, dict]]:
    """Return [(pred, dataset_row)] aligned by index (neutral examples skipped in both)."""
    preds = json.loads(pred_file.read_text())["predictions"]

    # Replicate the eval loop filter: skip neutral gold rows
    ds_rows = [ex for ex in ds if _gold_label(ex) not in ("neutral",)]

    if len(ds_rows) != len(preds):
        raise ValueError(
            f"Length mismatch: dataset has {len(ds_rows)} non-neutral rows "
            f"but prediction file has {len(preds)} entries.\n"
            f"File: {pred_file.name}"
        )
    return list(zip(preds, ds_rows))


def compute_rates(aligned: list[tuple[dict, dict]],
                  scenario_golds: dict[str, dict[str, str]]) -> dict:
    us_matches   = []
    null_matches = []

    for pred, row in aligned:
        if pred.get("us_pred") is None:
            continue  # no US probe for this example
        p     = pred["pred"]
        us_p  = pred["us_pred"]
        c     = _country(row)
        scen  = _scenario_text(row)

        us_matches.append(1 if p == us_p else 0)

        # Null baseline: match rate against all non-US, non-current countries for this scenario
        others = {k: v for k, v in scenario_golds.get(scen, {}).items()
                  if k != c and k != "US"}
        if others:
            null_matches.append(sum(1 for g in others.values() if p == g) / len(others))

    return {
        "us_match_rate":   sum(us_matches) / len(us_matches) if us_matches else float("nan"),
        "null_match_rate": sum(null_matches) / len(null_matches) if null_matches else float("nan"),
        "n_us":   len(us_matches),
        "n_null": len(null_matches),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="8b", choices=["8b", "gemma4", "qwen35"])
    args = parser.parse_args()
    ms = args.model_size

    print("Loading NormAd dataset...")
    ds = load_normad_dataset()
    print(f"  {len(ds):,} total rows")

    print("Building scenario→country→gold map...")
    scenario_golds = build_scenario_golds(ds)
    n_scenarios = len(scenario_golds)
    n_single = sum(1 for g in scenario_golds.values() if len(g) == 1)
    print(f"  {n_scenarios:,} scenarios ({n_scenarios - n_single:,} with 2+ countries)")

    print(f"\n{'Condition':<22} {'US-Match':>9} {'Null-Match':>10} {'Delta':>7}")
    print("-" * 53)

    for cond, label in CONDITIONS.items():
        pred_file = BEHAVIORAL_DIR / f"normad_{cond}_{ms}_nfs_mpw_usprobe_batch_calibrated.json"
        if not pred_file.exists():
            print(f"  {label:<20}  MISSING")
            continue
        try:
            aligned = align_predictions_to_dataset(ds, pred_file)
        except ValueError as e:
            print(f"  {label:<20}  ERROR: {e}")
            continue

        r = compute_rates(aligned, scenario_golds)
        delta = r["us_match_rate"] - r["null_match_rate"]
        print(f"  {label:<20} {r['us_match_rate']:>9.3f} {r['null_match_rate']:>10.3f} {delta:>+7.3f}")


if __name__ == "__main__":
    main()
