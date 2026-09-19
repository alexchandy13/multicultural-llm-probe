"""US-default error rate broken out by group (Western vs Non-Western).

For each pipeline condition, computes:
  P(pred == us_pred | pred ≠ gold, group == G)   — error agreement
  P(pred == us_pred | pred == gold, group == G)  — correct agreement (baseline)

The correct-agreement baseline controls for the model's natural tendency to
agree with the US prediction independent of being wrong. The relevant signal
is error agreement >> correct agreement for Non-Western examples.

Also shows Western error agreement as a second reference point.

Prints tables for NormAd and BLEnD across the alignment pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL_DIR = PROJECT_ROOT / "outputs" / "behavioral"

PIPELINE = [
    ("Base",             "base"),
    ("SFT cult",         "sft_aya_cult"),
    ("SFT+DPO cult",     "sftdpo_aya_cult"),
    ("SFT nocult",       "sft_aya_nocult"),
    ("SFT+DPO nocult",   "sftdpo_aya_nocult"),
]


def us_match_rate(preds: list[dict], group: str, correct: bool, yn_only: bool) -> tuple[float, int]:
    """P(pred == us_pred) among correct (correct=True) or wrong (correct=False) examples."""
    subset = [p for p in preds
              if p.get("us_pred") is not None
              and p.get("country") != "US"
              and p.get("group") == group]
    if yn_only:
        subset = [p for p in subset if p["gold"] in ("yes", "no")]
    if correct:
        subset = [p for p in subset if p["pred"] == p["gold"]]
    else:
        subset = [p for p in subset if p["pred"] != p["gold"]]
    if not subset:
        return float("nan"), 0
    rate = sum(1 for p in subset if p["pred"] == p["us_pred"]) / len(subset)
    return rate, len(subset)


def load_normad(cond: str, ms: str) -> list[dict] | None:
    path = BEHAVIORAL_DIR / f"normad_{cond}_{ms}_nfs_mpw_usprobe_batch_calibrated.json"
    return json.loads(path.read_text())["predictions"] if path.exists() else None


def load_blend(cond: str, ms: str) -> list[dict] | None:
    path = BEHAVIORAL_DIR / f"blend_{cond}_{ms}_nfs_usprobe.json"
    return json.loads(path.read_text())["predictions"] if path.exists() else None


def fmt(v: float) -> str:
    return f"{v:.3f}" if not math.isnan(v) else "  n/a"


def print_table(rows: list, title: str) -> None:
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    hdr = (f"{'Condition':<22} "
           f"{'NW err':>8} {'NW err n':>9} "
           f"{'NW cor':>8} {'NW cor n':>9} "
           f"{'W err':>8} {'W err n':>8}")
    print(hdr)
    print(f"  (NW err = NW error US-match | NW cor = NW correct US-match [baseline] | W err = Western error US-match)")
    print(f"{'-'*90}")
    for label, nw_err, nw_err_n, nw_cor, nw_cor_n, w_err, w_err_n in rows:
        print(f"{label:<22} "
              f"{fmt(nw_err):>8} {nw_err_n:>9} "
              f"{fmt(nw_cor):>8} {nw_cor_n:>9} "
              f"{fmt(w_err):>8} {w_err_n:>8}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="8b", choices=["8b", "gemma4", "qwen35"])
    args = parser.parse_args()
    ms = args.model_size

    for benchmark, loader, yn_only in [
        ("NormAd", load_normad, True),
        ("BLEnD",  load_blend,  False),
    ]:
        rows = []
        for label, cond in PIPELINE:
            preds = loader(cond, ms)
            if preds is None:
                continue
            nw_err, nw_err_n = us_match_rate(preds, "Non-Western", correct=False, yn_only=yn_only)
            nw_cor, nw_cor_n = us_match_rate(preds, "Non-Western", correct=True,  yn_only=yn_only)
            w_err,  w_err_n  = us_match_rate(preds, "Western",     correct=False, yn_only=yn_only)
            rows.append((label, nw_err, nw_err_n, nw_cor, nw_cor_n, w_err, w_err_n))
        print_table(rows, f"{benchmark} — US-match rate by condition  [{ms}]")


if __name__ == "__main__":
    main()
