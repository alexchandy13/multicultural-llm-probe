"""US-default error rate broken out by group (Western vs Non-Western).

For each pipeline condition, computes:
  P(pred == us_pred | pred ≠ gold, group == G)

for G in {Non-Western, Western, All (non-US)}.

Chance baseline for yes/no binary: 0.500 (only one wrong answer exists).
Prints a table for NormAd and BLEnD across the alignment pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL_DIR = PROJECT_ROOT / "outputs" / "behavioral"

PIPELINE = [
    ("Base",     "base"),
    ("SFT cult", "sft_aya_cult"),
    ("SFT+DPO cult", "sftdpo_aya_cult"),
    ("SFT nocult",   "sft_aya_nocult"),
    ("SFT+DPO nocult", "sftdpo_aya_nocult"),
]


def us_default_err(preds: list[dict], group: str | None = None, yn_only: bool = True) -> tuple[float, int]:
    """Return (rate, n_errors) for US-default among errors, optionally filtered by group."""
    subset = [p for p in preds
              if p.get("us_pred") is not None
              and p["pred"] != p["gold"]
              and p.get("country") != "US"]
    if yn_only:
        subset = [p for p in subset if p["gold"] in ("yes", "no")]
    if group:
        subset = [p for p in subset if p.get("group") == group]
    if not subset:
        return float("nan"), 0
    rate = sum(1 for p in subset if p["pred"] == p["us_pred"]) / len(subset)
    return rate, len(subset)


def load_normad(cond: str, ms: str) -> list[dict] | None:
    path = BEHAVIORAL_DIR / f"normad_{cond}_{ms}_nfs_mpw_usprobe_batch_calibrated.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["predictions"]


def load_blend(cond: str, ms: str) -> list[dict] | None:
    path = BEHAVIORAL_DIR / f"blend_{cond}_{ms}_nfs_usprobe.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["predictions"]


def print_table(rows: list, title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"{'Condition':<22} {'NW errors':>10} {'NW rate':>9} {'W errors':>10} {'W rate':>9} {'chance':>8}")
    print(f"{'-'*70}")
    for label, nw_rate, nw_n, w_rate, w_n in rows:
        nw_str = f"{nw_rate:.3f}" if not math.isnan(nw_rate) else "  n/a"
        w_str  = f"{w_rate:.3f}"  if not math.isnan(w_rate)  else "  n/a"
        print(f"{label:<22} {nw_n:>10} {nw_str:>9} {w_n:>10} {w_str:>9} {'0.500':>8}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="8b", choices=["8b", "gemma4", "qwen35"])
    args = parser.parse_args()
    ms = args.model_size

    # NormAd
    normad_rows = []
    for label, cond in PIPELINE:
        preds = load_normad(cond, ms)
        if preds is None:
            continue
        nw_rate, nw_n = us_default_err(preds, group="Non-Western", yn_only=True)
        w_rate,  w_n  = us_default_err(preds, group="Western",     yn_only=True)
        normad_rows.append((label, nw_rate, nw_n, w_rate, w_n))
    print_table(normad_rows, f"NormAd — US-default rate among errors  [{ms}]")

    # BLEnD
    blend_rows = []
    for label, cond in PIPELINE:
        preds = load_blend(cond, ms)
        if preds is None:
            continue
        nw_rate, nw_n = us_default_err(preds, group="Non-Western", yn_only=False)
        w_rate,  w_n  = us_default_err(preds, group="Western",     yn_only=False)
        blend_rows.append((label, nw_rate, nw_n, w_rate, w_n))
    print_table(blend_rows, f"BLEnD — US-default rate among errors  [{ms}]")


if __name__ == "__main__":
    main()
