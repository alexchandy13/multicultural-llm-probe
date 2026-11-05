#!/usr/bin/env python3
"""Derive a stage-1 baseline file from a legacy _usprobe run, without a GPU.

The old layout scored the baseline and the US swap in one run, so a
*_usprobe.json holds both: `pred`/`scores` are the baseline (true country in
the prompt) and `us_pred`/`us_scores` are the swap. Only the swap half was ever
wrong -- it used a different prompt scheme and a raw CLI country string -- so
the baseline half can be reused as-is and the eval does not need re-running.

This writes the same file with the probe fields dropped and `_usprobe` removed
from the name, which is exactly what stage 2 (add_probe) reads. Dropping the
fields rather than leaving them matters: a file named as a baseline that still
carried stale us_pred would be read as a probe file by anything that checks for
the key.

Costs no GPU time. For the four off-the-shelf instruct models across three
benchmarks that is 12 eval runs avoided, including BLEnD's ~40 min each.

  python scripts/baseline_from_usprobe.py outputs/behavioral/*_usprobe_raw.json
  python scripts/baseline_from_usprobe.py --dry-run outputs/behavioral/normad_*_usprobe_raw.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Written by the probe pass, not the baseline pass.
PROBE_ROW_FIELDS = ("us_pred", "us_scores")
PROBE_TOP_FIELDS = ("probe_country", "probe_sample_frac", "probe_n_scored")


def baseline_name(name: str) -> str:
    """Strip _usprobe, keeping any _raw/_sys tail that says how it was built."""
    m = re.search(r"_usprobe((?:_raw|_sys)*)", name)
    if not m:
        raise ValueError(f"{name}: not a _usprobe file")
    return f"{name[:m.start()]}{m.group(1)}.json"


def convert(src: Path, dry_run: bool, force: bool) -> str:
    dst = src.with_name(baseline_name(src.name))
    if dst.exists() and not force:
        return f"skip   {dst.name} (exists; --force to overwrite)"

    data = json.loads(src.read_text())
    rows = data["predictions"]
    missing = sum(1 for r in rows if r.get("pred") is None)
    if missing:
        return f"REFUSE {src.name}: {missing}/{len(rows)} rows have no baseline pred"

    stripped = 0
    out_rows = []
    for r in rows:
        row = {k: v for k, v in r.items() if k not in PROBE_ROW_FIELDS}
        stripped += len(r) - len(row)
        out_rows.append(row)

    out = {k: v for k, v in data.items() if k not in PROBE_TOP_FIELDS}
    out["predictions"] = out_rows
    # Recorded so the file's origin is not guesswork later: it was derived, not
    # produced by an eval run of its own.
    out["derived_from"] = src.name

    if dry_run:
        return f"would write {dst.name}  ({len(out_rows)} rows, {stripped} probe fields dropped)"
    dst.write_text(json.dumps(out))
    return f"wrote  {dst.name}  ({len(out_rows)} rows, {stripped} probe fields dropped)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path, help="*_usprobe*.json files")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing baseline file.")
    args = ap.parse_args()

    bad = 0
    for src in args.files:
        if not src.exists():
            print(f"missing {src}")
            bad += 1
            continue
        try:
            msg = convert(src, args.dry_run, args.force)
        except ValueError as e:
            print(f"skip   {e}")
            continue
        print(msg)
        if msg.startswith("REFUSE"):
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
