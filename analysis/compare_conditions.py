"""Cross-condition behavioral analysis — produces the paper's Table 1.

For each (benchmark, condition):
  - Overall accuracy / win rate
  - Western vs. Non-Western breakdown
  - Delta from base (C1)
  - Western - Non-Western gap

Outputs:
  outputs/behavioral/table1.csv     # machine-readable
  outputs/behavioral/table1.md      # paper-ready
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL_DIR = PROJECT_ROOT / "outputs" / "behavioral"

CONDITIONS = ["base", "sft", "dpo", "instruct"]


def load(benchmark: str, cond: str) -> dict | None:
    path = BEHAVIORAL_DIR / f"{benchmark}_{cond}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def metric_keys(benchmark: str) -> tuple[str, str]:
    if benchmark != "normad":
        raise ValueError(f"unsupported benchmark: {benchmark}")
    return "accuracy_overall", "accuracy_by_group"


def build_rows(benchmark: str):
    overall_k, group_k = metric_keys(benchmark)
    base = load(benchmark, "base")
    base_overall = base[overall_k] if base else None
    base_w = base[group_k]["Western"] if base else None
    base_nw = base[group_k]["Non-Western"] if base else None

    rows = []
    for cond in CONDITIONS:
        d = load(benchmark, cond)
        if d is None:
            rows.append({"condition": cond, "missing": True})
            continue
        overall = d[overall_k]
        west = d[group_k]["Western"]
        nwest = d[group_k]["Non-Western"]
        rows.append({
            "condition": cond,
            "overall": overall,
            "western": west,
            "non_western": nwest,
            "gap_w_minus_nw": (west - nwest) if (west is not None and nwest is not None) else None,
            "delta_overall_from_base": (overall - base_overall) if (overall is not None and base_overall is not None) else None,
            "delta_nw_from_base": (nwest - base_nw) if (nwest is not None and base_nw is not None) else None,
        })
    return rows


def to_csv(rows_by_bench: dict, path: Path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "condition", "overall", "western", "non_western",
                    "gap_w_minus_nw", "delta_overall_from_base", "delta_nw_from_base"])
        for bench, rows in rows_by_bench.items():
            for r in rows:
                if r.get("missing"):
                    w.writerow([bench, r["condition"], "", "", "", "", "", ""])
                else:
                    w.writerow([bench, r["condition"], r["overall"], r["western"],
                                r["non_western"], r["gap_w_minus_nw"],
                                r["delta_overall_from_base"], r["delta_nw_from_base"]])


def fmt(x):
    if x is None:
        return "—"
    return f"{x:+.3f}" if isinstance(x, float) and abs(x) < 1 and "." in f"{x:.3f}" else f"{x:.3f}"


def to_md(rows_by_bench: dict, path: Path):
    lines = ["# Table 1 — Behavioral results across conditions\n"]
    for bench, rows in rows_by_bench.items():
        lines.append(f"\n## {bench.upper()}\n")
        lines.append("| Condition | Overall | Western | Non-Western | W-NW gap | ΔOverall vs. base | ΔNon-Western vs. base |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in rows:
            if r.get("missing"):
                lines.append(f"| {r['condition']} | (missing) | | | | | |")
                continue
            lines.append(
                f"| {r['condition']} | {fmt(r['overall'])} | {fmt(r['western'])} | "
                f"{fmt(r['non_western'])} | {fmt(r['gap_w_minus_nw'])} | "
                f"{fmt(r['delta_overall_from_base'])} | {fmt(r['delta_nw_from_base'])} |"
            )
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=["normad"])
    args = parser.parse_args()

    rows_by_bench = {b: build_rows(b) for b in args.benchmarks}
    csv_path = BEHAVIORAL_DIR / "table1.csv"
    md_path = BEHAVIORAL_DIR / "table1.md"
    to_csv(rows_by_bench, csv_path)
    to_md(rows_by_bench, md_path)
    print(f"Wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
