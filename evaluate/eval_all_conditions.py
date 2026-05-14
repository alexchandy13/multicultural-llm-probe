"""Run NormAd across all four conditions and emit a unified results table.

Each (benchmark, condition) sub-eval is delegated to eval_normad.py via subprocess
so we don't keep four models in memory simultaneously. The final aggregate table
lives at outputs/behavioral/summary.json + summary.md.

CARE was dropped from the plan: its only cultures are Arab/Chinese/Japanese (no
Western coverage, so it can't answer the Western-vs-Non-Western RQ), its prompts
are in non-English (out of Llama 3.2 3B's primary language), and the canonical
eval requires an LLM judge. NormAd alone covers the primary contrast.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evaluate._common import PROJECT_ROOT, conditions_from_env


BEHAVIORAL_DIR = PROJECT_ROOT / "outputs" / "behavioral"


def run_one(benchmark: str, condition: str):
    if benchmark != "normad":
        raise ValueError(f"unsupported benchmark: {benchmark}")
    cmd = [sys.executable, str(PROJECT_ROOT / "evaluate" / "eval_normad.py"),
           "--condition", condition]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def summarize():
    summary = {}
    for benchmark in ("normad",):
        summary[benchmark] = {}
        for cond in conditions_from_env():
            path = BEHAVIORAL_DIR / f"{benchmark}_{cond}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            summary[benchmark][cond] = {
                "overall": data.get("accuracy_overall"),
                "Western": data.get("accuracy_by_group", {}).get("Western"),
                "Non-Western": data.get("accuracy_by_group", {}).get("Non-Western"),
            }

    out_json = BEHAVIORAL_DIR / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    lines = ["# Behavioral results\n"]
    for benchmark, by_cond in summary.items():
        lines.append(f"\n## {benchmark.upper()}\n")
        lines.append("| Condition | Overall | Western | Non-Western | Gap |")
        lines.append("|---|---|---|---|---|")
        for cond, m in by_cond.items():
            w, n = m.get("Western"), m.get("Non-Western")
            gap = (w - n) if (w is not None and n is not None) else None
            def fmt(x): return f"{x:.3f}" if isinstance(x, float) else "—"
            lines.append(f"| {cond} | {fmt(m.get('overall'))} | {fmt(w)} | {fmt(n)} | {fmt(gap)} |")
    out_md = BEHAVIORAL_DIR / "summary.md"
    out_md.write_text("\n".join(lines))
    print(f"Wrote {out_json} and {out_md}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-eval", action="store_true", help="Only re-summarize from existing JSON.")
    parser.add_argument("--benchmarks", nargs="+", default=["normad"])
    parser.add_argument("--conditions", nargs="+", default=None)
    args = parser.parse_args()

    conditions = args.conditions or conditions_from_env()
    if not args.skip_eval:
        for cond in conditions:
            for bench in args.benchmarks:
                run_one(bench, cond)
    summarize()


if __name__ == "__main__":
    main()
