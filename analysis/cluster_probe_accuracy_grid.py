"""Cluster-default rate beside accuracy, for every condition x cluster cell.

Two numbers per cell:

  CDR  cluster-default rate among errors -- of the model's errors, the fraction
       whose prediction matches what it predicts when that cluster's
       representative country is swapped into the prompt. Computed over ALL
       examples, with the rep country substituted everywhere.
  Acc  accuracy on the examples whose OWN country belongs to that cluster,
       example-weighted, from the unswapped predictions.

The two are deliberately different slices, so read a cell as "how well does it
do on this cluster's questions" next to "how much does this cluster's country
drive its errors" -- not as two views of one set of rows.

Both use the shot variant whose probe grid is complete (NormAd _fs2_yn,
CulturalBench _nfs4, BLEnD _nfs), so CDR and Acc always come from the same
prompt condition.

    python analysis/cluster_probe_accuracy_grid.py
    python analysis/cluster_probe_accuracy_grid.py --benchmark blend
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.cluster_probe_analysis import (CLUSTER_ORDER, CLUSTER_REPS,
                                             CONDITIONS, PROBE_SUFFIX,
                                             default_rate_among_errors,
                                             load_probe_file)

# CLUSTER_REPS covers the seven non-US clusters; the US is the EnglishSpeaking
# representative and its probe file is the *_usprobe.json one, which
# load_probe_file already yields for the slug "us".
ORDER = ["EnglishSpeaking"] + list(CLUSTER_ORDER)
REPS = {"EnglishSpeaking": "us", **CLUSTER_REPS}
from analysis.cluster_accuracy_pipeline import cluster_acc, find

SHORT = {"EnglishSpeaking": "Engl", "ProtestantEurope": "Prot",
         "CatholicEurope": "Cath",
         "Orthodox": "Orth", "LatinAmerica": "Lati", "Confucian": "Conf",
         "SouthAsia": "Sout", "AfricanIslamic": "Afri"}
BENCHES = ["normad", "culturalbench", "blend"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", nargs="*", default=BENCHES, choices=BENCHES)
    ap.add_argument("--model", nargs="*", default=["8b", "gemma4"],
                    choices=["8b", "gemma4"])
    ap.add_argument("--min-n", type=int, default=30,
                    help="Blank the accuracy of a cluster with fewer than this "
                         "many examples.")
    args = ap.parse_args()

    for bench in args.benchmark:
        sfx = PROBE_SUFFIX[bench]
        for model in args.model:
            rows = []
            for cond in CONDITIONS:
                acc = cluster_acc(find(bench, cond, model, sfx))
                cells = []
                for cl in ORDER:
                    pr = load_probe_file(cond, model, REPS[cl],
                                         benchmark=bench)
                    cdr = default_rate_among_errors(pr, REPS[cl])[0] \
                        if pr else None
                    a = acc.get(cl) if acc else None
                    cells.append((cdr, a))
                if any(c != (None, None) for c in cells):
                    rows.append((cond, cells))
            if not rows:
                continue

            print(f"\n### {bench}{sfx} {model}   CDR = cluster-default rate "
                  f"among errors,  Acc = accuracy on that cluster's questions")
            print(" " * 20 + "  ".join(f"{SHORT[c]:^13s}" for c in ORDER))
            print(f"{'condition':20s}" + "  ".join(f"{'CDR':>6s}{'Acc':>7s}"
                                                   for _ in ORDER))
            for cond, cells in rows:
                out = f"{cond:20s}"
                for cdr, a in cells:
                    out += "  " + (f"{100*cdr:6.1f}" if cdr is not None else "     .")
                    out += (f"{a[0]:7.1f}" if a and a[1] >= args.min_n else "      .")
                print(out)
            ns = cluster_acc(find(bench, "base", model, sfx))
            if ns:
                print("  n per cluster: " + "  ".join(
                    f"{SHORT[c]}={ns[c][1]}" for c in ORDER if c in ns))


if __name__ == "__main__":
    main()
