"""US-default rate through the training pipeline.

UDR is the fraction of the model's ERRORS whose prediction matches what the
model predicts when the country in the prompt is swapped for the US:

    UDR = P(pred == us_pred | pred != gold)

Read alone it is misleading, because a model that answers the same thing to
everything agrees with its own counterfactual by construction. Two companions
are printed with it:

  chance  the agreement two independent draws from the error-conditioned
          marginals would reach, sum_c P(pred=c|err) * P(us_pred=c|err) --
          i.e. Cohen's kappa expected agreement
  kappa   (UDR - chance) / (1 - chance), the share of the available headroom
          above chance that is actually taken up

A rising UDR with a flat kappa is label collapse, not growing US-centrism. Flip
is the unconditional P(pred != us_pred) over ALL examples, which has no
selection effect at all.

US-anchored examples are dropped throughout: an example whose country already
is the US matches its own US counterfactual trivially.

    python analysis/udr_pipeline.py
    python analysis/udr_pipeline.py --shot zero
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.cluster_accuracy_pipeline import (BENCH_BY_SHOT, PIPELINES, find)

US_ALIASES = {"us", "usa", "u.s.", "u.s.a.", "united_states",
              "united states", "united_states_of_america",
              "united states of america", "america"}


def is_us(country: str) -> bool:
    return str(country).strip().lower().replace("-", "_") in US_ALIASES


def kappa_se(path: Path, n_boot: int = 600, seed: int = 0) -> float:
    """Bootstrap SE of kappa, resampling errors with replacement.

    Needed because n_errors ranges from ~170 (NormAd gemma4) to ~9000 (BLEnD),
    so the same kappa gap is decisive in one cell and noise in another.
    """
    rng = np.random.default_rng(seed)
    pr = [x for x in json.load(open(path))["predictions"]
          if x.get("us_pred") is not None and not is_us(x.get("country", ""))]
    err = [x for x in pr if x["pred"] != x["gold"]]
    labs = sorted({x["pred"] for x in err} | {x["us_pred"] for x in err})
    idx = {l: i for i, l in enumerate(labs)}
    a = np.array([idx[x["pred"]] for x in err])
    b = np.array([idx[x["us_pred"]] for x in err])

    def k(a, b):
        n, L = len(a), len(labs)
        pe = ((np.bincount(a, minlength=L) / n)
              * (np.bincount(b, minlength=L) / n)).sum()
        return ((a == b).mean() - pe) / (1 - pe) if pe < 1 else np.nan

    draws = rng.integers(0, len(a), (n_boot, len(a)))
    return float(np.nanstd([k(a[s], b[s]) for s in draws]))


def udr(path: Path | None):
    """(UDR%, chance%, kappa, flip%, n_errors) over non-US examples."""
    if path is None:
        return None
    pr = [x for x in json.load(open(path))["predictions"]
          if x.get("us_pred") is not None and not is_us(x.get("country", ""))]
    if not pr:
        return None
    flip = 100 * sum(x["pred"] != x["us_pred"] for x in pr) / len(pr)
    err = [x for x in pr if x["pred"] != x["gold"]]
    if not err:
        return None
    agree = sum(x["pred"] == x["us_pred"] for x in err) / len(err)
    p = Counter(x["pred"] for x in err)
    u = Counter(x["us_pred"] for x in err)
    chance = sum(p[c] * u[c] for c in set(p) | set(u)) / len(err) ** 2
    kappa = (agree - chance) / (1 - chance) if chance < 1 else float("nan")
    return 100 * agree, 100 * chance, kappa, flip, len(err)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default="fs2", choices=list(BENCH_BY_SHOT))
    ap.add_argument("--pipeline", nargs="*", default=list(PIPELINES),
                    choices=list(PIPELINES))
    ap.add_argument("--bootstrap", action="store_true",
                    help="Add a bootstrap SE on kappa. Slow but necessary to "
                         "read any gap between stages.")
    args = ap.parse_args()

    for bench, mid in BENCH_BY_SHOT[args.shot]:
        for name in args.pipeline:
            spec = PIPELINES[name]
            for size in spec["sizes"]:
                lines = []
                for arm, stages in spec["arms"].items():
                    for label, stage in zip(spec["xlabels"], stages):
                        path = (find(bench, stage, size, mid)
                                if spec["slot"] == "cond"
                                else find(bench, "base", stage, mid))
                        r = udr(path)
                        if r is None:
                            continue
                        if label == spec["xlabels"][0] and lines:
                            continue        # base is shared between arms
                        tag = label if label == spec["xlabels"][0] \
                            else f"{label} {arm}"
                        se = kappa_se(path) if args.bootstrap else None
                        lines.append((tag, r, se))
                if not lines:
                    continue
                print(f"\n### {bench}{mid} {size} ({name})   "
                      f"non-US examples only")
                print(f"{'stage':18s} {'UDR':>6s} {'chance':>7s} {'kappa':>7s}"
                      + (f" {'+-SE':>6s}" if args.bootstrap else "")
                      + f" {'flip':>6s} {'n_err':>7s}")
                for tag, (a, c, k, f, n), se in lines:
                    print(f"{tag:18s} {a:6.1f} {c:7.1f} {k:7.3f}"
                          + (f" {se:6.3f}" if se is not None else "")
                          + f" {f:6.1f} {n:7d}")


if __name__ == "__main__":
    main()
