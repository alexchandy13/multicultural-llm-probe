"""Summarise every cluster/country membership file: raw vs residualized.

One row per (condition, model, dataset, grouping, raw|resid), with the numbers
that decide whether a grouping carries real structure:

  rank1%    share of squared magnitude in the leading component. The attribution
            takes a max over token positions and multiplies by the gold-label
            probability, so most neurons' per-group scores are one shared pattern
            times a per-group scalar. When rank1% is ~99 the overlap statistics
            below are forced regardless of the grouping, which is why the
            residualized rows are the ones to read.
  jaccard   mean pairwise overlap of the top-5% sets.
  spearman  mean off-diagonal rank correlation over ALL pool neurons, so it is
            threshold-free. Under a pure rank-1 model this is exactly 1.0.
  grp/neu   mean number of groups a selected neuron belongs to.
  iw_auc    P(a within-IW-cluster pair is more similar than a cross-cluster pair).
            0.5 means IW explains nothing. Only meaningful for country grouping;
            for cluster grouping the groups *are* the IW clusters.

    python analysis/membership_summary.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from culnig.clusters import cluster_of

NEURONS_ROOT = PROJECT_ROOT / "outputs" / "neurons"


def load_head(path: Path, probe: int = 1 << 23) -> dict:
    """Parse everything before the (huge) neurons array."""
    raw = path.read_text(encoding="utf-8")[:probe]
    i = raw.find('"neurons"')
    if i == -1:
        return json.loads(raw)
    return json.loads(raw[:i].rstrip().rstrip(",") + "}")


def load_matrix(path: Path):
    """Per-neuron x per-group score matrix, from the neurons array."""
    d = json.load(open(path))
    groups = d["clusters"]
    M = np.array([[n["scores"][g] for g in groups] for n in d["neurons"]])
    lay = np.array([n["layer_idx"] for n in d["neurons"]])
    return groups, M, lay, d


def iw_auc(groups, S) -> float | None:
    lab = {g: cluster_of(g) for g in groups}
    if any(v is None for v in lab.values()) or len(set(lab.values())) < 2:
        return None
    win, bet = [], []
    for a, b in itertools.combinations(range(len(groups)), 2):
        (win if lab[groups[a]] == lab[groups[b]] else bet).append(S[a, b])
    if not win or not bet:
        return None
    allv = np.concatenate([win, bet])
    r = allv.argsort().argsort().astype(float) + 1
    rw = r[:len(win)].sum()
    return (rw - len(win)*(len(win)+1)/2) / (len(win)*len(bet))


def summarise(path: Path):
    head = load_head(path)
    groups, M, lay, d = load_matrix(path)
    n_g = len(groups)

    sq = (M ** 2).sum()
    G = M.T @ M
    w, V = np.linalg.eigh(G)
    rank1 = 100 * w[-1] / w.sum() if w.sum() > 0 else float("nan")

    k = max(int(len(M) * 0.05), 1)
    sets = {i: set(np.argsort(-M[:, i])[:k]) for i in range(n_g)}
    J = np.eye(n_g)
    for a, b in itertools.combinations(range(n_g), 2):
        v = len(sets[a] & sets[b]) / len(sets[a] | sets[b])
        J[a, b] = J[b, a] = v
    jac = np.mean([J[a, b] for a, b in itertools.combinations(range(n_g), 2)])

    rk = np.argsort(np.argsort(M, axis=0), axis=0).astype(float)
    sp = np.corrcoef(rk, rowvar=False)
    spear = np.mean([sp[a, b] for a, b in itertools.combinations(range(n_g), 2)])

    hist = {int(x): y for x, y in head.get("membership_histogram", {}).items()}
    uniq = sum(hist.values()) or 1
    gpn = sum(x * y for x, y in hist.items()) / uniq

    return dict(n_groups=n_g, n_neurons=len(M), rank1=rank1, jaccard=jac,
                spearman=spear, gpn=gpn, iw=iw_auc(groups, J),
                excl=100 * hist.get(1, 0) / uniq)


def main():
    rows = []
    for p in sorted(NEURONS_ROOT.glob("*/[cq]*_membership_*_max.json")):
        cond_dir = p.parent.name
        name = p.name
        grouping = "country" if name.startswith("country") else "cluster"
        tag = name[len(grouping) + len("_membership_"):-len("_max.json")]
        resid = tag.endswith("_resid")
        ds = tag[:-len("_resid")] if resid else tag
        try:
            r = summarise(p)
        except Exception as e:
            print(f"  SKIP {cond_dir}/{name}: {type(e).__name__}: {e}")
            continue
        rows.append((cond_dir, ds, grouping, "resid" if resid else "raw", r))

    if not rows:
        print("No membership files found under outputs/neurons/*/")
        return

    hdr = (f"{'condition':26s} {'dataset':14s} {'grp':8s} {'ver':6s} "
           f"{'#grp':>5s} {'#neu':>8s} {'rank1%':>7s} {'jacc':>6s} "
           f"{'spear':>6s} {'grp/neu':>8s} {'excl%':>6s} {'iw_auc':>7s}")
    print(hdr); print("-" * len(hdr))
    for cond, ds, grp, ver, r in sorted(rows, key=lambda x: (x[1], x[2], x[3], x[0])):
        iw = f"{r['iw']:7.3f}" if r["iw"] is not None else "      -"
        print(f"{cond:26s} {ds:14s} {grp:8s} {ver:6s} {r['n_groups']:5d} "
              f"{r['n_neurons']:8d} {r['rank1']:7.2f} {r['jaccard']:6.3f} "
              f"{r['spearman']:6.3f} {r['gpn']:8.2f} {r['excl']:6.1f} {iw}")


if __name__ == "__main__":
    main()
