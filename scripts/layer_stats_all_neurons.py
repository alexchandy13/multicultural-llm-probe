"""Per-layer attribution stats over ALL neurons, not just the selected culture ones.

The all_neurons_* files only contain the selected neurons, so any figure drawn from
them describes the selection. This computes the same per-layer quantities across
every scored neuron, which gives the baseline the selection is standing out from.

The attribution is the same delta decide_culture_neurons.py ranks on:

    delta[key] = sum(dataset scores)/n_dataset - sum(control scores)/n_control

Streaming, so it peaks well under 1 GB on multi-GB inputs. Results cache to
outputs/neurons/{cond}_{size}/layer_stats_{dataset}.json, a few KB, so figures can
be redrawn without re-reading the scores.

    python scripts/layer_stats_all_neurons.py --model-size 8b \
        --conditions base sft_aya_cult sft_aya_nocult sftdpo_aya_cult sftdpo_aya_nocult \
        --datasets normad culturalbench blend
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEURONS = PROJECT_ROOT / "outputs" / "neurons"
sys.path.insert(0, str(PROJECT_ROOT))

from culnig.score_io import dataset_ids as dataset_ids_of  # noqa: E402
from culnig.score_io import neuron_sums, scores_exist  # noqa: E402

# Matches MLP_SAVE_MODULES / ATTENTION_SAVE_MODULES in decide_culture_neurons.py,
# so "all neurons" means all neurons that were eligible to be selected.
SAVE_MODULES = {"mlp.gate_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}

# dataset -> (score file stem, control file stem, key used inside dataset_ids)
DATASETS = {
    "normad":        ("normad_yn", "normadcontrol"),
    "culturalbench": ("culturalbench", "culturalbenchcontrol"),
    "blend":         ("blend", "blendcontrol"),
}


def layer_of(key: str) -> tuple[str, int] | None:
    parts = key.split("_")
    if len(parts) < 3:
        return None
    module = "_".join(parts[:-2])
    if module not in SAVE_MODULES:
        return None
    try:
        return module, int(parts[-2])
    except ValueError:
        return None


def compute(cond: str, size: str, dataset: str) -> dict | None:
    cond_dir = NEURONS / f"{cond}_{size}"
    score_ds, ctrl_ds = DATASETS[dataset]
    score_p = cond_dir / f"{score_ds}_max_scores.json"
    ctrl_p = cond_dir / f"{ctrl_ds}_max_scores.json"
    if not scores_exist(score_p) or not scores_exist(ctrl_p):
        print(f"  [skip] {cond}_{size}/{dataset}: score files not present")
        return None

    n_s = len(dataset_ids_of(score_p)[score_ds])
    n_c = len(dataset_ids_of(ctrl_p)[ctrl_ds])
    print(f"  {cond}_{size}/{dataset}: n_dataset={n_s} n_control={n_c}", flush=True)

    s_sums = neuron_sums(score_p)
    c_sums = neuron_sums(ctrl_p)

    per_layer = defaultdict(lambda: {"count": 0, "sum": 0.0})
    for key, total in s_sums.items():
        lm = layer_of(key)
        if lm is None:
            continue
        delta = total / n_s - c_sums.get(key, 0.0) / n_c
        slot = per_layer[lm[1]]
        slot["count"] += 1
        slot["sum"] += delta
    out = {str(l): {"count": v["count"], "sum": v["sum"],
                    "mean": v["sum"] / v["count"] if v["count"] else 0.0}
           for l, v in sorted(per_layer.items())}
    return {"condition": cond, "model_size": size, "dataset": dataset,
            "n_dataset": n_s, "n_control": n_c,
            "n_neurons": sum(v["count"] for v in out.values()), "per_layer": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--model-size", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--force", action="store_true", help="recompute even if the cache exists")
    args = ap.parse_args()

    for dataset in args.datasets:
        print(f"=== {dataset} ===")
        for cond in args.conditions:
            cache = NEURONS / f"{cond}_{args.model_size}" / f"layer_stats_{dataset}.json"
            if cache.exists() and not args.force:
                print(f"  [cached] {cache.relative_to(PROJECT_ROOT)}")
                continue
            res = compute(cond, args.model_size, dataset)
            if res is None:
                continue
            cache.write_text(json.dumps(res, indent=2))
            print(f"  wrote {cache.relative_to(PROJECT_ROOT)} "
                  f"({res['n_neurons']:,} neurons)", flush=True)


if __name__ == "__main__":
    main()
