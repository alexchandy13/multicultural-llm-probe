"""Select cluster-specific culture neurons.

This is the cluster-level analogue of upstream's decide_culture_specific_neuron.py,
which selects *country*-specific neurons. It is a different procedure from
decide_culture_neurons.py (the culture-general port in this repo): that one sums
every country into one score and takes the top 5% MLP / 1% attention. This one
keeps the country dimension, groups countries into Inglehart-Welzel clusters, and
adds the step that makes a neuron "specific" rather than merely active — a z-score
across clusters.

Per cluster C, mirroring upstream with cluster substituted for country:

  1. delta[n][c] = scores[n][c]/N_ds - control[n][c]/N_ctrl     (global denominators,
                                                                 as upstream does)
     cluster value = mean of delta[n][c] over the countries c in C
  2. all modules ranked in one pool (upstream folds attention into
     MLP_TARGET_MODULES and leaves ATTENTION_TARGET_MODULES empty)
  3. take the top CLUSTER_NEURON_PROPORTION
  4. drop the top COUNTRYRC_NEURON_PROPORTION by countryrc restricted to C
  5. keep only neurons whose cluster value is at least ZSCORE_THRESHOLD standard
     deviations above that neuron's mean across all clusters

Step 5 is the point of the method. Without it the selection returns whatever is
most active for the cluster, which is largely the same neurons for every cluster.
Step 4 matters more here than in the general selection: with 9 Confucian countries
instead of 69, a neuron that merely encodes the token "Japan" has far more leverage
on the ranking.

Cluster values are means over each cluster's countries, not sums, so that clusters
of different size are comparable in step 5 — EnglishSpeaking has 5 countries in
CulturalBench where AfricanIslamic has 10.

    python culnig/decide_cluster_neurons.py --condition base --model-size 8b \
        --dataset-names culturalbench
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from culnig.clusters import CLUSTERS, cluster_index, cluster_of, report_coverage
from culnig.score_io import dataset_ids as dataset_ids_of
from culnig.score_io import neuron_group_sums

NEURONS_ROOT = PROJECT_ROOT / "outputs" / "neurons"

# Upstream's culture-specific hyper-parameters. Note these are NOT the
# culture-general ones (0.05 / 0.01): per-unit selection takes a far smaller slice,
# because a neuron only has to beat the field for one cluster rather than overall.
CLUSTER_NEURON_PROPORTION = 0.003
COUNTRYRC_NEURON_PROPORTION = 0.01
ZSCORE_THRESHOLD = 0.5
# One pool: upstream puts gate_proj and q/k/v_proj in MLP_TARGET_MODULES together
# and leaves ATTENTION_TARGET_MODULES empty, so attention is ranked against MLP.
TARGET_MODULES = [
    "mlp.gate_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.k_proj",
]


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def _module_of(key: str) -> str:
    return "_".join(key.split("_")[:-2])


def _load_matrix(path, group_index, n_groups, logger):
    """Stream one score file into an (n_keys, n_groups) float64 matrix."""
    import numpy as np
    keys, flat, seen = neuron_group_sums(path, group_index, n_groups)
    mat = np.frombuffer(flat, dtype=np.float64).reshape(len(keys), n_groups).copy()
    logger.info(f"  {Path(path).name}: {len(keys)} neurons, {len(seen)} countries")
    return keys, mat, seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                        choices=["base", "dpo", "sft", "sftdpo",
                                 "sft_aya_cult", "sft_aya_nocult",
                                 "sftdpo_aya_cult", "sftdpo_aya_nocult"])
    parser.add_argument("--dataset-names", nargs="+", default=["normad"])
    parser.add_argument("--model-size", default="3b",
                        choices=["3b", "8b", "gemma4", "qwen35"])
    parser.add_argument("--yn-only", action="store_true",
                        help="Read normad_yn_max_scores.json instead of "
                             "normad_max_scores.json. Output gains a _yn suffix.")
    parser.add_argument("--clusters", nargs="+", default=CLUSTERS, choices=CLUSTERS,
                        help="Clusters to write selections for. Default: all 8.")
    parser.add_argument("--min-countries", type=int, default=2,
                        help="Skip a cluster represented by fewer than this many "
                             "countries in the data. A 1-country cluster is a "
                             "country selection wearing a cluster label, and its "
                             "z-score is driven by that one country. Default 2.")
    parser.add_argument("--zscore-threshold", type=float, default=ZSCORE_THRESHOLD)
    parser.add_argument(
        "--min-spread", type=float, default=0.0,
        help="Also require sd >= min_spread * |mean| across clusters. Default 0.0 "
             "reproduces upstream exactly. Upstream's z-score is purely relative, "
             "so a neuron that is equally active for every cluster still has half "
             "its clusters above its own mean, and a near-zero sd turns control "
             "noise into a large z — such a neuron is selected as 'specific' to "
             "whichever clusters happen to land high. A value like 0.1 requires "
             "real spread before a neuron can be called cluster-specific.",
    )
    parser.add_argument("--proportion", type=float, default=CLUSTER_NEURON_PROPORTION)
    args = parser.parse_args()

    import numpy as np
    logger = setup_logging()
    dataset_names = sorted(args.dataset_names)
    size_sfx = "" if args.model_size == "3b" else f"_{args.model_size}"
    cond_dir = NEURONS_ROOT / f"{args.condition}{size_sfx}"
    gi = cluster_index()
    ng = len(CLUSTERS)

    logger.info(f"Cluster-specific selection: condition={args.condition} "
                f"size={args.model_size} datasets={dataset_names}")
    logger.info(f"proportion={args.proportion} countryrc={COUNTRYRC_NEURON_PROPORTION} "
                f"z>={args.zscore_threshold}")

    keys: list[str] | None = None
    kidx: dict[str, int] = {}
    delta = None                      # (n_keys, 8) accumulated across datasets
    n_countries = np.zeros(ng)        # countries per cluster, max over datasets
    dataset_ids: dict[str, list] = defaultdict(list)

    for dataset_name in dataset_names:
        score_name = (f"{dataset_name}_yn"
                      if (args.yn_only and dataset_name == "normad") else dataset_name)
        score_path = cond_dir / f"{score_name}_max_scores.json"
        ctrl_path = cond_dir / f"{dataset_name}control_max_scores.json"

        # The country->cluster map has to be built from the countries actually in
        # the file, so stream the dataset file first with an empty map purely to
        # collect them? No — group_index is keyed by country name and unknown
        # countries are skipped, so a map over every known name works directly.
        group_index = {}
        for c, cl in ((c, cluster_of(c)) for c in _peek_countries(score_path)):
            if cl is not None:
                group_index[c] = gi[cl]

        by_cluster, unresolved = report_coverage(_peek_countries(score_path))
        if unresolved:
            logger.warning(f"  {score_name}: unresolved countries dropped: {unresolved}")
        for cl, cs in by_cluster.items():
            n_countries[gi[cl]] = max(n_countries[gi[cl]], len(cs))

        logger.info(f"Reading {score_name}")
        k_ds, m_ds, _ = _load_matrix(score_path, group_index, ng, logger)
        k_ct, m_ct, _ = _load_matrix(ctrl_path, group_index, ng, logger)

        score_ids = dataset_ids_of(score_path)
        ctrl_ids = dataset_ids_of(ctrl_path)
        for dname, ids in score_ids.items():
            dataset_ids[dname].extend(ids)
        for dname, ids in ctrl_ids.items():
            dataset_ids[dname].extend(ids)
        n_ds = len(score_ids[score_name])
        n_ctrl = len(ctrl_ids[f"{dataset_name}control"])
        logger.info(f"  N_ds={n_ds} N_ctrl={n_ctrl}")

        if keys is None:
            keys = k_ds
            kidx = {k: i for i, k in enumerate(keys)}
            delta = np.zeros((len(keys), ng))

        d = _align(m_ds, k_ds, keys, kidx, np) / n_ds
        d -= _align(m_ct, k_ct, keys, kidx, np) / n_ctrl
        delta += d

    # Mean over each cluster's countries, so cluster size does not set the scale.
    safe = np.where(n_countries > 0, n_countries, 1.0)
    delta /= safe

    # countryrc, restricted per cluster, same normalization
    crc_path = cond_dir / "countryrc_max_scores.json"
    crc_group = {}
    for c in _peek_countries(crc_path):
        cl = cluster_of(c)
        if cl is not None:
            crc_group[c] = gi[cl]
    crc_by_cluster, crc_unresolved = report_coverage(_peek_countries(crc_path))
    if crc_unresolved:
        logger.warning(f"  countryrc: unresolved countries dropped: {crc_unresolved}")
    logger.info("Reading countryrc")
    k_rc, m_rc, _ = _load_matrix(crc_path, crc_group, ng, logger)
    for dname, ids in dataset_ids_of(crc_path).items():
        dataset_ids[dname].extend(ids)
    crc_n = np.array([max(len(crc_by_cluster[c]), 1) for c in CLUSTERS], dtype=float)
    crc = _align(m_rc, k_rc, keys, kidx, np) / crc_n

    # Restrict to scored modules once; upstream ranks them in a single pool.
    keep = np.array([_module_of(k) in TARGET_MODULES for k in keys])
    logger.info(f"Neurons in target modules: {int(keep.sum())} / {len(keys)}")
    idx_pool = np.flatnonzero(keep)

    # z-score across clusters, per neuron — over POPULATED clusters only.
    # A cluster with no countries in this benchmark has delta 0 by construction,
    # and letting those zeros into the statistics drags the mean toward 0 and
    # inflates sd, so every active neuron clears the threshold for every cluster
    # it does have data for. BLEnD has no ProtestantEurope countries at all.
    populated = n_countries > 0
    logger.info(f"Populated clusters ({int(populated.sum())}/{ng}): "
                f"{[c for c in CLUSTERS if populated[gi[c]]]}")
    sub = delta[:, populated]
    mu = sub.mean(axis=1, keepdims=True)
    sd = sub.std(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (delta - mu) / sd, 0.0)

    out_suffix = "".join(dataset_names) + ("_yn" if args.yn_only else "")
    n_take = int(len(idx_pool) * args.proportion)
    n_crc = int(len(idx_pool) * COUNTRYRC_NEURON_PROPORTION)
    logger.info(f"Per cluster: top {n_take} candidates, excluding top {n_crc} countryrc")

    flat_mask = sd[:, 0] < args.min_spread * np.abs(mu[:, 0]) if args.min_spread > 0 \
        else np.zeros(len(keys), dtype=bool)
    if args.min_spread > 0:
        logger.info(f"min-spread {args.min_spread}: {int(flat_mask.sum())} neurons "
                    f"too flat across clusters to be called specific")

    for cl in args.clusters:
        g = gi[cl]
        if n_countries[g] < args.min_countries:
            logger.warning(f"{cl}: only {int(n_countries[g])} countries present "
                           f"(< --min-countries {args.min_countries}) — skipped")
            continue

        order = idx_pool[np.argsort(-delta[idx_pool, g], kind="stable")]
        cand = order[:n_take]
        crc_order = idx_pool[np.argsort(-crc[idx_pool, g], kind="stable")]
        excluded = set(crc_order[:n_crc].tolist())

        refined = []
        module_count: dict[str, int] = defaultdict(int)
        n_crc_dropped = n_z_dropped = 0
        for i in cand:
            if int(i) in excluded:
                n_crc_dropped += 1
                continue
            if z[i, g] < args.zscore_threshold or flat_mask[i]:
                n_z_dropped += 1
                continue
            key = keys[i]
            parts = key.split("_")
            refined.append({
                "module_name": "_".join(parts[:-2]),
                "layer_idx": int(parts[-2]),
                "neuron_idx": int(parts[-1]),
                "attribute_score": float(delta[i, g]),
                "zscore": float(z[i, g]),
                "scores": {c: float(delta[i, gi[c]]) for c in CLUSTERS},
            })
            module_count["_".join(parts[:-2])] += 1

        out_path = cond_dir / f"{cl}_neurons_{out_suffix}_max.json"
        out_path.write_text(json.dumps({
            "condition": args.condition,
            "cluster": cl,
            "countries": sorted(by_cluster.get(cl, [])),
            "dataset_ids": {k: v for k, v in dataset_ids.items()},
            "top_neurons": refined,
        }, indent=2))
        logger.info(f"{cl}: {len(refined)} neurons "
                    f"(dropped {n_crc_dropped} countryrc, {n_z_dropped} z<thr) "
                    f"| per module: {dict(module_count)} -> {out_path.name}")


def _align(mat, mat_keys, keys, kidx, np):
    """Reorder mat's rows onto `keys`. Fast path when the orders already match."""
    if mat_keys == keys:
        return mat
    out = np.zeros((len(keys), mat.shape[1]))
    rows = [kidx.get(k, -1) for k in mat_keys]
    for src, dst in enumerate(rows):
        if dst >= 0:
            out[dst] = mat[src]
    return out


def _peek_countries(path) -> list[str]:
    """Country keys of the first neuron, which every neuron shares."""
    import re
    from culnig.score_io import _open_text, resolve_scores_path
    resolved = resolve_scores_path(path)
    if resolved is None:
        raise FileNotFoundError(path)
    with _open_text(resolved) as fh:
        buf = ""
        while len(buf) < 1 << 22:
            chunk = fh.read(1 << 18)
            if not chunk:
                break
            buf += chunk
            i = buf.find('"neuron_scores"')
            if i == -1:
                continue
            m = re.search(r'"[A-Za-z_.]+_\d+_\d+"\s*:\s*\{([^{}]*)\}', buf[i:], re.S)
            if m:
                return re.findall(r'"([^"]+)"\s*:', m.group(1))
    raise ValueError(f"{path}: could not read per-country keys")


if __name__ == "__main__":
    main()
