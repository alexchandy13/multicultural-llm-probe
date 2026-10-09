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
  2. two ranking pools, as the culture-general selection uses: mlp.gate_proj
     separately from self_attn.{q,k,v}_proj
  3. take the top 5% of the MLP pool and the top 1% of the attention pool
  4. drop the top 1% of each pool by countryrc restricted to C
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

# Thresholds and pool structure are deliberately identical to
# decide_culture_neurons.py (the culture-general selection), NOT to upstream's
# culture-specific script. Upstream used one pool at 0.3% with attention folded
# into the MLP list; keeping the general selection's two pools at 5%/1% instead
# means the cluster sets and the general set are directly comparable — same
# candidate count, same per-family quotas — so a difference between them is a
# difference in the clustering, not in the hyper-parameters.
#
# Consequence worth knowing: each cluster now takes the SAME slice the general
# selection takes overall (24,903 candidates at 8b), so 8 clusters span up to 8x
# the general selection's footprint. The overlap metrics in the membership file
# are what tell you whether that is 8 distinct sets or 8 views of one.
MLP_CULTURE_NEURON_PROPORTION = 0.05
MLP_COUNTRYRC_NEURON_PROPORTION = 0.01
ATTENTION_CULTURE_NEURON_PROPORTION = 0.01
ATTENTION_COUNTRYRC_NEURON_PROPORTION = 0.01
ZSCORE_THRESHOLD = 0.5
MLP_TARGET_MODULES = ["mlp.gate_proj"]
ATTENTION_TARGET_MODULES = ["self_attn.v_proj", "self_attn.q_proj", "self_attn.k_proj"]


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def _slug(group: str) -> str:
    """Filename-safe group token. Country keys carry spaces ("South Korea") and the
    NormAd ones carry underscores and non-ASCII ("türkiye"), so normalise rather
    than writing a path with a space in it."""
    safe = group.replace(" ", "_")
    # Belt and braces: a country name that still carries something path-hostile
    # must not reach the filesystem. Türkiye already round-trips via
    # unescape_key, but a stray separator would otherwise raise OSError 22
    # only after the whole run has finished computing.
    return "".join(ch if (ch.isalnum() or ch in "_-.") else "_" for ch in safe)


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
    parser.add_argument("--clusters", nargs="+", default=None,
                        help="Groups to write selections for. Default: all. With "
                             "--by-country these are country keys, otherwise IW "
                             "cluster names.")
    parser.add_argument(
        "--by-country", action="store_true",
        help="Group by individual country instead of IW cluster, so no taxonomy is "
             "imposed. The overlap matrix then has one row per country and the "
             "grouping can be read off it rather than assumed — which is the only "
             "way to ask whether the model's neuron sharing recovers the "
             "Inglehart-Welzel map or some other structure (language, script, "
             "training-data volume). With the z filter this is exactly upstream's "
             "decide_culture_specific_neuron.py. Output files are named "
             "country_* instead of cluster_*.",
    )
    parser.add_argument("--min-countries", type=int, default=2,
                        help="Skip a cluster represented by fewer than this many "
                             "countries in the data. A 1-country cluster is a "
                             "country selection wearing a cluster label, and its "
                             "z-score is driven by that one country. Default 2.")
    parser.add_argument("--zscore-threshold", type=float, default=ZSCORE_THRESHOLD)
    parser.add_argument(
        "--skip-membership", action="store_true",
        help="Do not write cluster_membership_*.json. That file is the pre-z-score "
             "view — which clusters each neuron is a culture neuron for, plus "
             "Jaccard, Pearson and Spearman overlap between clusters. It costs "
             "nothing extra since the delta matrix is already in memory.",
    )
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
    parser.add_argument(
        "--mlp-proportion", type=float, default=MLP_CULTURE_NEURON_PROPORTION,
        help=f"Top fraction of the mlp.gate_proj pool per cluster (default "
             f"{MLP_CULTURE_NEURON_PROPORTION}, matching the culture-general "
             f"selection). Jaccard in the membership file shifts with this; the "
             f"Pearson/Spearman matrices are threshold-free and do not.",
    )
    parser.add_argument(
        "--attn-proportion", type=float, default=ATTENTION_CULTURE_NEURON_PROPORTION,
        help=f"Top fraction of the attention pool per cluster (default "
             f"{ATTENTION_CULTURE_NEURON_PROPORTION}). Separate from MLP because "
             f"attention attributions are not on the same scale, which is why the "
             f"general selection quotas them apart rather than ranking them together.",
    )
    parser.add_argument(
        "--countryrc-proportion", type=float, default=MLP_COUNTRYRC_NEURON_PROPORTION,
        help=f"Top fraction of EACH pool excluded as countryrc (default "
             f"{MLP_COUNTRYRC_NEURON_PROPORTION}, applied to both, as the general "
             f"selection does). Measured intersection with culture candidates is "
             f"2.5-13.4%% on 8b — small, but 2.5-13x above chance, so it is doing "
             f"real work. Watch the per-cluster drop counts in the log.",
    )
    args = parser.parse_args()

    import numpy as np
    logger = setup_logging()
    dataset_names = sorted(args.dataset_names)
    size_sfx = "" if args.model_size == "3b" else f"_{args.model_size}"
    cond_dir = NEURONS_ROOT / f"{args.condition}{size_sfx}"
    # Grouping is decided once here; everything downstream works off `groups`,
    # `gi` and `group_of`, so the two modes share one code path.
    first = (f"{dataset_names[0]}_yn"
             if (args.yn_only and dataset_names[0] == "normad") else dataset_names[0])
    seen_countries = _peek_countries(cond_dir / f"{first}_max_scores.json")

    if args.by_country:
        groups = sorted(seen_countries)
        gi = {g: i for i, g in enumerate(groups)}
        group_of = lambda c: c if c in gi else None
        members_of = lambda g: [g]
        if args.min_countries > 1:
            logger.info("--by-country: forcing --min-countries 1 "
                        "(each group is one country by definition)")
            args.min_countries = 1
    else:
        groups = list(CLUSTERS)
        gi = cluster_index()
        group_of = cluster_of
        _bc, _ = report_coverage(seen_countries)
        members_of = lambda g: _bc.get(g, [])

    ng = len(groups)
    if args.clusters is None:
        args.clusters = list(groups)
    unknown = [g for g in args.clusters if g not in gi]
    if unknown:
        raise SystemExit(f"--clusters: not in this grouping: {unknown}")

    label = "Country" if args.by_country else "Cluster"
    logger.info(f"{label} grouping: {ng} groups")
    logger.info(f"{label}-specific selection: condition={args.condition} "
                f"size={args.model_size} datasets={dataset_names}")
    logger.info(f"mlp={args.mlp_proportion} attn={args.attn_proportion} "
                f"countryrc={args.countryrc_proportion} "
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

        # group_index is keyed by the country names as the file spells them;
        # anything group_of cannot place is skipped, and reported rather than
        # silently shrinking a group's mean.
        file_countries = _peek_countries(score_path)
        group_index = {}
        unresolved = []
        for c in file_countries:
            g = group_of(c)
            if g is None:
                unresolved.append(c)
            else:
                group_index[c] = gi[g]
        if unresolved:
            logger.warning(f"  {score_name}: unresolved countries dropped: {unresolved}")

        by_cluster = {g: [] for g in groups}
        for c in file_countries:
            g = group_of(c)
            if g is not None:
                by_cluster[g].append(c)
        for g, cs in by_cluster.items():
            n_countries[gi[g]] = max(n_countries[gi[g]], len(cs))

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
    crc_countries = _peek_countries(crc_path)
    crc_group = {}
    crc_by_cluster = {g: [] for g in groups}
    crc_unresolved = []
    for c in crc_countries:
        g = group_of(c)
        if g is None:
            crc_unresolved.append(c)
        else:
            crc_group[c] = gi[g]
            crc_by_cluster[g].append(c)
    if crc_unresolved:
        logger.info(f"  countryrc: {len(crc_unresolved)} countries not in this "
                    f"grouping, ignored for the exclusion")
    missing = [g for g in groups if not crc_by_cluster[g]]
    if missing:
        logger.warning(f"  countryrc has no countries for {len(missing)} group(s): "
                       f"{missing[:8]}{' ...' if len(missing) > 8 else ''} — their "
                       f"surface-form exclusion will be empty")
    logger.info("Reading countryrc")
    k_rc, m_rc, _ = _load_matrix(crc_path, crc_group, ng, logger)
    for dname, ids in dataset_ids_of(crc_path).items():
        dataset_ids[dname].extend(ids)
    crc_n = np.array([max(len(crc_by_cluster[c]), 1) for c in groups], dtype=float)
    crc = _align(m_rc, k_rc, keys, kidx, np) / crc_n

    # Restrict to scored modules once; upstream ranks them in a single pool.
    # Two pools, quotaed separately, exactly as the culture-general selection does.
    mods = [_module_of(k) for k in keys]
    mlp_pool = np.flatnonzero(np.array([m in MLP_TARGET_MODULES for m in mods]))
    attn_pool = np.flatnonzero(np.array([m in ATTENTION_TARGET_MODULES for m in mods]))
    idx_pool = np.concatenate([mlp_pool, attn_pool])
    logger.info(f"Pools: MLP {len(mlp_pool)}, attention {len(attn_pool)} "
                f"(of {len(keys)} scored; mlp.up_proj is not a target module)")

    # z-score across clusters, per neuron — over POPULATED clusters only.
    # A cluster with no countries in this benchmark has delta 0 by construction,
    # and letting those zeros into the statistics drags the mean toward 0 and
    # inflates sd, so every active neuron clears the threshold for every cluster
    # it does have data for. BLEnD has no ProtestantEurope countries at all.
    populated = n_countries > 0
    logger.info(f"Populated clusters ({int(populated.sum())}/{ng}): "
                f"{[c for c in groups if populated[gi[c]]][:12]}")
    sub = delta[:, populated]
    mu = sub.mean(axis=1, keepdims=True)
    sd = sub.std(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (delta - mu) / sd, 0.0)

    out_suffix = "".join(dataset_names) + ("_yn" if args.yn_only else "")
    n_mlp = int(len(mlp_pool) * args.mlp_proportion)
    n_attn = int(len(attn_pool) * args.attn_proportion)
    n_crc_mlp = int(len(mlp_pool) * args.countryrc_proportion)
    n_crc_attn = int(len(attn_pool) * args.countryrc_proportion)
    logger.info(f"Per cluster: top {n_mlp} MLP + {n_attn} attention = "
                f"{n_mlp + n_attn} candidates; excluding top {n_crc_mlp} MLP + "
                f"{n_crc_attn} attention countryrc")

    flat_mask = sd[:, 0] < args.min_spread * np.abs(mu[:, 0]) if args.min_spread > 0 \
        else np.zeros(len(keys), dtype=bool)
    if args.min_spread > 0:
        logger.info(f"min-spread {args.min_spread}: {int(flat_mask.sum())} neurons "
                    f"too flat across clusters to be called specific")

    # ---- Stage A: candidates per cluster, BEFORE the z-score filter.
    #
    # These are the "culture neurons for cluster C" sets: top-k% by cluster delta,
    # minus that cluster's countryrc neurons. A neuron may belong to several, and
    # that is the point — membership across clusters is what the overlap metrics
    # measure. The z-score filter (stage B) is what narrows these to neurons
    # specific to ONE cluster, so it must not run yet.
    eligible = [cl for cl in args.clusters if n_countries[gi[cl]] >= args.min_countries]
    for cl in args.clusters:
        if cl not in eligible:
            logger.warning(f"{cl}: only {int(n_countries[gi[cl]])} countries present "
                           f"(< --min-countries {args.min_countries}) — skipped")

    candidates: dict[str, list[int]] = {}
    for cl in eligible:
        g = gi[cl]
        cand, excluded = [], set()
        for pool, n_c, n_x in ((mlp_pool, n_mlp, n_crc_mlp),
                               (attn_pool, n_attn, n_crc_attn)):
            if len(pool) == 0:
                continue
            order = pool[np.argsort(-delta[pool, g], kind="stable")]
            crc_order = pool[np.argsort(-crc[pool, g], kind="stable")]
            cand.extend(int(i) for i in order[:n_c])
            excluded |= set(int(i) for i in crc_order[:n_x])
        kept = [i for i in cand if i not in excluded]
        candidates[cl] = kept
        logger.info(f"{cl}: {len(kept)} culture neurons "
                    f"({len(cand) - len(kept)} dropped as countryrc, "
                    f"{100*(len(cand)-len(kept))/max(len(cand),1):.1f}%)")

    # ---- Stage B: the specific selections — candidates that survive the z filter.
    for cl in eligible:
        g = gi[cl]
        refined = []
        module_count: dict[str, int] = defaultdict(int)
        n_z_dropped = 0
        for i in candidates[cl]:
            if z[i, g] < args.zscore_threshold or flat_mask[i]:
                n_z_dropped += 1
                continue
            parts = keys[i].split("_")
            refined.append({
                "module_name": "_".join(parts[:-2]),
                "layer_idx": int(parts[-2]),
                "neuron_idx": int(parts[-1]),
                "attribute_score": float(delta[i, g]),
                "zscore": float(z[i, g]),
                "scores": {c: float(delta[i, gi[c]]) for c in eligible},
            })
            module_count["_".join(parts[:-2])] += 1

        out_path = cond_dir / f"{_slug(cl)}_neurons_{out_suffix}_max.json"
        out_path.write_text(json.dumps({
            "condition": args.condition,
            "grouping": "country" if args.by_country else "cluster",
            "cluster": cl,
            "countries": sorted(by_cluster.get(cl, [])),
            "dataset_ids": {k: v for k, v in dataset_ids.items()},
            "top_neurons": refined,
        }, indent=2))
        logger.info(f"{cl}: {len(refined)} SPECIFIC neurons "
                    f"({n_z_dropped} dropped z<{args.zscore_threshold}) "
                    f"| per module: {dict(module_count)} -> {out_path.name}")

    # ---- Stage C: membership and overlap, from the stage-A sets.
    if args.skip_membership:
        return
    _write_membership(cond_dir, out_suffix, args, eligible, candidates, keys,
                      delta, z, crc, idx_pool, gi, by_cluster, logger, np)


def _write_membership(cond_dir, out_suffix, args, eligible, candidates, keys,
                      delta, z, crc, idx_pool, gi, by_cluster, logger, np):
    """Write per-neuron cluster membership plus cluster-by-cluster overlap.

    Answers "which clusters is this neuron a culture neuron for" directly, rather
    than leaving it to be reconstructed by intersecting the eight specific files —
    which would not work, since those are post-z and therefore near-disjoint by
    construction.

    Two overlap measures, because they fail differently. Jaccard over the top-k%
    sets is interpretable but moves with --proportion. Correlation over the full
    delta columns uses every neuron in the pool and no threshold at all, so it
    cannot be shifted by the quota. Disagreement between them means the overlap is
    concentrated in the tail rather than being a property of the whole population.
    """
    from collections import Counter

    member: dict[int, list[str]] = defaultdict(list)
    for cl in eligible:
        for i in candidates[cl]:
            member[i].append(cl)

    counts = Counter(len(v) for v in member.values())
    logger.info("Membership histogram (neurons that are culture neurons for N clusters):")
    for n in sorted(counts):
        logger.info(f"   {n} cluster(s): {counts[n]} neurons")

    # Jaccard and raw intersection over the top-k% sets
    sets = {cl: set(candidates[cl]) for cl in eligible}
    jac = {a: {} for a in eligible}
    inter = {a: {} for a in eligible}
    for a in eligible:
        for b in eligible:
            ia = len(sets[a] & sets[b])
            un = len(sets[a] | sets[b])
            inter[a][b] = ia
            jac[a][b] = ia / un if un else 0.0

    # Threshold-free similarity between clusters, four ways, because attribution
    # scores are heavy-tailed and raw Pearson is dominated by a handful of extreme
    # neurons: two clusters sharing two outliers can read as r=0.98 while their
    # rank structure is unrelated.
    #
    #   spearman   - ranks, so every neuron has equal weight. The headline measure.
    #   pearson    - raw values. Keep it as a diagnostic, not a conclusion.
    #   winsorized - values clipped to the 1st/99th percentile per cluster, so
    #                outliers are capped rather than dropped. Nothing is discarded.
    #   drop_top1  - Pearson after removing the 1% of neurons with the largest
    #                |delta| in any cluster. If pearson is high and this is not,
    #                the correlation was carried by those neurons.
    cols = np.array([gi[cl] for cl in eligible])
    sub = delta[np.ix_(idx_pool, cols)]

    pear = np.corrcoef(sub, rowvar=False)
    ranks = np.argsort(np.argsort(sub, axis=0), axis=0).astype(np.float64)
    spear = np.corrcoef(ranks, rowvar=False)

    lo = np.percentile(sub, 1, axis=0)
    hi = np.percentile(sub, 99, axis=0)
    wins = np.corrcoef(np.clip(sub, lo, hi), rowvar=False)

    peak = np.abs(sub).max(axis=1)
    cut = np.percentile(peak, 99)
    kept_rows = peak <= cut
    drop1 = (np.corrcoef(sub[kept_rows], rowvar=False)
             if kept_rows.sum() > 2 else np.full_like(pear, np.nan))
    logger.info(f"Leverage check: dropped {int((~kept_rows).sum())} of {len(sub)} "
                f"pool neurons (top 1% by |delta|) for drop_top1")

    # Per-layer: where do exclusive vs shared neurons sit?
    layers: dict[int, dict[str, int]] = defaultdict(lambda: {"exclusive": 0, "shared": 0})
    for i, cls in member.items():
        lay = int(keys[i].split("_")[-2])
        layers[lay]["exclusive" if len(cls) == 1 else "shared"] += 1

    neurons = []
    for i, cls in sorted(member.items(), key=lambda kv: -len(kv[1])):
        parts = keys[i].split("_")
        neurons.append({
            "module_name": "_".join(parts[:-2]),
            "layer_idx": int(parts[-2]),
            "neuron_idx": int(parts[-1]),
            "n_clusters": len(cls),
            "clusters": cls,
            "scores": {cl: float(delta[i, gi[cl]]) for cl in eligible},
            "zscores": {cl: float(z[i, gi[cl]]) for cl in eligible},
        })

    kind = "country" if args.by_country else "cluster"
    out = cond_dir / f"{kind}_membership_{out_suffix}_max.json"
    out.write_text(json.dumps({
        "condition": args.condition,
        "grouping": kind,
        "clusters": eligible,
        "countries": {cl: sorted(by_cluster.get(cl, [])) for cl in eligible},
        "params": {
            "countryrc_proportion": args.countryrc_proportion,
            "pool_size": int(len(idx_pool)),
            "mlp_proportion": args.mlp_proportion,
            "attn_proportion": args.attn_proportion,
            "note": "membership is pre-z-score; the per-group *_neurons_*.json "
                    "files are post-z and so are near-disjoint by construction",
        },
        "n_selected": {cl: len(candidates[cl]) for cl in eligible},
        "membership_histogram": {str(n): counts[n] for n in sorted(counts)},
        "jaccard": jac,
        "intersection": inter,
        "pearson": {a: {b: float(pear[i, j]) for j, b in enumerate(eligible)}
                    for i, a in enumerate(eligible)},
        "spearman": {a: {b: float(spear[i, j]) for j, b in enumerate(eligible)}
                     for i, a in enumerate(eligible)},
        "pearson_winsorized": {a: {b: float(wins[i, j]) for j, b in enumerate(eligible)}
                               for i, a in enumerate(eligible)},
        "pearson_drop_top1": {a: {b: float(drop1[i, j]) for j, b in enumerate(eligible)}
                              for i, a in enumerate(eligible)},
        "per_layer": {str(k): v for k, v in sorted(layers.items())},
        "neurons": neurons,
    }, indent=2))

    uniq = len(member)
    total = sum(len(v) for v in member.values())
    logger.info(f"Union {uniq} neurons across {total} cluster slots "
                f"(mean {total/max(uniq,1):.2f} clusters per neuron) -> {out.name}")
    offdiag = [jac[a][b] for a in eligible for b in eligible if a != b]
    if offdiag:
        logger.info(f"Pairwise Jaccard: min={min(offdiag):.3f} "
                    f"mean={sum(offdiag)/len(offdiag):.3f} max={max(offdiag):.3f}")


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
    from culnig.score_io import _open_text, resolve_scores_path, unescape_key
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
                return [unescape_key(c)
                        for c in re.findall(r'"((?:[^"\\]|\\.)*)"\s*:', m.group(1))]
    raise ValueError(f"{path}: could not read per-country keys")


if __name__ == "__main__":
    main()
