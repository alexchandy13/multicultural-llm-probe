"""Sum sharded .npz partials from calc_neuron_score.py --shard into the final JSON.

The accumulator in calc_neuron_score is a pure per-sample sum, so summing shard
partials reproduces an unsharded run exactly, up to float64 reassociation
(~1e-13 relative over ~20k terms — selection thresholds are top-1%/5%, so this
cannot move a ranking).

Usage:
  python scripts/merge_shards.py --condition base --model-size gemma4 --dataset blend
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEURONS_ROOT = PROJECT_ROOT / "outputs" / "neurons"


def _text(arr) -> str:
    return bytes(arr).decode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model-size", default="gemma4",
                        choices=["3b", "8b", "gemma4", "qwen35"])
    parser.add_argument("--dataset", required=True,
                        help="Dataset name as used in the shard filenames, "
                             "e.g. blend, blendcontrol, normad_yn.")
    parser.add_argument("--keep-shards", action="store_true",
                        help="Keep the _shards/*.npz partials after a successful merge.")
    args = parser.parse_args()

    size_sfx = "" if args.model_size == "3b" else f"_{args.model_size}"
    cond_dir = NEURONS_ROOT / f"{args.condition}{size_sfx}"
    shard_dir = cond_dir / "_shards"

    shards = sorted(shard_dir.glob(f"{args.dataset}_*of*.npz"))
    if not shards:
        sys.exit(f"No shards matching {args.dataset}_*of*.npz in {shard_dir}")

    expected_n = int(shards[0].stem.split("of")[-1])
    got = {int(p.stem.split("_")[-1].split("of")[0]) for p in shards}
    missing = set(range(expected_n)) - got
    if missing:
        sys.exit(f"Incomplete: missing shard(s) {sorted(missing)} of {expected_n}. "
                 "Merging now would silently undercount — rerun them first.")
    print(f"Merging {expected_n} shards for {args.condition}{size_sfx}/{args.dataset}")

    total = None
    keys: list[str] = []
    countries: list[str] = []
    probs: dict[str, float] = {}
    ids: dict[str, list] = {}

    for path in shards:
        with np.load(path) as z:
            s_keys = _text(z["keys"]).split("\n")
            s_countries = _text(z["countries"]).split("\n")
            s_scores = z["scores"]
            s_probs = z["probs"]
            s_ids = json.loads(_text(z["ids_json"]))

            if total is None:
                keys = s_keys
                countries = list(s_countries)
                total = np.zeros((len(keys), len(countries)), dtype=np.float64)
            elif s_keys != keys:
                sys.exit(f"{path.name}: key set differs from the first shard — "
                         "partials came from different models or code versions.")

            # Stride slicing gives every shard the same countries, but union
            # rather than assume, so a narrower shard can't drop a column.
            new = [c for c in s_countries if c not in countries]
            if new:
                countries.extend(new)
                total = np.hstack([
                    total, np.zeros((len(keys), len(new)), dtype=np.float64)
                ])
            cpos = {c: i for i, c in enumerate(countries)}
            total[:, [cpos[c] for c in s_countries]] += s_scores

            for c, p in zip(s_countries, s_probs.tolist()):
                probs[c] = probs.get(c, 0.0) + p
            # Union, first-seen order. decide_culture_neurons.py divides by
            # len(dataset_ids[name]), so a dropped id inflates every mean.
            for name, id_list in s_ids.items():
                seen = ids.setdefault(name, [])
                known = set(seen)
                for i in id_list:
                    if i not in known:
                        known.add(i)
                        seen.append(i)

        print(f"  + {path.name}")

    out_file = cond_dir / f"{args.dataset}_max_scores.json"
    tmp = out_file.with_suffix(".json.tmp")

    # Streamed per key: the assembled dict would be ~1.9M nested dicts of Python
    # floats, tens of GB resident. The array stays the only full copy.
    with open(tmp, "w") as fh:
        fh.write('{"neuron_scores":{')
        for i, key in enumerate(keys):
            if i:
                fh.write(",")
            row = dict(zip(countries, total[i].tolist()))
            fh.write(f"{json.dumps(key)}:{json.dumps(row)}")
        fh.write('},"total_probabilities_per_country":')
        fh.write(json.dumps(probs))
        fh.write(',"dataset_ids":')
        fh.write(json.dumps(ids))
        fh.write("}")
    tmp.replace(out_file)

    n_ids = {k: len(v) for k, v in ids.items()}
    print(f"Wrote {out_file} ({out_file.stat().st_size / 1e9:.2f} GB)")
    print(f"  {len(keys)} keys x {len(countries)} countries; unique ids: {n_ids}")

    if not args.keep_shards:
        for path in shards:
            path.unlink()
        print(f"Removed {len(shards)} partials from {shard_dir}")


if __name__ == "__main__":
    main()
