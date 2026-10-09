"""Fork of upstream `CULNIG/decide_culture_general_neurons.py`.

Only difference from upstream: reads from our `outputs/neurons/{condition}/` tree
and writes results there too, instead of upstream's `../outputs/{model_name}/`.
The selection logic (top-t% on (NormAd - NormAdctrl), subtract CountryRC top-r%)
is byte-for-byte upstream.

Usage:
    python culnig/decide_culture_neurons.py --condition sft --dataset-names normad
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEURONS_ROOT = PROJECT_ROOT / "outputs" / "neurons"
sys.path.insert(0, str(PROJECT_ROOT))   # run as a path, not a module

from culnig.score_io import dataset_ids as dataset_ids_of  # noqa: E402
from culnig.score_io import neuron_sums  # noqa: E402

# Upstream hyperparameters — copied verbatim.
MLP_CULTURE_NEURON_PROPORTION = 0.05
MLP_COUNTRYRC_NEURON_PROPORTION = 0.01
ATTENTION_CULTURE_NEURON_PROPORTION = 0.01
ATTENTION_COUNTRYRC_NEURON_PROPORTION = 0.01
MLP_TARGET_MODULES = ["mlp.gate_proj"]
ATTENTION_TARGET_MODULES = ["self_attn.v_proj", "self_attn.q_proj", "self_attn.k_proj"]
MLP_SAVE_MODULES = ["mlp.gate_proj"]
ATTENTION_SAVE_MODULES = ["self_attn.v_proj", "self_attn.q_proj", "self_attn.k_proj"]


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                        choices=["base", "dpo", "sft", "sftdpo",
                                 "sft_aya_cult", "sft_aya_nocult",
                                 "sftdpo_aya_cult", "sftdpo_aya_nocult",
                                 "tulu3_sft", "tulu3_dpo"])
    parser.add_argument("--dataset-names", nargs="+", default=["normad"])
    parser.add_argument(
        "--model-size", default="3b", choices=["3b", "8b", "gemma4", "qwen35"],
        help="Base model size. Reads scores from outputs/neurons/{condition}"
             "{_size_suffix}/ and writes selected neurons there. Must match the "
             "size used during calc_neuron_score.py.",
    )
    parser.add_argument("--yn-only", action="store_true",
                        help="Read normad_yn_max_scores.json instead of normad_max_scores.json. "
                             "Control file remains normadcontrol_max_scores.json. "
                             "Output gains a _yn suffix.")
    args = parser.parse_args()

    logger = setup_logging()
    dataset_names = sorted(args.dataset_names)
    size_sfx = "" if args.model_size == "3b" else f"_{args.model_size}"
    cond_dir = NEURONS_ROOT / f"{args.condition}{size_sfx}"
    logger.info(f"Deciding culture neurons for condition={args.condition} "
                f"size={args.model_size} datasets={dataset_names}")

    mlp_scores = defaultdict(float)
    attn_scores = defaultdict(float)
    dataset_ids = defaultdict(list)

    for dataset_name in dataset_names:
        score_name = f"{dataset_name}_yn" if (args.yn_only and dataset_name == "normad") else dataset_name
        score_path = cond_dir / f"{score_name}_max_scores.json"
        ctrl_path = cond_dir / f"{dataset_name}control_max_scores.json"
        # Streamed: only the per-neuron totals are needed, and holding the full
        # nested per-country structure peaks near 34 GB on a gemma4 normad pass.
        score_sums = neuron_sums(score_path)     # .json or .json.gz
        ctrl_sums = neuron_sums(ctrl_path)
        score_ids = dataset_ids_of(score_path)
        ctrl_ids = dataset_ids_of(ctrl_path)

        for dname, ids in score_ids.items():
            dataset_ids[dname].extend(ids)
        n_samples = len(score_ids[score_name])
        n_ctrl = len(ctrl_ids[f"{dataset_name}control"])

        for key, total in score_sums.items():
            parts = key.split("_")
            module_name = "_".join(parts[:-2])
            ds_mean = total / n_samples
            ctrl_mean = ctrl_sums.get(key, 0.0) / n_ctrl
            delta = ds_mean - ctrl_mean
            if module_name in MLP_TARGET_MODULES:
                mlp_scores[key] += delta
            elif module_name in ATTENTION_TARGET_MODULES:
                attn_scores[key] += delta

    # Top-t% per module family
    mlp_sorted = sorted(mlp_scores.items(), key=lambda x: x[1], reverse=True)
    attn_sorted = sorted(attn_scores.items(), key=lambda x: x[1], reverse=True)
    mlp_top = mlp_sorted[: int(len(mlp_sorted) * MLP_CULTURE_NEURON_PROPORTION)]
    attn_top = attn_sorted[: int(len(attn_sorted) * ATTENTION_CULTURE_NEURON_PROPORTION)]
    logger.info(f"MLP culture candidates: {len(mlp_top)}; Attn: {len(attn_top)}")
    culture_candidates = mlp_top + attn_top

    # CountryRC — exclude top-r% as language/country surface-form neurons
    crc_path = cond_dir / "countryrc_max_scores.json"
    crc_sums = neuron_sums(crc_path)   # accepts countryrc_max_scores.json.gz too
    for dname, ids in dataset_ids_of(crc_path).items():
        dataset_ids[dname].extend(ids)

    mlp_crc = defaultdict(float)
    attn_crc = defaultdict(float)
    for key, total in crc_sums.items():
        parts = key.split("_")
        module_name = "_".join(parts[:-2])
        if module_name in MLP_TARGET_MODULES:
            mlp_crc[key] = total
        elif module_name in ATTENTION_TARGET_MODULES:
            attn_crc[key] = total

    mlp_crc_sorted = sorted(mlp_crc.items(), key=lambda x: x[1], reverse=True)
    attn_crc_sorted = sorted(attn_crc.items(), key=lambda x: x[1], reverse=True)
    mlp_crc_top = {k for k, _ in mlp_crc_sorted[: int(len(mlp_crc_sorted) * MLP_COUNTRYRC_NEURON_PROPORTION)]}
    attn_crc_top = {k for k, _ in attn_crc_sorted[: int(len(attn_crc_sorted) * ATTENTION_COUNTRYRC_NEURON_PROPORTION)]}
    crc_excluded = mlp_crc_top | attn_crc_top

    refined = []
    module_count = defaultdict(int)
    for neuron, score in culture_candidates:
        if neuron in crc_excluded:
            continue
        parts = neuron.split("_")
        neuron_idx = int(parts[-1])
        layer_idx = int(parts[-2])
        module_name = "_".join(parts[:-2])
        if module_name not in MLP_SAVE_MODULES and module_name not in ATTENTION_SAVE_MODULES:
            continue
        refined.append({
            "module_name": module_name,
            "layer_idx": layer_idx,
            "neuron_idx": neuron_idx,
            "attribute_score": score,
        })
        module_count[module_name] += 1

    logger.info(f"Refined culture neurons: {len(refined)} | per module: {dict(module_count)}")

    out_suffix = "".join(dataset_names) + ("_yn" if args.yn_only else "")
    out_path = cond_dir / f"all_neurons_{out_suffix}_max.json"
    out_path.write_text(json.dumps({
        "condition": args.condition,
        "dataset_ids": dict(dataset_ids),
        "top_neurons": refined,
    }, indent=2))
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
