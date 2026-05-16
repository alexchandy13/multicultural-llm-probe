"""Fork of upstream `CULNIG/decide_culture_general_neurons.py`.

Only difference from upstream: reads from our `outputs/neurons/{condition}/` tree
and writes results there too, instead of upstream's `../outputs/{model_name}/`.
The selection logic (top-t% on (NormAd - NormAdctrl), subtract CountryRC top-r%)
is byte-for-byte upstream.

Usage:
    python culnig/decide_culture_neurons_normad.py --condition sft --dataset-names normad
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEURONS_ROOT = PROJECT_ROOT / "outputs" / "neurons"

# Upstream hyperparameters — copied verbatim.
MLP_CULTURE_NEURON_PROPORTION = 0.01
MLP_COUNTRYRC_NEURON_PROPORTION = 0.01
ATTENTION_CULTURE_NEURON_PROPORTION = 0.002
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
                        choices=["base", "sft", "dpo", "sftdpo", "instruct",
                                 "sft_alpaca", "sftdpo_alpaca"])
    parser.add_argument("--dataset-names", nargs="+", default=["normad"])
    args = parser.parse_args()

    logger = setup_logging()
    dataset_names = sorted(args.dataset_names)
    cond_dir = NEURONS_ROOT / args.condition
    logger.info(f"Deciding culture neurons for condition={args.condition} datasets={dataset_names}")

    mlp_scores = defaultdict(float)
    attn_scores = defaultdict(float)
    dataset_ids = defaultdict(list)

    for dataset_name in dataset_names:
        score_path = cond_dir / f"{dataset_name}_max_scores.json"
        ctrl_path = cond_dir / f"{dataset_name}control_max_scores.json"
        if not score_path.exists():
            raise FileNotFoundError(score_path)
        if not ctrl_path.exists():
            raise FileNotFoundError(ctrl_path)
        scores_dict = json.loads(score_path.read_text())
        control_dict = json.loads(ctrl_path.read_text())

        for dname, ids in scores_dict["dataset_ids"].items():
            dataset_ids[dname].extend(ids)
        n_samples = len(scores_dict["dataset_ids"][dataset_name])
        n_ctrl = len(control_dict["dataset_ids"][f"{dataset_name}control"])

        for key, score in scores_dict["neuron_scores"].items():
            parts = key.split("_")
            module_name = "_".join(parts[:-2])
            ctrl_score = control_dict["neuron_scores"].get(key, {})
            ds_mean = sum(score.values()) / n_samples
            ctrl_mean = sum(ctrl_score.values()) / n_ctrl if ctrl_score else 0.0
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
    if not crc_path.exists():
        raise FileNotFoundError(crc_path)
    crc_dict = json.loads(crc_path.read_text())
    for dname, ids in crc_dict["dataset_ids"].items():
        dataset_ids[dname].extend(ids)

    mlp_crc = defaultdict(float)
    attn_crc = defaultdict(float)
    for key, score in crc_dict["neuron_scores"].items():
        parts = key.split("_")
        module_name = "_".join(parts[:-2])
        if module_name in MLP_TARGET_MODULES:
            mlp_crc[key] = sum(score.values())
        elif module_name in ATTENTION_TARGET_MODULES:
            attn_crc[key] = sum(score.values())

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

    out_path = cond_dir / f"all_neurons_{''.join(dataset_names)}_max.json"
    out_path.write_text(json.dumps({
        "condition": args.condition,
        "dataset_ids": dict(dataset_ids),
        "top_neurons": refined,
    }, indent=2))
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
