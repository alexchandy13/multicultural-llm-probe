"""Extend an existing countryrc_max_scores.json with new countries.

Loads the model for a given condition, scores only the countries NOT already
in the existing file, and merges the results back. Safe to re-run — already-
scored countries are always skipped.

Usage (on cluster, after `source env.sh && conda activate llm`):
  python scripts/extend_countryrc.py \
      --condition base --model-size 8b \
      --new-countries Germany Japan India Russia Brazil Zimbabwe

To add ALL normad/culturalbench countries not already covered, omit
--new-countries and pass --all-extra instead.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT_ROOT / "culnig" / "_upstream"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(PROJECT_ROOT))

import culnig.dataset_ext  # noqa: F401 — installs normadcontrol patch
from CULNIG import calc_neuron_score as upstream_score
from culnig.calc_neuron_score import (
    BATCH_SIZE,
    calculate_scores_memory_efficient,
    load_model_for_culnig,
)

NEURONS_ROOT = PROJECT_ROOT / "outputs" / "neurons"

# All countries appearing in normad, culturalbench, or blend that are NOT in
# the original 8 TARGET_COUNTRIES. Add more here as needed.
ALL_EXTRA_COUNTRIES = [
    # normad / culturalbench only (same 14-country set minus the original 8)
    "Germany", "Japan", "India", "Russia", "Brazil", "Zimbabwe",
    # blend-only
    "Algeria", "Assam", "Greece", "Ethiopia", "Nigeria", "North Korea",
    "West Java", "Azerbaijan",
]


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                        choices=["base", "dpo", "sft", "sftdpo",
                                 "sft_aya_cult", "sft_aya_nocult",
                                 "sftdpo_aya_cult", "sftdpo_aya_nocult"])
    parser.add_argument("--model-size", default="8b",
                        choices=["3b", "8b", "gemma4", "qwen35"])
    parser.add_argument("--precision", default="matched_bf16",
                        choices=["matched_bf16", "qlora_4bit"])
    parser.add_argument("--new-countries", nargs="+", default=None,
                        help="Country display names to add, e.g. Germany Japan India. "
                             "Skips any already in the existing file.")
    parser.add_argument("--all-extra", action="store_true",
                        help="Add all extra countries (ALL_EXTRA_COUNTRIES) "
                             "not already scored.")
    args = parser.parse_args()

    logger = setup_logging()

    size_sfx = "" if args.model_size == "3b" else f"_{args.model_size}"
    cond_dir = NEURONS_ROOT / f"{args.condition}{size_sfx}"
    crc_path = cond_dir / "countryrc_max_scores.json"

    if not crc_path.exists():
        logger.error(f"No existing countryrc file at {crc_path}. "
                     "Run calc_neuron_score.py first.")
        sys.exit(1)

    existing = json.loads(crc_path.read_text())
    existing_countries = set(existing["total_probabilities_per_country"].keys())
    logger.info(f"Existing countries ({len(existing_countries)}): "
                f"{sorted(existing_countries)}")

    if args.all_extra:
        requested = ALL_EXTRA_COUNTRIES
    elif args.new_countries:
        requested = args.new_countries
    else:
        parser.error("Provide --new-countries or --all-extra")

    new_countries = [c for c in requested if c not in existing_countries]
    if not new_countries:
        logger.info("All requested countries already scored — nothing to do.")
        return
    logger.info(f"Scoring {len(new_countries)} new countries: {new_countries}")

    model, tokenizer = load_model_for_culnig(
        args.condition, model_size=args.model_size, precision=args.precision
    )
    logger.info(f"Model loaded on {model.device}")

    dataloader = upstream_score.load_dataset_neuron_scores(
        dataset_names=["countryrc"],
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        target_countries=new_countries,
        target_data="neuron",
    )

    raw_scores, new_probs = calculate_scores_memory_efficient(
        model, tokenizer, dataloader, logger
    )

    # Flatten (module_name, layer_idx, neuron_idx) keys → string keys
    new_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (module_name, layer_idx, neuron_idx), per_country in raw_scores.items():
        key = f"{module_name}_{layer_idx}_{neuron_idx}"
        for country, score in per_country.items():
            new_scores[key][country] += score

    # Collect new dataset IDs
    new_ids: list[int] = []
    for item in dataloader.dataset:
        if item["id"] not in new_ids:
            new_ids.append(item["id"])

    # --- Merge into existing ---
    merged_neuron_scores = existing["neuron_scores"]
    for key, country_scores in new_scores.items():
        if key not in merged_neuron_scores:
            merged_neuron_scores[key] = {}
        for country, score in country_scores.items():
            merged_neuron_scores[key][country] = score

    merged_probs = existing["total_probabilities_per_country"]
    for country, prob in new_probs.items():
        merged_probs[country] = prob

    merged_ids = existing["dataset_ids"]
    old_crc_ids = merged_ids.get("countryrc", [])
    merged_ids["countryrc"] = sorted(set(old_crc_ids) | set(new_ids))

    crc_path.write_text(json.dumps({
        "neuron_scores": merged_neuron_scores,
        "total_probabilities_per_country": merged_probs,
        "dataset_ids": merged_ids,
    }, indent=2))
    logger.info(f"Updated {crc_path} — now covers "
                f"{len(merged_probs)} countries: {sorted(merged_probs.keys())}")


if __name__ == "__main__":
    main()
