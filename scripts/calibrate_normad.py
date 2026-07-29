"""Post-hoc calibration for NormAd predictions that were saved with raw scores.

Loads a prediction JSON produced by eval_normad.py (predictions must contain a "scores"
field — present in JSONs produced after the raw-scores change), computes unconditional
priors from the model once, subtracts them, re-argmaxes, and writes a new JSON with
a _calibrated suffix.

Supports two modes detected from the filename:
  *_nfs_mpw_usprobe.json   → multi-prompt-word: 4 null prompts, one per template
  *_nfs_yn_usprobe.json    → simple yn: 1 null prompt

Usage:
  python scripts/calibrate_normad.py \
      --input outputs/behavioral/normad_sft_aya_cult_8b_nfs_mpw_usprobe.json \
      --condition sft_aya_cult --model-size 8b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate._common import load_model_for_eval, resolve_condition
from evaluate.eval_normad import compute_priors, build_neutral_fewshot_prefix

CHOICES_YN = ["yes", "no"]


def _yn_suffix(path: Path) -> str | None:
    name = path.stem
    if "mpw" in name:
        return "mpw"
    if "_yn_" in name or name.endswith("_yn"):
        return "yn"
    return None


def calibrate(input_path: Path, condition: str, model_size: str) -> None:
    raw = json.loads(input_path.read_text())
    preds = raw["predictions"]

    missing = [i for i, p in enumerate(preds) if p.get("scores") is None]
    if missing:
        raise ValueError(
            f"{len(missing)} predictions have no 'scores' field. "
            "Re-run eval with the updated eval_normad.py that saves raw_scores."
        )

    mode = _yn_suffix(input_path)
    if mode is None:
        raise ValueError(
            "Cannot detect mode (mpw/yn) from filename. "
            "Expected '*_mpw_*' or '*_yn_*' in the path."
        )

    tokenizer, model = load_model_for_eval(resolve_condition(condition, model_size=model_size))

    print(f"Computing priors (mode={mode})...")
    if mode == "mpw":
        # build_neutral_fewshot_prefix returns list[str] for mpw (one per template)
        prefix = build_neutral_fewshot_prefix(multi_prompt_word=True)
        priors = compute_priors(
            model, tokenizer, CHOICES_YN,
            prefix=prefix, multi_prompt_word=True,
        )
        # priors is list of 4 [yes_prior, no_prior]; sum matches accumulation in eval
        prior_yes = sum(p[0] for p in priors)
        prior_no  = sum(p[1] for p in priors)
    else:
        prefix = build_neutral_fewshot_prefix()  # returns str for simple yn
        priors = compute_priors(
            model, tokenizer, CHOICES_YN,
            prefix=prefix, yn_only=True,
        )
        prior_yes, prior_no = priors[0], priors[1]

    print(f"  prior_yes={prior_yes:.4f}  prior_no={prior_no:.4f}")

    n_flipped = 0
    new_preds = []
    for p in preds:
        raw_yes, raw_no = p["scores"][0], p["scores"][1]
        cal_yes = raw_yes - prior_yes
        cal_no  = raw_no  - prior_no
        new_pred = "yes" if cal_yes > cal_no else "no"
        if new_pred != p["pred"]:
            n_flipped += 1
        new_preds.append({**p, "pred": new_pred, "scores_calibrated": [cal_yes, cal_no]})

    print(f"Flipped {n_flipped}/{len(preds)} predictions after calibration.")

    # Recompute accuracy
    total, correct = 0, 0
    for p in new_preds:
        if p["gold"] in ("yes", "no"):
            total += 1
            if p["pred"] == p["gold"]:
                correct += 1
    acc = correct / total if total else None
    print(f"Calibrated accuracy: {acc:.3f}")

    out = {**raw, "predictions": new_preds, "accuracy_overall_calibrated": acc}
    out_path = input_path.with_name(input_path.stem + "_calibrated.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to prediction JSON with scores field")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model-size", default="8b", choices=["3b", "8b", "gemma4", "qwen35"])
    args = parser.parse_args()
    calibrate(Path(args.input), args.condition, args.model_size)


if __name__ == "__main__":
    main()
