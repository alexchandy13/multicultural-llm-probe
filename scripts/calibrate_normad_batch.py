"""Batch calibration for NormAd prediction JSONs (Zhou et al. 2023).

For each label, subtract the mean score across all examples before argmax:
  cal_score_i[c] = score_i[c] - mean_j(score_j[c])

Applied post-hoc to existing JSONs that contain a "scores" field in predictions.
Writes a _batch_calibrated.json alongside the input file.

Works for both yn and mpw modes (detected from filename).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def batch_calibrate(input_path: Path) -> None:
    d = json.loads(input_path.read_text())
    preds = d["predictions"]

    missing = [i for i, p in enumerate(preds) if p.get("scores") is None]
    if missing:
        raise ValueError(f"{len(missing)} predictions missing 'scores'. Re-run eval with updated script.")

    n = len(preds)
    n_labels = len(preds[0]["scores"])

    # Compute per-label mean across all examples
    means = [sum(p["scores"][k] for p in preds) / n for k in range(n_labels)]
    print(f"Label means: {[f'{m:.4f}' for m in means]}")

    labels = ["yes", "no"] if n_labels == 2 else ["A", "B", "C", "D"]

    n_flipped = 0
    new_preds = []
    for p in preds:
        cal = [p["scores"][k] - means[k] for k in range(n_labels)]
        new_pred = labels[max(range(n_labels), key=cal.__getitem__)]
        if new_pred != p["pred"]:
            n_flipped += 1
        entry = {**p, "pred": new_pred, "scores_batch_calibrated": cal}
        if p.get("us_scores") is not None:
            us_cal = [p["us_scores"][k] - means[k] for k in range(n_labels)]
            entry["us_pred"] = labels[max(range(n_labels), key=us_cal.__getitem__)]
            entry["us_scores_batch_calibrated"] = us_cal
        new_preds.append(entry)

    print(f"Flipped {n_flipped}/{n} predictions.")

    total = sum(1 for p in new_preds if p["gold"] in labels)
    correct = sum(1 for p in new_preds if p["gold"] in labels and p["pred"] == p["gold"])
    acc = correct / total if total else None
    print(f"Batch-calibrated accuracy: {acc:.3f}")

    out = {**d, "predictions": new_preds, "accuracy_overall_batch_calibrated": acc}
    out_path = input_path.with_name(input_path.stem + "_batch_calibrated.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Prediction JSONs with scores field")
    args = parser.parse_args()
    for p in args.inputs:
        print(f"\n=== {p} ===")
        batch_calibrate(Path(p))


if __name__ == "__main__":
    main()
