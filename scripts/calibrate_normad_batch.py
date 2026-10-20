"""Batch calibration for prediction JSONs (Zhou et al. 2023).

For each label, subtract the mean score across all examples before argmax:
  cal_score_i[c] = score_i[c] - mean_j(score_j[c])

Applied post-hoc to existing JSONs that contain a "scores" field in predictions.
Writes a _batch_calibrated.json alongside the input file.

Works for yes/no and any multiple-choice label count (inferred from the length
of the scores vector: 2 -> yes/no, otherwise A, B, C, ...).

--target-prior picks WHICH label distribution the calibration aims at:

  uniform  Zhou et al. as published: subtract each label's mean score, which
           drives the predictions toward equal label shares. Correct when the
           benchmark is balanced (BLEnD, CSQA) and wrong when it is not --
           CulturalBench gold is 27% yes / 73% no, so uniform calibration
           overshoots from 98% "no" straight past 73% to about 46%, costing
           plain accuracy that balanced accuracy then does not see.
  gold     match the file's own gold label distribution. For two labels this is
           exact (threshold the score margin at the right quantile); for more it
           is an iterative offset fit. Note this reads the test labels' marginal,
           so it is mild oracle access -- defensible when the prior is treated
           as a known property of the benchmark, but it must be declared, and it
           is applied identically to every checkpoint so comparisons stay fair.

For a prior-free comparison of two-label runs, the printed AUROC of the score
margin needs no threshold and no prior at all.

By default the US-swapped predictions are recalibrated with the SAME per-label
means as the originals, which is what the existing _batch_calibrated NormAd
files did. --us-own-means instead computes a second set of means from us_scores.
That matters because the all-US prompts are a different prompt distribution with
their own label skew: sharing the means leaves that skew uncorrected, while
separate means remove each prompt set's own bias. Shared means keep the two
prediction sets on one scale and so measure "same prompt bias, different
country"; separate means ask "would a calibrated model still agree with itself".
Neither is wrong, but a US-default rate computed under one is not comparable to
one computed under the other.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _auroc(margins: list[float], positive: list[bool]) -> float | None:
    """AUROC of a score margin against a binary gold, by rank-sum with ties.

    Threshold-free and prior-free, so it separates "can the model rank true
    statements above false ones" from "does it put the decision line in the
    right place" -- the two things a collapsed predictor confounds.
    """
    n_pos = sum(positive)
    n_neg = len(positive) - n_pos
    if not n_pos or not n_neg:
        return None
    order = sorted(range(len(margins)), key=margins.__getitem__)
    ranks = [0.0] * len(margins)
    i = 0
    while i < len(order):                      # average ranks within ties
        j = i
        while j + 1 < len(order) and margins[order[j + 1]] == margins[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rsum = sum(r for r, pos in zip(ranks, positive) if pos)
    return (rsum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _offsets_for_prior(scores: list[list[float]], target: list[float],
                       n_labels: int, iters: int = 200) -> list[float]:
    """Additive per-label offsets whose argmax hits `target` label shares.

    Two labels is solved exactly: the argmax rule reduces to thresholding the
    margin score[0]-score[1], so the offset is just the matching quantile.
    More labels have no closed form, so nudge each offset by
    log(target/observed) until the shares settle.
    """
    n = len(scores)
    if n_labels == 2:
        margins = sorted(sc[0] - sc[1] for sc in scores)
        k = min(max(int(round((1 - target[0]) * n)), 0), n - 1)
        return [-margins[k], 0.0]
    off = [0.0] * n_labels
    for _ in range(iters):
        counts = [0] * n_labels
        for sc in scores:
            counts[max(range(n_labels),
                       key=lambda c: sc[c] + off[c])] += 1
        share = [c / n for c in counts]
        if all(abs(share[c] - target[c]) < 0.002 for c in range(n_labels)):
            break
        for c in range(n_labels):
            obs = max(share[c], 1e-6)
            off[c] += 0.3 * math.log(max(target[c], 1e-6) / obs)
    return off


def batch_calibrate(input_path: Path, us_own_means: bool = False,
                    target_prior: str = "uniform") -> None:
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

    labels = (["yes", "no"] if n_labels == 2
              else [chr(ord("A") + k) for k in range(n_labels)])

    us_means = means
    scored_us = [p for p in preds if p.get("us_scores") is not None]
    if us_own_means and scored_us:
        us_means = [sum(p["us_scores"][k] for p in scored_us) / len(scored_us)
                    for k in range(n_labels)]
        print(f"US label means: {[f'{m:.4f}' for m in us_means]}")

    # Shift from the uniform target to the gold marginal, if asked. The offsets
    # are fitted on the mean-subtracted scores so the two modes compose.
    off = [0.0] * n_labels
    us_off = [0.0] * n_labels
    if target_prior == "gold":
        gold_n = [sum(1 for p in preds if p["gold"] == l) for l in labels]
        tot = sum(gold_n)
        target = [g / tot for g in gold_n]
        print(f"Gold prior: {[f'{t:.3f}' for t in target]}")
        centred = [[p["scores"][k] - means[k] for k in range(n_labels)]
                   for p in preds]
        off = _offsets_for_prior(centred, target, n_labels)
        with_us = [p for p in preds if p.get("us_scores") is not None]
        if with_us:
            us_centred = [[p["us_scores"][k] - us_means[k]
                           for k in range(n_labels)] for p in with_us]
            us_off = _offsets_for_prior(us_centred, target, n_labels)

    n_flipped = 0
    new_preds = []
    for p in preds:
        cal = [p["scores"][k] - means[k] + off[k] for k in range(n_labels)]
        new_pred = labels[max(range(n_labels), key=cal.__getitem__)]
        if new_pred != p["pred"]:
            n_flipped += 1
        entry = {**p, "pred": new_pred, "scores_batch_calibrated": cal}
        if p.get("us_scores") is not None:
            us_cal = [p["us_scores"][k] - us_means[k] + us_off[k]
                      for k in range(n_labels)]
            entry["us_pred"] = labels[max(range(n_labels), key=us_cal.__getitem__)]
            entry["us_scores_batch_calibrated"] = us_cal
        elif p.get("us_pred") is not None:
            # A raw us_pred next to a calibrated pred is not comparable, and the
            # US-default rate would silently read it as if it were. Without
            # us_scores it cannot be recalibrated, so drop it.
            entry["us_pred"] = None
            entry["us_pred_dropped_no_us_scores"] = True
        new_preds.append(entry)

    print(f"Flipped {n_flipped}/{n} predictions.")
    if any(p.get("us_pred_dropped_no_us_scores") for p in new_preds):
        print("NOTE: us_pred dropped -- the file has us_pred but no us_scores, "
              "so it cannot be recalibrated. No US-default rate from this file.")

    total = sum(1 for p in new_preds if p["gold"] in labels)
    correct = sum(1 for p in new_preds if p["gold"] in labels and p["pred"] == p["gold"])
    acc = correct / total if total else None
    # balanced accuracy too: plain accuracy hides a collapsed predictor, which
    # is the whole reason for calibrating these files
    per_label = []
    for lab in labels:
        sub = [p for p in new_preds if p["gold"] == lab]
        if sub:
            per_label.append(sum(p["pred"] == lab for p in sub) / len(sub))
    bal = sum(per_label) / len(per_label) if per_label else None
    modal = max(labels, key=lambda l: sum(p["pred"] == l for p in new_preds))
    modal_share = sum(p["pred"] == modal for p in new_preds) / n
    auroc = None
    if n_labels == 2:
        auroc = _auroc([p["scores"][0] - p["scores"][1] for p in preds],
                       [p["gold"] == labels[0] for p in preds])
    print(f"Batch-calibrated accuracy: {acc:.3f}  balanced: {bal:.3f}  "
          f"modal pred: {modal}={modal_share:.1%}"
          + (f"  AUROC: {auroc:.3f}" if auroc is not None else ""))

    out = {**d, "predictions": new_preds,
           "accuracy_overall_batch_calibrated": acc,
           "accuracy_balanced_batch_calibrated": bal,
           "auroc_margin": auroc,
           "batch_calibration_us_own_means": us_own_means,
           "batch_calibration_target_prior": target_prior}
    tag = "_batch_calibrated" if target_prior == "uniform" \
        else f"_batch_calibrated_{target_prior}prior"
    out_path = input_path.with_name(input_path.stem + tag + ".json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Prediction JSONs with scores field")
    parser.add_argument("--target-prior", default="uniform",
                        choices=["uniform", "gold"],
                        help="Label distribution the calibration aims at. "
                             "uniform is Zhou et al.; gold matches the file's "
                             "own gold marginal (writes a separate "
                             "_goldprior file).")
    parser.add_argument("--us-own-means", action="store_true",
                        help="Calibrate us_scores with their own per-label means "
                             "instead of reusing the originals'.")
    args = parser.parse_args()
    for p in args.inputs:
        print(f"\n=== {p} ===")
        batch_calibrate(Path(p), us_own_means=args.us_own_means,
                        target_prior=args.target_prior)


if __name__ == "__main__":
    main()
