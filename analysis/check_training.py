"""Post-training health check for SFT / DPO / SFT+DPO runs.

For each named condition (default: all three trainable ones), reports:
  - Final checkpoint path + adapter file size
  - Loss trajectory (first 3, last 3, min, max)
  - DPO reward margins and accuracies (only if present in trainer_state.json)
  - Any error/traceback/NaN lines found in the corresponding SLURM stderr

Usage:
    python3.12 analysis/check_training.py                        # sft, dpo, sftdpo
    python3.12 analysis/check_training.py sft dpo                # subset
    python3.12 analysis/check_training.py --slurm-dir slurm      # alt log location
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CKPTS_DIR = PROJECT_ROOT / "checkpoints"
SLURM_DIR = PROJECT_ROOT / "slurm"
TRAIN_CONDITIONS = ["sft_alpaca", "dpo", "sftdpo_alpaca"]
ERROR_REGEX = re.compile(r"error|traceback|nan|cuda out of memory|reward margin <= 0", re.IGNORECASE)


def latest_checkpoint(parent: Path) -> Path | None:
    """Return the highest-numbered `checkpoint-N` subdir, or None if missing."""
    if not parent.exists():
        return None
    candidates = []
    for p in parent.glob("checkpoint-*"):
        suffix = p.name.rsplit("-", 1)[-1]
        if suffix.isdigit():
            candidates.append((int(suffix), p))
    if not candidates:
        return None
    return max(candidates)[1]


def fmt_list(xs: list[float], n: int = 3) -> str:
    return f"[{', '.join(f'{x:.3f}' for x in xs[:n])}]" if xs else "[]"


def adapter_size_mb(ckpt: Path) -> float | None:
    f = ckpt / "adapter_model.safetensors"
    return f.stat().st_size / (1024 * 1024) if f.exists() else None


def check_trajectory(ckpt: Path) -> dict:
    """Pull loss / margin / accuracy series from trainer_state.json."""
    state_path = ckpt / "trainer_state.json"
    if not state_path.exists():
        return {"missing_state": True}
    hist = json.loads(state_path.read_text()).get("log_history", [])
    losses = [h["loss"] for h in hist if "loss" in h]
    margins = [h["rewards/margins"] for h in hist if "rewards/margins" in h]
    accs = [h["rewards/accuracies"] for h in hist if "rewards/accuracies" in h]
    return {"losses": losses, "margins": margins, "accs": accs}


def find_stderr_logs(cond: str, slurm_dir: Path) -> list[Path]:
    """Match SLURM stderr files for this condition (e.g. slurm/sft.6835622.err)."""
    return sorted(slurm_dir.glob(f"{cond}.*.err"), key=lambda p: p.stat().st_mtime)


def scan_errors(log: Path, max_hits: int = 5) -> list[str]:
    """Return matching lines (or empty if clean)."""
    hits = []
    try:
        with log.open() as f:
            for line in f:
                if ERROR_REGEX.search(line):
                    hits.append(line.rstrip())
                    if len(hits) >= max_hits:
                        break
    except OSError:
        return []
    return hits


def verdict(cond: str, trajectory: dict) -> str:
    """One-line PASS/CHECK summary based on trajectory shape."""
    losses = trajectory.get("losses", [])
    margins = trajectory.get("margins", [])
    if not losses:
        return "?  no loss values logged"

    # Loss should decline over training. Allow some noise — last quartile mean < first quartile mean.
    q1 = sum(losses[: max(1, len(losses) // 4)]) / max(1, len(losses) // 4)
    q4 = sum(losses[-max(1, len(losses) // 4):]) / max(1, len(losses) // 4)
    loss_dropped = q4 < q1

    if cond in ("dpo", "sftdpo_alpaca"):
        if not margins:
            return "?  loss looks " + ("OK" if loss_dropped else "FLAT/RISING") + ", but no reward margins logged"
        final_margin = sum(margins[-max(1, len(margins) // 5):]) / max(1, len(margins) // 5)
        margin_pos = final_margin > 0
        if loss_dropped and margin_pos:
            return f"PASS  loss {q1:.3f} → {q4:.3f}, final margin {final_margin:+.3f}"
        if not margin_pos:
            return f"FAIL  reward margin collapsed (final {final_margin:+.3f}) — use earlier epoch"
        return f"CHECK loss not dropping (q1 {q1:.3f}, q4 {q4:.3f}), but margin {final_margin:+.3f}"
    # SFT: only loss matters
    return f"PASS  loss {q1:.3f} → {q4:.3f}" if loss_dropped else f"CHECK loss flat/rising ({q1:.3f} → {q4:.3f})"


def print_condition(cond: str, slurm_dir: Path):
    print(f"\n========== {cond.upper()} ==========")
    parent = CKPTS_DIR / cond
    ckpt = latest_checkpoint(parent)
    if ckpt is None:
        print(f"  no checkpoint found under {parent}")
        return

    size = adapter_size_mb(ckpt)
    size_str = f"{size:.1f} MB" if size is not None else "(missing)"
    print(f"  checkpoint:  {ckpt}")
    print(f"  adapter:     {size_str}")

    traj = check_trajectory(ckpt)
    if traj.get("missing_state"):
        print(f"  no trainer_state.json under {ckpt}")
        return

    losses, margins, accs = traj["losses"], traj["margins"], traj["accs"]
    if losses:
        print(f"  loss     n={len(losses):4d}  first={fmt_list(losses)}  last={fmt_list(losses[-3:])}  "
              f"min={min(losses):.3f}  max={max(losses):.3f}")
    if margins:
        print(f"  margin   n={len(margins):4d}  first={fmt_list(margins)}  last={fmt_list(margins[-3:])}  "
              f"min={min(margins):.3f}  max={max(margins):.3f}")
    if accs:
        print(f"  acc      n={len(accs):4d}  first={fmt_list(accs)}  last={fmt_list(accs[-3:])}")

    logs = find_stderr_logs(cond, slurm_dir)
    if not logs:
        print(f"  no stderr files matching {slurm_dir}/{cond}.*.err")
    else:
        latest_log = logs[-1]
        hits = scan_errors(latest_log)
        if hits:
            print(f"  stderr ({latest_log.name}) has {len(hits)} matching lines:")
            for h in hits:
                print(f"    {h}")
        else:
            print(f"  stderr ({latest_log.name}):  clean")

    print(f"\n  >>> {verdict(cond, traj)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("conditions", nargs="*",
                        help=f"One or more of {TRAIN_CONDITIONS}. Default: all of them.")
    parser.add_argument("--slurm-dir", default=str(SLURM_DIR))
    args = parser.parse_args()

    wanted = args.conditions or TRAIN_CONDITIONS
    bad = [c for c in wanted if c not in TRAIN_CONDITIONS]
    if bad:
        print(f"unknown conditions: {bad}; valid: {TRAIN_CONDITIONS}", file=sys.stderr)
        sys.exit(2)

    for cond in wanted:
        print_condition(cond, Path(args.slurm_dir))


if __name__ == "__main__":
    main()
