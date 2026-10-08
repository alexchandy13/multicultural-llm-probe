"""Watch for complete shard sets, merge them, then run the culture-neuron selection.

Removes the merge-then-select babysitting from a sharded gemma4 scoring run. Polls
the condition dirs and, in order:

  1. any {dataset}_*of{N}.npz set that is complete  -> merge_shards.py
  2. any benchmark whose score and control files are both present, and whose
     all_neurons_* file is missing or older than them -> decide_culture_neurons.py

Both steps are idempotent, so a restart picks up wherever it left off. Work is
serialized: selection streams several GB and two at once would thrash NFS.

Usage (from the repo root, on the cluster where the scores are written):

    nohup python scripts/auto_merge_select.py --model-size gemma4 \
        --conditions sftdpo_aya_cult sftdpo_aya_nocult \
        > outputs/auto_merge_select.log 2>&1 &

    python scripts/auto_merge_select.py --model-size gemma4 --once --dry-run

--once does a single pass and exits, for checking what it would do.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEURONS = PROJECT_ROOT / "outputs" / "neurons"

# benchmark -> (score dataset, control dataset, --dataset-names value for selection)
BENCHMARKS = {
    "normad":        ("normad_yn", "normadcontrol", ["normad", "--yn-only"]),
    "culturalbench": ("culturalbench", "culturalbenchcontrol", ["culturalbench"]),
    "blend":         ("blend", "blendcontrol", ["blend"]),
}


LOCK = PROJECT_ROOT / "outputs" / ".auto_merge_select.lock"


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


def claim_lock() -> None:
    """Refuse to start if another watcher is live.

    Two watchers, or a watcher alongside a hand-run selection, will both see a
    missing all_neurons file and both start one. decide_culture_neurons.py writes
    with write_text(), which is not atomic, so concurrent writers can interleave
    and leave invalid JSON.
    """
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip())
        except (ValueError, OSError):
            pid = None
        if pid is not None:
            try:
                import os
                os.kill(pid, 0)          # signal 0 only tests for existence
            except ProcessLookupError:
                log(f"clearing stale lock from pid {pid}")
            except PermissionError:
                sys.exit(f"lock held by pid {pid} (another user?); remove {LOCK} if wrong")
            else:
                sys.exit(f"another watcher is running (pid {pid}); "
                         f"stop it first or remove {LOCK} if stale")
    import os
    LOCK.write_text(str(os.getpid()))


def release_lock() -> None:
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass


def scores_path(cond_dir: Path, dataset: str) -> Path | None:
    for name in (f"{dataset}_max_scores.json", f"{dataset}_max_scores.json.gz"):
        p = cond_dir / name
        if p.exists():
            return p
    return None


def complete_shard_sets(cond_dir: Path) -> list[tuple[str, int]]:
    """Datasets in _shards/ whose full set of N partials is present."""
    shard_dir = cond_dir / "_shards"
    if not shard_dir.is_dir():
        return []
    seen: dict[tuple[str, int], set[int]] = {}
    for p in shard_dir.glob("*of*.npz"):
        m = re.fullmatch(r"(.+)_(\d+)of(\d+)\.npz", p.name)
        if not m:
            continue
        ds, i, n = m.group(1), int(m.group(2)), int(m.group(3))
        seen.setdefault((ds, n), set()).add(i)
    return [(ds, n) for (ds, n), idx in sorted(seen.items()) if len(idx) == n]


def run(cmd: list[str], dry: bool) -> bool:
    log(("DRY-RUN " if dry else "RUN     ") + " ".join(cmd))
    if dry:
        return True
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        log(f"FAILED (exit {proc.returncode}): {' '.join(cmd)}")
        return False
    return True


def pass_once(conditions: list[str], size: str, benchmarks: list[str], dry: bool) -> int:
    """One sweep. Returns the number of actions taken."""
    acted = 0
    for cond in conditions:
        cond_dir = NEURONS / f"{cond}_{size}"
        if not cond_dir.is_dir():
            continue

        # 1. merge any complete shard set
        for ds, n in complete_shard_sets(cond_dir):
            log(f"{cond}/{ds}: {n}/{n} shards present -> merging")
            if run([sys.executable, "scripts/merge_shards.py", "--condition", cond,
                    "--model-size", size, "--dataset", ds], dry):
                acted += 1
            if dry:   # without a real merge the shards stay, so stop re-reporting them
                continue

        # 2. select where both score files are ready and the selection is stale
        for bench in benchmarks:
            score_ds, ctrl_ds, sel_args = BENCHMARKS[bench]
            score_p = scores_path(cond_dir, score_ds)
            ctrl_p = scores_path(cond_dir, ctrl_ds)
            if not score_p or not ctrl_p:
                continue
            suffix = "normad_yn" if bench == "normad" else bench
            sel = cond_dir / f"all_neurons_{suffix}_max.json"
            newest = max(score_p.stat().st_mtime, ctrl_p.stat().st_mtime)
            if sel.exists() and sel.stat().st_mtime >= newest:
                continue
            why = "missing" if not sel.exists() else "older than its scores"
            log(f"{cond}/{bench}: selection {why} -> deciding")
            if run([sys.executable, "culnig/decide_culture_neurons.py", "--condition", cond,
                    "--model-size", size, "--dataset-names"] + sel_args, dry):
                acted += 1
    return acted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=[
        "base", "sft_aya_cult", "sft_aya_nocult", "sftdpo_aya_cult", "sftdpo_aya_nocult"])
    ap.add_argument("--model-size", default="gemma4")
    ap.add_argument("--benchmarks", nargs="+", default=["normad", "culturalbench"],
                    choices=list(BENCHMARKS))
    ap.add_argument("--interval", type=int, default=300, help="seconds between sweeps")
    ap.add_argument("--once", action="store_true", help="one sweep, then exit")
    ap.add_argument("--dry-run", action="store_true", help="report actions without running them")
    args = ap.parse_args()

    if not args.dry_run:
        claim_lock()
    log(f"watching {args.model_size}: {', '.join(args.conditions)}")
    log(f"benchmarks: {', '.join(args.benchmarks)}; interval {args.interval}s")

    try:
        idle = 0
        while True:
            acted = pass_once(args.conditions, args.model_size, args.benchmarks, args.dry_run)
            if args.once:
                log(f"single pass done, {acted} action(s)")
                return
            idle = 0 if acted else idle + 1
            if idle == 1:
                log("nothing to do; will keep polling quietly")
            time.sleep(args.interval)
    finally:
        if not args.dry_run:
            release_lock()


if __name__ == "__main__":
    main()
