"""Gzip the CULNIG score files in outputs/neurons, verifying before deleting.

culnig/score_io.py already reads either spelling -- countryrc_max_scores.json or
countryrc_max_scores.json.gz -- and its docstring says compressed is the intended
storage format. Files that were written uncompressed are pure waste: dense float
JSON gzips about 3.7x, so a 278 GB tree becomes roughly 75 GB and every consumer
keeps working untouched.

SAFETY. A file is removed only after its gzip has been read back and md5-matched
against the original byte-for-byte. Size and mtime are not checked, because
neither would catch a truncated write -- which is exactly the failure mode here,
since the filesystem is the thing that ran out of room. If the hashes disagree
the .gz is deleted and the original is left alone.

ORDER. Largest file first, so the first few completions free the most room. That
matters when starting near zero free space: writing the .gz needs the compressed
size available before the original can be released.

resolve_scores_path() prefers the uncompressed file when both exist, so a file is
never unreadable mid-run.

    python scripts/compress_neuron_scores.py --dry-run
    python scripts/compress_neuron_scores.py --workers 4
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNK = 8 << 20


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def md5_of_gzip(path: Path) -> str:
    """md5 of the DECOMPRESSED stream, so it is comparable to the original."""
    h = hashlib.md5()
    with gzip.open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def compress_one(src: str, level: int) -> tuple[str, str, int, int]:
    """Returns (path, status, bytes_before, bytes_after). Never raises."""
    p = Path(src)
    gz = p.with_suffix(p.suffix + ".gz")
    try:
        before = p.stat().st_size
    except FileNotFoundError:
        return (src, "vanished", 0, 0)

    if gz.exists():
        # A previous run may have been interrupted between writing and verifying.
        try:
            if md5_of_gzip(gz) == md5_of_file(p):
                p.unlink()
                return (src, "already-ok", before, gz.stat().st_size)
        except OSError:
            pass
        gz.unlink(missing_ok=True)

    tmp = gz.with_suffix(gz.suffix + ".partial")
    try:
        with open(p, "rb") as fin, gzip.open(tmp, "wb", compresslevel=level) as fout:
            shutil.copyfileobj(fin, fout, CHUNK)
        os.replace(tmp, gz)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        return (src, f"write-failed: {e}", before, 0)

    try:
        if md5_of_gzip(gz) != md5_of_file(p):
            gz.unlink(missing_ok=True)
            return (src, "MD5 MISMATCH — original kept", before, 0)
    except OSError as e:
        gz.unlink(missing_ok=True)
        return (src, f"verify-failed: {e}", before, 0)

    after = gz.stat().st_size
    p.unlink()
    return (src, "ok", before, after)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT / "outputs" / "neurons"),
                    help="Directory to walk (follows the scratch symlink).")
    ap.add_argument("--pattern", default="*_max_scores.json")
    ap.add_argument("--min-size", type=float, default=0.1, metavar="GB",
                    help="Skip files smaller than this; the tiny ones are not "
                         "worth a verify pass (default 0.1 GB).")
    ap.add_argument("--level", type=int, default=6,
                    help="gzip level. 6 is the default; 9 costs much more CPU "
                         "for a few percent on float JSON.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Files compressed in parallel. Each needs its output "
                         "size free on disk, so start at 1 when nearly full.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    cutoff = int(args.min_size * 1e9)
    files = sorted((p for p in root.rglob(args.pattern)
                    if p.is_file() and p.stat().st_size >= cutoff),
                   key=lambda p: -p.stat().st_size)
    total = sum(p.stat().st_size for p in files)
    print(f"root     : {root}")
    print(f"matched  : {len(files)} files, {total/1e9:.1f} GB")
    if not files:
        return
    if args.dry_run:
        for p in files[:15]:
            print(f"  {p.stat().st_size/1e9:7.2f} GB  {p}")
        if len(files) > 15:
            print(f"  ... and {len(files)-15} more")
        print(f"\nprojected at 3.7x: ~{total/3.7/1e9:.1f} GB kept, "
              f"~{total*(1-1/3.7)/1e9:.1f} GB freed")
        return

    freed = failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(compress_one, str(p), args.level): p for p in files}
        for i, fut in enumerate(as_completed(futs), 1):
            path, status, before, after = fut.result()
            if status in ("ok", "already-ok"):
                freed += before - after
                print(f"[{i}/{len(files)}] {status:10s} "
                      f"{before/1e9:6.2f} -> {after/1e9:5.2f} GB  "
                      f"(freed {freed/1e9:.1f} GB total)  {Path(path).name}",
                      flush=True)
            else:
                failures += 1
                print(f"[{i}/{len(files)}] {status}  {path}", file=sys.stderr, flush=True)

    print(f"\nfreed {freed/1e9:.1f} GB; {failures} failure(s)")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
