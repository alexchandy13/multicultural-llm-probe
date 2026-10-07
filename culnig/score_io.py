"""Transparent reading of CULNIG score files, compressed or not.

The per-country score files are dense float64 JSON and get large — the extended
81-country countryrc pass is ~3.5 GB per 8b condition and ~5.9 GB per gemma4 one,
47 GB across all ten. They gzip about 3.7x, so they are stored compressed and read
through here. Both spellings are accepted so older uncompressed files keep working:

    countryrc_max_scores.json
    countryrc_max_scores.json.gz

Deliberately dependency-free (no torch) so analysis/ can import it as cheaply as
culnig/ does.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path


def resolve_scores_path(path: str | Path) -> Path | None:
    """Return the path that exists — as given, or with .gz appended — else None.

    If both exist the uncompressed one wins, so a fresh scoring run's output is
    picked up ahead of a stale archive.
    """
    p = Path(path)
    if p.exists():
        return p
    gz = p.with_suffix(p.suffix + ".gz")
    return gz if gz.exists() else None


def scores_exist(path: str | Path) -> bool:
    return resolve_scores_path(path) is not None


def read_scores(path: str | Path) -> dict:
    """Load a score file, decompressing if it is stored as .gz.

    Raises FileNotFoundError naming the uncompressed path, since that is the name
    callers ask for.
    """
    resolved = resolve_scores_path(path)
    if resolved is None:
        raise FileNotFoundError(path)
    if resolved.suffix == ".gz":
        with gzip.open(resolved, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(resolved.read_text())
