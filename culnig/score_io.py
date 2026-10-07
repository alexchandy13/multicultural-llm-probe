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
import re
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


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


# A neuron key maps to an object of per-country floats; a dataset_ids key maps to
# a list. So "key": { means a neuron, "key": <number> means one of its countries,
# and nothing inside dataset_ids matches either.
_TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"\s*:\s*(\{|-?\d[\d.eE+\-]*)')
_CHUNK = 1 << 22
# A match ending at the buffer edge may be truncated ("0.0001" cut from "0.00012"),
# so a match is only consumed once a delimiter proves the value ended.
_AFTER = frozenset(',}]\n\r\t ')
# Neuron keys are "<module>_<layer>_<index>", e.g. mlp.gate_proj_12_3. The files
# carry sibling top-level objects that also hold per-country numbers
# (total_probabilities_per_country, dataset_ids), so a {-valued key that is not
# neuron-shaped marks the end of neuron_scores.
_NEURON_KEY = re.compile(r"^[A-Za-z_.]+_\d+_\d+$")


def neuron_sums(path: str | Path) -> dict[str, float]:
    """Return {neuron_key: sum of its per-country scores}, without loading the file.

    Every consumer of these files only ever wants sum(score.values()) per neuron —
    see decide_culture_neurons.py — but json.loads holds the whole nested structure
    live, which peaks near 34 GB for a gemma4 normad pass. This keeps one float per
    neuron instead, a few tens of MB, so the selection step runs on a laptop.
    """
    resolved = resolve_scores_path(path)
    if resolved is None:
        raise FileNotFoundError(path)

    sums: dict[str, float] = {}
    cur: str | None = None
    acc = 0.0
    carry = ""
    started = False
    done = False

    with _open_text(resolved) as fh:
        while not done:
            chunk = fh.read(_CHUNK)
            at_eof = not chunk
            buf = carry + chunk
            last = 0
            for m in _TOKEN.finditer(buf):
                # Leave an edge-touching match for the next chunk; it may be cut short.
                if m.end() >= len(buf) and not at_eof:
                    break
                if not at_eof and buf[m.end()] not in _AFTER:
                    break
                key, val = m.group(1), m.group(2)
                last = m.end()
                if val == "{":
                    if key == "neuron_scores":
                        started = True
                        continue
                    if not started:
                        continue
                    if not _NEURON_KEY.match(key):
                        done = True        # left neuron_scores for a sibling object
                        break
                    if cur is not None:
                        sums[cur] = acc
                    cur, acc = key, 0.0
                elif cur is not None:
                    acc += float(val)
            if at_eof:
                break
            carry = buf[last:]             # keep ALL unconsumed text, never discard

    if cur is not None:
        sums[cur] = acc
    return sums


def dataset_ids(path: str | Path) -> dict[str, list]:
    """Return just the dataset_ids mapping, by brace-matching that one object."""
    resolved = resolve_scores_path(path)
    if resolved is None:
        raise FileNotFoundError(path)

    needle = '"dataset_ids"'
    buf = ""
    with _open_text(resolved) as fh:
        # dataset_ids is written after neuron_scores, so scan forward for it and
        # then keep only from there — the object itself is small.
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                raise ValueError(f"{resolved}: no dataset_ids object found")
            buf += chunk
            i = buf.find(needle)
            if i != -1:
                buf = buf[i + len(needle):]
                break
            buf = buf[-len(needle):]
        start = buf.find("{")
        while start == -1:
            more = fh.read(_CHUNK)
            if not more:
                raise ValueError(f"{resolved}: dataset_ids has no object")
            buf += more
            start = buf.find("{")
        depth, in_str, esc, j = 0, False, False, start
        while True:
            while j < len(buf):
                c = buf[j]
                if in_str:
                    if esc: esc = False
                    elif c == "\\": esc = True
                    elif c == '"': in_str = False
                elif c == '"': in_str = True
                elif c == "{": depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(buf[start:j + 1])
                j += 1
            more = fh.read(_CHUNK)
            if not more:
                raise ValueError(f"{resolved}: dataset_ids object is unterminated")
            buf += more
