#!/bin/bash
# Move raw CULNIG score files to cluster scratch, verifying before anything is deleted.
#
# Local holds ~155 GB of *_max_scores.json in ~12 GB of headroom, and these are the
# only copies — the cluster versions were removed after each earlier transfer. So the
# order is push, checksum both sides, compare, and only then delete.
#
# What stays local: all_neurons_*.json and layer_stats_*.json, ~0.14 GB, which is
# everything the figures actually read.
#
#   bash scripts/archive_scores_to_cluster.sh push    <pattern>   # upload
#   bash scripts/archive_scores_to_cluster.sh verify  <pattern>   # md5 both sides
#   bash scripts/archive_scores_to_cluster.sh delete  <pattern>   # local rm, verify first
#
# <pattern> is a filename glob, e.g. 'blend*_max_scores.json' or 'countryrc*'.
#
#   bash scripts/archive_scores_to_cluster.sh push   'blend*_max_scores.json'
#   bash scripts/archive_scores_to_cluster.sh verify 'blend*_max_scores.json'
#   bash scripts/archive_scores_to_cluster.sh delete 'blend*_max_scores.json'

set -euo pipefail
REMOTE=achandy@nexusclip.umiacs.umd.edu
RDIR=/fs/nexus-scratch/achandy/multicultural-llm-probe/outputs/neurons
LDIR="$(cd "$(dirname "$0")/.." && pwd)/outputs/neurons"
ACTION="${1:?push | verify | delete}"
PATTERN="${2:?filename glob, e.g. 'blend*_max_scores.json'}"
SUMS=/tmp/archive_scores_sums

case "$ACTION" in
  push)
    echo "[push] $PATTERN -> $RDIR"
    rsync -av -W --include='*/' --include="$PATTERN" --exclude='*' \
        "$LDIR/" "$REMOTE:$RDIR/"
    echo
    echo "next: bash scripts/archive_scores_to_cluster.sh verify '$PATTERN'"
    ;;

  verify)
    echo "[verify] hashing locally (this reads every matched file)"
    ( cd "$LDIR" && find . -name "$PATTERN" | sed 's|^\./||' | sort \
        | while read -r f; do echo "$(md5 -q "$f")  $f"; done ) > "$SUMS.local"
    echo "  local files: $(wc -l < "$SUMS.local")"
    echo
    echo "Run this on the cluster, then paste the output into $SUMS.remote :"
    echo
    echo "  cd $RDIR && find . -name '$PATTERN' | sed 's|^\./||' | sort | xargs md5sum"
    echo
    echo "then: bash scripts/archive_scores_to_cluster.sh compare '$PATTERN'"
    ;;

  compare)
    [[ -f "$SUMS.local" && -f "$SUMS.remote" ]] || {
        echo "need $SUMS.local and $SUMS.remote (run verify first)" >&2; exit 1; }
    python3 - "$SUMS.local" "$SUMS.remote" <<'PY'
import sys
def load(p):
    d={}
    for line in open(p):
        line=line.strip()
        if not line: continue
        h,_,path=line.partition("  ")
        d["/".join(path.split("/")[-2:])]=h
    return d
L,R=load(sys.argv[1]),load(sys.argv[2])
both=sorted(set(L)&set(R)); bad=[k for k in both if L[k]!=R[k]]
for k in both: print(("MATCH " if L[k]==R[k] else "DIFFER") + "  " + k)
for k in sorted(set(L)-set(R)): print("LOCAL ONLY (not uploaded):", k)
for k in sorted(set(R)-set(L)): print("REMOTE ONLY:", k)
print(f"\n{len(both)-len(bad)}/{len(set(L)|set(R))} verified identical, {len(bad)} mismatched")
print("SAFE TO DELETE" if both and not bad and not (set(L)-set(R)) else "DO NOT DELETE")
PY
    ;;

  delete)
    grep -q "SAFE TO DELETE" "$SUMS.verdict" 2>/dev/null || {
        echo "refusing: run compare first and save its output to $SUMS.verdict" >&2
        echo "  bash scripts/archive_scores_to_cluster.sh compare '$PATTERN' | tee $SUMS.verdict" >&2
        exit 1; }
    echo "[delete] removing local copies of $PATTERN"
    find "$LDIR" -name "$PATTERN" -print -delete
    df -h / | tail -1
    ;;

  *) echo "unknown action: $ACTION" >&2; exit 1 ;;
esac
