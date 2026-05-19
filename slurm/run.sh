#!/bin/bash
# One-command wrapper: sync repo, submit a SLURM job, wait for it, dump logs.
#
# Usage:
#   slurm/run.sh                              # defaults to slurm/sft_job.sh
#   slurm/run.sh slurm/dpo_job.sh             # any other job script
#
# Exits 0 if the job completed successfully, 1 otherwise. Handles both single
# jobs (--output=...%j...) and array jobs (--output=...%A_%a...).

set -euo pipefail

JOB_SCRIPT="${1:-slurm/sft_job.sh}"

if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "[error] $JOB_SCRIPT not found" >&2
    exit 1
fi

# 1. Pull latest code.
echo "[sync] git fetch + pull"
git fetch
git pull

# 2. Submit and capture jobid (--parsable strips the "Submitted batch job " prefix).
JOBID=$(sbatch --parsable "$JOB_SCRIPT")
echo "[submit] $JOB_SCRIPT -> job $JOBID"

# 3. Resolve log file globs from #SBATCH --output / --error directives.
#    %j -> jobid (single jobs); %A -> jobid + %a -> '*' (array tasks).
resolve_glob() {
    local pattern="$1"
    pattern="${pattern//%j/$JOBID}"
    pattern="${pattern//%A/$JOBID}"
    pattern="${pattern//%a/*}"
    echo "$pattern"
}
OUT_PATTERN=$(grep -oE -- '--output=[^ ]+' "$JOB_SCRIPT" | sed 's/--output=//')
ERR_PATTERN=$(grep -oE -- '--error=[^ ]+' "$JOB_SCRIPT" | sed 's/--error=//')
OUT_GLOB=$(resolve_glob "$OUT_PATTERN")
ERR_GLOB=$(resolve_glob "$ERR_PATTERN")

# 4. Poll until the job leaves the queue. Print state every 30s so a hung
#    PENDING is visible (most common cause: QoS or partition mismatch).
echo "[wait] polling squeue for $JOBID every 30s"
while true; do
    STATE=$(squeue -h -j "$JOBID" -o "%T" 2>/dev/null || true)
    if [[ -z "$STATE" ]]; then
        break
    fi
    echo "[wait]   state=$STATE  $(date +%H:%M:%S)"
    sleep 300
done

# 5. Final state from sacct (squeue forgets the job once it's done).
FINAL=$(sacct -j "$JOBID" -X -n -o State 2>/dev/null | head -1 | xargs)
echo "[done] job $JOBID final state: ${FINAL:-UNKNOWN}"

# 6. Dump all matching log files. Glob expansion handles array jobs.
shopt -s nullglob
echo
for f in $OUT_GLOB; do
    echo "=== stdout: $f ==="
    cat "$f"
    echo
done
for f in $ERR_GLOB; do
    echo "=== stderr: $f ==="
    cat "$f"
    echo
done

# 7. Exit code reflects success.
case "$FINAL" in
    COMPLETED) exit 0 ;;
    *)         exit 1 ;;
esac
