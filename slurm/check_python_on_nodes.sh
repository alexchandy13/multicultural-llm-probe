#!/bin/bash
# Probe a set of Tron nodes for python3.12 + GPU availability.
#
# Compute nodes in Tron are heterogeneous — some have python3.12 at
# /usr/bin/python3.12, others only have python3.9 (old image). Running a job
# on a "bad" node fails instantly with "No such file or directory" since our
# packages are pinned to python3.12.
#
# This script submits one tiny probe job per node in the requested range,
# writes its result to slurm/node_checks/<node>.out, then prints a summary
# table once all jobs complete.
#
# Usage:
#   slurm/check_python_on_nodes.sh                       # default: A5000 (46-61) + A6000 (0-5)
#   slurm/check_python_on_nodes.sh rtxa5000 46 61        # custom GPU type + range
#   slurm/check_python_on_nodes.sh rtxa6000 0 5
#
# After it finishes, summary table tells you which nodes are safe to put in
# #SBATCH --nodelist= for the real jobs.

set -euo pipefail

GPU="${1:-rtxa5000}"
LO="${2:-46}"
HI="${3:-61}"

mkdir -p slurm/node_checks

for n in $(seq "$LO" "$HI"); do
    NODE="tron$(printf %02d $n)"     # tron06, tron47, etc.
    OUT="slurm/node_checks/${NODE}.out"
    rm -f "$OUT"
    cat > "/tmp/probe_${NODE}.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=probe_${NODE}
#SBATCH --nodelist=${NODE}
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:${GPU}:1
#SBATCH --time=00:02:00
#SBATCH --mem=1G
#SBATCH --output=${OUT}
#SBATCH --error=slurm/node_checks/${NODE}.err
echo "node=\$(hostname)"
if [ -x /usr/bin/python3.12 ]; then
    echo "python3.12: PRESENT"
else
    echo "python3.12: MISSING"
fi
ls /usr/bin/python3* 2>&1 | tr '\n' ' '
echo
EOF
    sbatch "/tmp/probe_${NODE}.sh" >/dev/null
done

echo "submitted probes for ${GPU} on tron${LO}..tron${HI}"
echo "wait ~2-3 minutes, then run:"
echo "    bash slurm/check_python_on_nodes.sh --summary ${GPU} ${LO} ${HI}"
echo
echo "or pass --summary as the first arg to this script to see results below"

# If invoked with --summary as 1st arg, print the table for the given range.
if [ "${1:-}" = "--summary" ]; then
    GPU="${2:-rtxa5000}"
    LO="${3:-46}"
    HI="${4:-61}"
    echo "=== Probe results for ${GPU} on tron${LO}..tron${HI} ==="
    for n in $(seq "$LO" "$HI"); do
        NODE="tron$(printf %02d $n)"
        f="slurm/node_checks/${NODE}.out"
        if [ -f "$f" ]; then
            status=$(grep "^python3.12:" "$f" 2>/dev/null | awk '{print $2}')
            echo "  ${NODE}: ${status:-NO_DATA}"
        else
            echo "  ${NODE}: NOT_RUN"
        fi
    done
fi
