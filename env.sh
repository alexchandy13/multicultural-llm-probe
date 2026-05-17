# Source on the login node before any job submission.
# Redirects HF caches to $HOME so we stay within the 30GB Nexus quota.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export BITSANDBYTES_NOWELCOME=1
export PYTHONUNBUFFERED=1   # flush stdout live so `tail -f *.out` shows progress

# Conditions used everywhere downstream.
export CONDITIONS="base sft dpo sftdpo instruct"

# Alpaca robustness variant — separate from CONDITIONS so the primary eval/CULNIG
# SLURM scripts (which read $CONDITIONS) keep targeting the HH-RLHF conditions.
# The alpaca-variant SLURM scripts (slurm/{eval,culnig}_alpaca_job.sh) define their
# own CONDS arrays inline rather than reading from this, but the variable is here
# for convenience when running things manually on the login node.
export CONDITIONS_ALPACA="sft_alpaca sftdpo_alpaca"

# LIMA robustness variant — same pattern as CONDITIONS_ALPACA. The lima-variant
# SLURM scripts (slurm/{eval,culnig}_lima_job.sh) define their own CONDS arrays
# inline; this is here for convenience on the login node.
export CONDITIONS_LIMA="sft_lima sftdpo_lima"

# Cultures from NormAd; split for paper-side analysis.
export WESTERN_CULTURES="US UK Germany Spain Australia"
export NON_WESTERN_CULTURES="Japan China India Iran Indonesia Nigeria Mexico South_Korea"
