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

# Cultures from NormAd; split for paper-side analysis.
export WESTERN_CULTURES="US UK Germany Spain Australia"
export NON_WESTERN_CULTURES="Japan China India Iran Indonesia Nigeria Mexico South_Korea"
