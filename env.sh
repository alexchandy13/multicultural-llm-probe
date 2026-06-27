# Source on the login node before any job submission.
# Redirects HF caches to $HOME so we stay within the 30GB Nexus quota.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT

# HF cache location. On Nexus, redirect to scratch (the 30GB home quota can't
# hold a single 16GB Llama-3.1-8B download); locally, fall back to the standard
# user-cache path. The check is path existence: if /fs/nexus-scratch/$USER
# exists we're on Nexus and route there, otherwise use home.
if [ -d "/fs/nexus-scratch/$USER" ]; then
    export HF_HOME="/fs/nexus-scratch/$USER/hf_cache"
else
    export HF_HOME="$HOME/.cache/huggingface"
fi
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export BITSANDBYTES_NOWELCOME=1
export PYTHONUNBUFFERED=1   # flush stdout live so `tail -f *.out` shows progress

# The four conditions used everywhere downstream:
#   base, sft (C2), dpo (C3), sftdpo (C4).
export CONDITIONS="base sft dpo sftdpo"

# Cultures from NormAd; split for paper-side analysis.
export WESTERN_CULTURES="US UK Germany Spain Australia"
export NON_WESTERN_CULTURES="Japan China India Iran Indonesia Nigeria Mexico South_Korea"
