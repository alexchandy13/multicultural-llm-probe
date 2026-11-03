"""Which candidate HF model ids actually resolve, and how big are they?

MODEL_REGISTRY entries are only useful if the id exists and the weights fit the
card we schedule. This asks the hub -- metadata only, no weights downloaded --
and prints parameter counts plus the bf16 footprint, so a candidate can be
accepted or dropped before anything is wired into the eval scripts.

Model names move fast and several candidates below are guesses at a spelling,
so both plausible forms are listed where it is unclear. A line marked MISSING
means that exact string is wrong, not necessarily that the model does not exist.

    python scripts/check_model_ids.py
    python scripts/check_model_ids.py --extra Qwen/Qwen3.5-14B-Instruct
"""
from __future__ import annotations

import argparse

CANDIDATES = [
    # --- already in MODEL_REGISTRY, re-checked as a control on this script ---
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-4-12b-it",
    "google/gemma-4-26B-A4B-it",
    "allenai/Olmo-3-7B-Instruct-DPO",
    "Qwen/Qwen3.5-9B-Base",
    # --- Qwen instruct: spelling unconfirmed ---
    "Qwen/Qwen3.5-9B-Instruct",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.8-27B-Instruct",
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen2.5-7B-Instruct",
    # --- Cohere, built for multilingual coverage ---
    "CohereLabs/aya-expanse-8b",
    "CohereForAI/aya-expanse-8b",
    # --- Mistral ---
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "mistralai/Mistral-Small-4-Instruct",
    # --- other recent open-weight families ---
    "deepseek-ai/DeepSeek-V4-Lite-Instruct",
    "moonshotai/Kimi-K3-Instruct",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[],
                    help="Additional ids to check.")
    ap.add_argument("--max-gb", type=float, default=45.0,
                    help="bf16 footprint we can actually schedule on one card "
                         "(default 45, i.e. an A6000 with headroom).")
    args = ap.parse_args()

    from huggingface_hub import model_info
    from huggingface_hub.utils import (GatedRepoError, RepositoryNotFoundError)

    rows = []
    for mid in CANDIDATES + args.extra:
        try:
            info = model_info(mid, files_metadata=False)
        except RepositoryNotFoundError:
            rows.append((mid, "MISSING", None, ""))
            continue
        except GatedRepoError:
            rows.append((mid, "GATED", None, "accept terms / need token"))
            continue
        except Exception as e:
            rows.append((mid, "ERROR", None, type(e).__name__))
            continue
        params = None
        st = getattr(info, "safetensors", None)
        if st is not None:
            params = getattr(st, "total", None) or None
        rows.append((mid, "ok", params, ""))

    print(f"{'model id':48s} {'status':8s} {'params':>10s} {'bf16 GB':>8s}  note")
    for mid, status, params, note in rows:
        if params:
            gb = params * 2 / 1e9
            fits = "" if gb <= args.max_gb else "  TOO BIG for one card"
            print(f"{mid:48s} {status:8s} {params/1e9:9.1f}B {gb:8.1f}{fits}  {note}")
        else:
            print(f"{mid:48s} {status:8s} {'-':>10s} {'-':>8s}  {note}")


if __name__ == "__main__":
    main()
