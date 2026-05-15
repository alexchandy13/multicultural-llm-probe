"""Smoke-test that a LoRA adapter checkpoint loads against the expected base model.

Useful when:
  - A checkpoint was transferred from another machine and you want to confirm
    PEFT/TRL version drift didn't break it before submitting downstream jobs.
  - A training job appeared to succeed but you want to verify the saved adapter
    actually round-trips through `PeftModel.from_pretrained`.

What it does (in order):
  1. Resolves the target — accepts either a specific `checkpoint-N` dir or a
     parent dir (in which case it picks the highest-numbered checkpoint).
  2. Reads `adapter_config.json` and prints the headline fields. This is where
     most version-drift errors surface (missing required fields, renamed enums).
  3. Loads the base model (default: 4-bit Llama 3.2 3B base).
  4. Applies the adapter via `PeftModel.from_pretrained`.
  5. Reports trainable parameter count to confirm the LoRA actually attached.

If anything fails, prints likely causes to stderr.

Usage:
    # The common case — verify the Alpaca SFT checkpoint
    python3.12 analysis/check_checkpoint.py checkpoints/sft_alpaca

    # Specify a different base (e.g. to check a Llama-3.1 adapter):
    python3.12 analysis/check_checkpoint.py --base meta-llama/Llama-3.1-8B-Instruct path/to/ckpt

    # Skip 4-bit if bitsandbytes is misbehaving:
    python3.12 analysis/check_checkpoint.py --no-quant checkpoints/sft_alpaca
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "meta-llama/Llama-3.2-3B"


def resolve_checkpoint(path: Path) -> Path:
    """Accept a specific `checkpoint-N` dir or a parent containing them."""
    if not path.exists():
        raise FileNotFoundError(f"path does not exist: {path}")
    if (path / "adapter_model.safetensors").exists() or (path / "adapter_config.json").exists():
        return path
    candidates = []
    for p in path.glob("checkpoint-*"):
        suffix = p.name.rsplit("-", 1)[-1]
        if suffix.isdigit():
            candidates.append((int(suffix), p))
    if not candidates:
        raise FileNotFoundError(
            f"no adapter files in {path}, and no checkpoint-* subdirs either"
        )
    return max(candidates)[1]


def show_adapter_config(ckpt: Path) -> dict | None:
    """Print key fields from adapter_config.json. Returns the parsed dict for later use."""
    cfg_path = ckpt / "adapter_config.json"
    if not cfg_path.exists():
        print(f"  (no adapter_config.json at {cfg_path})")
        return None
    cfg = json.loads(cfg_path.read_text())
    keys = ("peft_type", "task_type", "r", "lora_alpha", "lora_dropout",
            "bias", "target_modules", "base_model_name_or_path")
    print("[adapter_config.json]")
    for k in keys:
        if k not in cfg:
            continue
        v = cfg[k]
        if isinstance(v, list) and len(v) > 6:
            v = f"{v[:3]} ... +{len(v) - 3} more"
        print(f"  {k}: {v}")

    # Flag the most common source of pain — a base_model path that's
    # local-machine-specific. PEFT mostly ignores this if you pass an explicit
    # base, but flagging it helps diagnose the next bug.
    bmp = cfg.get("base_model_name_or_path", "")
    if bmp and (bmp.startswith("/") or bmp.startswith("~")):
        print(f"  [note] base_model_name_or_path is a local path; will be overridden "
              f"by --base. PEFT will still try to find it though if --base is wrong.")
    return cfg


def list_adapter_files(ckpt: Path):
    print("[checkpoint contents]")
    for f in sorted(ckpt.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            sz = f"{size / (1024*1024):.1f} MB"
        elif size > 1024:
            sz = f"{size / 1024:.1f} KB"
        else:
            sz = f"{size} B"
        marker = "/" if f.is_dir() else ""
        print(f"  {f.name}{marker:<2} {sz}")


def load_and_check(ckpt: Path, base_id: str, use_4bit: bool):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    print(f"\nLoading base model: {base_id!r}  ({'4-bit NF4' if use_4bit else 'bf16'})")
    kw = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    base = AutoModelForCausalLM.from_pretrained(base_id, **kw)
    print(f"  base loaded: {type(base).__name__}")

    print(f"\nApplying adapter from: {ckpt}")
    model = PeftModel.from_pretrained(base, str(ckpt))
    print("  adapter applied without error")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / total if total else 0.0
    print(f"\n  total params:     {total:,}")
    print(f"  trainable params: {trainable:,} ({pct:.3f}%)")

    # Sanity check: a Llama 3.2 3B QLoRA (r=16, 7 modules) has ~24M trainable.
    if trainable == 0:
        print("  [warn] zero trainable params — adapter may have attached as inference-only")
    elif trainable < 1_000_000:
        print("  [warn] trainable param count is unusually low for a LoRA r=16 setup")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Checkpoint dir or parent of checkpoint-N subdirs")
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help=f"HF model id for the base (default: {DEFAULT_BASE})")
    parser.add_argument("--no-quant", action="store_true",
                        help="Load base in bf16 instead of 4-bit. Useful if bitsandbytes "
                             "has issues, or if checking an adapter trained against a "
                             "bf16 base (e.g. C4 sftdpo).")
    parser.add_argument("--list-only", action="store_true",
                        help="Just inspect adapter_config.json and list files — don't load weights.")
    args = parser.parse_args()

    target = resolve_checkpoint(Path(args.path))
    print(f"==> checkpoint: {target}\n")
    list_adapter_files(target)
    print()
    show_adapter_config(target)

    if args.list_only:
        return

    load_and_check(target, args.base, use_4bit=not args.no_quant)
    print("\nOK")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Likely causes (in rough order of frequency):", file=sys.stderr)
        print("  1. PEFT version mismatch — adapter was saved with a different PEFT", file=sys.stderr)
        print("     version. Compare `pip show peft` here vs. where the ckpt was trained.", file=sys.stderr)
        print("     `cat <ckpt>/adapter_config.json` may have fields newer/older PEFT", file=sys.stderr)
        print("     doesn't recognize.", file=sys.stderr)
        print("  2. base_model_name_or_path mismatch — `--base` here doesn't match the", file=sys.stderr)
        print("     model the adapter expects. Re-run with the right `--base`.", file=sys.stderr)
        print("  3. Llama 3.2 license not accepted on this HF account.", file=sys.stderr)
        print("     `huggingface-cli login` and accept on huggingface.co/meta-llama/...", file=sys.stderr)
        print("  4. Corrupted safetensors. Verify with `ls -la <ckpt>/` — adapter_model.safetensors", file=sys.stderr)
        print("     should be ~46MB for Llama 3.2 3B + r=16 + 7 target modules.", file=sys.stderr)
        print("  5. bitsandbytes / CUDA issue (rare). Retry with --no-quant.", file=sys.stderr)
        sys.exit(1)
