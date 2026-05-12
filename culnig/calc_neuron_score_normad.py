"""Gradient-attribute MLP neurons on NormAd / NormAdctrl per condition.

Mirrors upstream CULNIG `calc_neuron_score.py`: for each example, compute |grad * act|
on each MLP intermediate neuron for the target token, average over examples, save a
[layer, neuron] tensor per condition.

The only differences from upstream are:
  - dataset = NormAd (vs. BLEnD)
  - control = NormAdctrl (vs. BLEnDctrl)
  - output dir = outputs/neurons/normad_{condition}/

Core gradient logic is intentionally identical to upstream.

Usage:
    python culnig/calc_neuron_score_normad.py --condition sft
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm

from evaluate._common import (
    PROJECT_ROOT,
    load_model_for_eval,
    resolve_condition,
)


def example_text(ex: dict) -> str:
    for key in ("story", "scenario", "situation", "text"):
        if key in ex and ex[key]:
            return str(ex[key])
    return ""


def register_mlp_hooks(model):
    """Capture intermediate activations from each MLP up_proj output.

    For Llama-family models the MLP is: down_proj(silu(gate_proj(x)) * up_proj(x)).
    We hook the post-activation (gate*up) tensor — the input to down_proj — since
    that is the canonical "neuron" CULNIG attributes to.
    """
    acts = {}
    handles = []

    def make_hook(layer_idx):
        def hook(_module, inputs, _output):
            # inputs[0] is the down_proj input, shape [B, T, intermediate]
            x = inputs[0]
            x.retain_grad()
            acts[layer_idx] = x
        return hook

    # transformers.LlamaModel exposes .model.layers; use that path.
    layers = model.base_model.model.model.layers if hasattr(model, "base_model") else model.model.layers
    for i, layer in enumerate(layers):
        h = layer.mlp.down_proj.register_forward_hook(make_hook(i))
        handles.append(h)
    return acts, handles


def score_one_split(model, tokenizer, dataset, max_examples: int) -> torch.Tensor:
    """Run CULNIG-style |grad * activation| attribution.

    Returns a tensor of shape [n_layers, intermediate_dim] = per-neuron scalar score
    averaged across examples.
    """
    model.eval()
    n_layers = model.config.num_hidden_layers
    inter = model.config.intermediate_size
    accum = torch.zeros(n_layers, inter, dtype=torch.float32)
    counted = 0

    for idx, ex in enumerate(tqdm(dataset, desc="scoring")):
        if idx >= max_examples:
            break
        text = example_text(ex)
        if not text:
            continue

        acts, handles = register_mlp_hooks(model)
        try:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(model.device) for k, v in enc.items()}

            # Forward
            out = model(**enc, labels=enc["input_ids"])
            loss = out.loss
            model.zero_grad(set_to_none=True)
            loss.backward()

            # Accumulate |grad * activation| for the last non-pad token, averaged across tokens.
            for layer_idx, act in acts.items():
                if act.grad is None:
                    continue
                score = (act.detach().abs() * act.grad.detach().abs()).sum(dim=(0, 1)).float().cpu()
                accum[layer_idx] += score
            counted += 1
        finally:
            for h in handles:
                h.remove()

    if counted == 0:
        raise RuntimeError("scored 0 examples — check dataset")
    return accum / counted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["base", "sft", "dpo", "instruct"])
    parser.add_argument("--normad-path", default=str(PROJECT_ROOT / "data" / "NormAd"))
    parser.add_argument("--normad-ctrl-path", default=str(PROJECT_ROOT / "data" / "NormAdctrl"))
    parser.add_argument("--max-examples", type=int, default=2000,
                        help="Upper bound — NormAd has ~2.3k scenarios.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cond = resolve_condition(args.condition)
    tokenizer, model = load_model_for_eval(cond)

    # NB: CULNIG runs the same scoring twice — once on the cultural set, once on the
    # culture-stripped control — then subtracts to isolate culture-specific neurons.
    normad = load_from_disk(args.normad_path)
    normad_ctrl = load_from_disk(args.normad_ctrl_path)

    scores_main = score_one_split(model, tokenizer, normad, args.max_examples)
    scores_ctrl = score_one_split(model, tokenizer, normad_ctrl, args.max_examples)

    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "outputs" / "neurons" / f"normad_{args.condition}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(scores_main, out_dir / "scores_normad.pt")
    torch.save(scores_ctrl, out_dir / "scores_normad_ctrl.pt")
    torch.save(scores_main - scores_ctrl, out_dir / "scores_delta.pt")

    meta = {
        "condition": args.condition,
        "n_layers": int(scores_main.shape[0]),
        "intermediate_size": int(scores_main.shape[1]),
        "max_examples": args.max_examples,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote scores to {out_dir}")


if __name__ == "__main__":
    main()
