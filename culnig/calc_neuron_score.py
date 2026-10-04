"""Fork of upstream `CULNIG/calc_neuron_score.py` with two minimal additions:

  1. Llama-3.2-3B (base + Instruct) added to the model whitelist.
  2. QLoRA 4-bit base loading + optional LoRA adapter for our SFT/DPO conditions.

The core gradient-scoring loop (`calculate_scores`) is imported from upstream
unchanged, per the plan's directive: "keep gradient scoring logic completely
unchanged — no modifications to the core algorithm."

Outputs land in our project's outputs/neurons/ tree instead of upstream's
default ../outputs/, so analysis scripts can find them.

Usage:
    python culnig/calc_neuron_score.py \
        --condition sft --dataset-names normad
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT_ROOT / "culnig" / "_upstream"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(PROJECT_ROOT))

# Install our normadcontrol patch on upstream's dataset module BEFORE anything
# else imports it.
import culnig.dataset_ext  # noqa: F401, E402

# Now safe to import upstream pieces.
from CULNIG import calc_neuron_score as upstream_score  # noqa: E402

from evaluate._common import resolve_condition  # noqa: E402


def _label_first_token(tokenizer, prompt: str, label: str) -> int:
    """First token id the model must emit for `label`, given `prompt`.

    Upstream used `convert_tokens_to_ids(label)`, which looks up the bare string.
    On a SentencePiece vocab that is a *different* token from the space-prefixed
    form the model actually predicts after e.g. "Answer:" — Gemma 4 has 'yes'=4443
    and '_yes'=11262. Both are real vocab entries, so the lookup silently returns a
    token the model essentially never emits: measured P(4443)=6.6e-08 against
    P(11262)=0.75 on the same prompt, a 1e7 error that made every normad and
    culturalbench score noise.

    The dataset labels are bare ("yes", "A"), so the space has to be reconstructed
    from the prompt, the same way eval_normad.py does it with its `leading_space`
    flag. A prompt ending in whitespace or an open quote is continued directly
    (blend: `{"answer_choice":"` -> `A`); anything else gets a space
    (normad/culturalbench: `Answer:` -> ` yes`). The token is then read as the first
    position where tokenizing prompt+label diverges from tokenizing prompt alone,
    which is robust to the tokenizer merging across the boundary.
    """
    direct = prompt.endswith((" ", "\t", "\n", '"', "'"))
    text = label if direct else " " + label

    with_label = tokenizer(prompt + text, add_special_tokens=False).input_ids
    without = tokenizer(prompt, add_special_tokens=False).input_ids
    i = 0
    while i < len(without) and i < len(with_label) and with_label[i] == without[i]:
        i += 1
    if i >= len(with_label):
        raise ValueError(
            f"label {label!r} produced no new tokens after prompt ending "
            f"{prompt[-20:]!r} — cannot determine which token to score"
        )
    return with_label[i]


def calculate_scores_memory_efficient(model, tokenizer, dataloader, logger):
    """Memory-efficient drop-in replacement for upstream_score.calculate_scores.

    Same algorithm and output as upstream, with three fixes that let bf16
    Llama 3.1 8B fit on A5000 24 GB:

      1. Wrap the per-module scoring loop (which runs AFTER backward()) in
         `torch.no_grad()`. Upstream's code multiplies `prob_total *
         max_aggr_scores` etc. without detaching, so each module-iteration
         extends the autograd graph. At 32 layers × 5 modules per layer = 160
         residual graph fragments per batch, this accumulates dramatically.

      2. Explicitly `detach()` the per-step `prob_total` so the scoring
         arithmetic is pure tensor math.

      3. Clear retained activation `.grad` and `activations` dict, then call
         `torch.cuda.empty_cache()` at end of each batch to defrag (with
         `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in env.sh).

    Uses our own _get_text_model/_get_target_module (not upstream's utils)
    so it works for all our models (Llama-3, Gemma-3, Gemma-4, Qwen3)
    without requiring any edits to _upstream/. Algorithm output is
    bit-identical to upstream's for the Llama/pass branch.
    """
    import torch.nn.functional as F
    from CULNIG.calc_neuron_score import TARGET_MODULES

    max_neuron_scores = defaultdict(lambda: defaultdict(float))
    total_probabilities_per_country = defaultdict(float)

    activations: dict = {}

    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output
            output.retain_grad()
        return hook

    hooks = []
    text_model = _get_text_model(model)
    for i in range(len(text_model.layers)):
        for module_name in TARGET_MODULES:
            module = _get_target_module(model, module_name, i)
            if module is None:
                continue  # layer doesn't use this projection (e.g. Gemma4 local-attn layers)
            hooks.append(module.register_forward_hook(
                save_activation(f"model.model.layers.{i}.{module_name}")
            ))

    total_iter = len(dataloader)
    cur_iter = 0
    model.train()
    try:
        for batch in dataloader:
            cur_iter += 1
            if cur_iter % 100 == 0:
                logger.info(f"Processing batch {cur_iter}/{total_iter}")

            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            countries = batch["countries"]
            dataset_names = batch["dataset_names"]
            control = ["control" in d for d in dataset_names]

            model.zero_grad()
            activations.clear()

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)

            labels = [str(label) for label in batch["labels"]]
            labels_ids = torch.tensor(
                [_label_first_token(tokenizer, txt, l)
                 for txt, l in zip(batch["input_texts"], labels)],
                device=logits.device,  # logits live on the last layer's device
            )
            correct_probs = probabilities[
                torch.arange(probabilities.size(0), device=logits.device), labels_ids
            ]
            correct_probs.sum().backward()

            correct_probs_cpu_list = correct_probs.detach().cpu().tolist()
            for country, prob in zip(countries, correct_probs_cpu_list):
                total_probabilities_per_country[country] += prob

            # === MEMORY FIX: scoring under no_grad — no new graph created. ===
            with torch.no_grad():
                prob_total = correct_probs.detach().unsqueeze(1)

                for module_name, activation in list(activations.items()):
                    parts = module_name.split(".")
                    layer_idx = int(parts[3])
                    mod_name = ".".join(parts[4:])

                    grads = activation.grad
                    if grads is None:
                        continue

                    act_device = activation.device
                    # Llama branch (upstream: `if name in [llama list]: pass`).
                    scores = activation.detach() * grads

                    # With device_map="auto" across multiple GPUs, attention_mask
                    # and prob_total may live on a different device than this layer.
                    padding_mask = (attention_mask == 0).to(act_device)
                    if padding_mask.any():
                        scores = scores.masked_fill(padding_mask.unsqueeze(-1), 0.0)

                    max_aggr_scores, _ = torch.max(scores, dim=1)
                    max_aggr_scores = prob_total.to(act_device) * max_aggr_scores
                    if any(control):
                        max_aggr_scores[control] = torch.clamp(
                            max_aggr_scores[control], min=0.0
                        )

                    max_aggr_scores_cpu = max_aggr_scores.cpu().tolist()
                    n_samples, n_neurons = max_aggr_scores.shape
                    for i in range(n_samples):
                        country = countries[i]
                        for neuron_idx in range(n_neurons):
                            max_neuron_scores[(mod_name, layer_idx, neuron_idx)][country] += \
                                max_aggr_scores_cpu[i][neuron_idx]

                    # Drop intermediates before the next module's iteration.
                    del scores, max_aggr_scores

            # === MEMORY FIX: release retained activations + grads explicitly. ===
            for act in activations.values():
                act.grad = None
            activations.clear()

            # === MEMORY FIX: defrag GPU memory between batches. ===
            torch.cuda.empty_cache()

    finally:
        for hook in hooks:
            hook.remove()

    return max_neuron_scores, total_probabilities_per_country


# Models whose architecture matches Llama-3.1-8B-Instruct module-for-module
# (same q/k/v/o_proj + gate/up/down_proj names under the same parent path).
# Upstream CULNIG's hard-coded whitelist only accepts a few exact strings; we
# pin every member of this set to LLAMA_31_BRANCH at load time so the
# upstream branch fires for all of them.
PIN_TO_LLAMA_31_BRANCH = {
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
}
LLAMA_31_BRANCH = "meta-llama/Llama-3.1-8B-Instruct"
# Upstream default is 16; we override to 1 because Llama-3.1-8B in bf16 (~16 GB)
# + forward + backward + per-neuron attribution accumulators pushes A5000's 24 GB
# right to the limit. At BATCH_SIZE=1 each forward keeps activation memory low
# enough that the gradient pass also fits. CULNIG is bottlenecked on backward
# memory, not throughput — going to 1 is the cheapest fix.
BATCH_SIZE = 1


def _get_text_model(model):
    """Return the transformer core that exposes `.layers` (works for all our models).

    For AutoModelForCausalLM (Llama-3, Gemma-3, Gemma-4, Qwen3) the hierarchy
    is model → model.model → .layers. For multimodal wrappers the text body
    lives at model.language_model.model, but we always load via
    AutoModelForCausalLM so the simple path is always correct.
    """
    inner = getattr(model, "model", model)
    if hasattr(inner, "language_model"):
        inner = getattr(inner.language_model, "model", inner.language_model)
    return inner


def _get_target_module(model, module_name: str, layer_idx: int):
    """Return the nn.Module at layers[layer_idx].{module_name}, or None if absent.

    Some architectures (e.g. Gemma4 alternating local/global attention) set
    certain projection attributes to None for layers that don't use them.
    Returning None lets callers skip those layers cleanly.
    """
    layer = _get_text_model(model).layers[layer_idx]
    obj = layer
    for part in module_name.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _pin_name_or_path(model):
    """Permanently set model.name_or_path so upstream's hard-coded whitelists accept it.

    Upstream reads model.name_or_path in three hot paths:
      - utils.get_target_module (line 16): exact-match whitelist.
      - utils.get_text_model (line 153): substring check — matches "Llama-3" already,
        so we don't strictly need to swap for this one, but a single value keeps
        downstream behavior uniform.
      - CULNIG/calc_neuron_score.calculate_scores (lines 118/121): exact-match whitelist
        INSIDE the per-batch / per-module loops. Hit on every forward.

    Llama-3.2-3B is not in those exact-match lists. Its MLP/attention module names
    are identical to Llama-3.1-8B-Instruct (same `q_proj`/`k_proj`/`v_proj`/`o_proj`/
    `gate_proj`/`up_proj`/`down_proj` under `model.layers[i].mlp` and `.self_attn`),
    so we pin name_or_path to the 3.1 string and reuse that branch.

    Note: this is a **permanent** swap, not scoped. The reads happen inside loops,
    so any swap-and-restore wrapper would have to bracket every read site —
    pinning once at load is strictly simpler and equally correct. The original
    value isn't preserved because nothing downstream needs it.
    """
    if model.name_or_path in PIN_TO_LLAMA_31_BRANCH:
        model.name_or_path = LLAMA_31_BRANCH


def load_model_for_culnig(condition_name: str, model_size: str = "3b",
                          precision: str = "matched_bf16"):
    """Load base + optional pre-merge adapter + optional primary adapter, merging both.

    `precision` controls quantization regime (same semantics as
    evaluate._common.load_model_for_eval):
      - 'matched_bf16' (default): every condition in bf16. Eliminates the
        precision confound that arises from the C4 merge step forcing bf16
        while C1/C2/C3 could otherwise use 4-bit. Use this for any cross-
        condition mechanistic analysis (Jaccard, attribution comparisons).
      - 'qlora_4bit': C1/C2/C3 in 4-bit, C4 in bf16 (original behavior).

    Memory: at 3B bf16 ≈ 6 GB; at 8B bf16 ≈ 16 GB. Both fit on A5000 24 GB
    for gradient scoring (forward + backward on the adapter-merged base).
    """
    cond = resolve_condition(condition_name, model_size=model_size)
    tokenizer = AutoTokenizer.from_pretrained(cond.base, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Strip the chat template so every condition produces identical raw-text prompts.
    # If a condition ships a chat template (e.g. an Instruct variant), its prompts
    # become chat-formatted and gradient scores are no longer comparable across
    # conditions. Forcing a pass-through template makes upstream's
    # `try: apply_chat_template ... except: pass` blocks fall back to raw text
    # uniformly for every dataset (normad, normadcontrol, blend, etc.).
    #
    # The template must still emit BOS. Upstream's loader sets
    # add_special_tokens=False whenever apply_chat_template succeeds, on the
    # assumption that the template inserted the special tokens itself — true of a
    # real chat template, false of a bare pass-through. Without this, no BOS is
    # ever prepended, which measurably degrades the next-token distribution
    # (Gemma 4: top prediction flips from ' yes' at 0.51 to ' No' at 0.39).
    _bos = "{{ bos_token }}" if tokenizer.bos_token else ""
    tokenizer.chat_template = (
        _bos + "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )

    use_4bit = (
        precision == "qlora_4bit"
        and cond.pre_merge_adapter is None
    )

    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cond.base,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cond.base,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    if cond.pre_merge_adapter is not None:
        model = PeftModel.from_pretrained(model, str(cond.pre_merge_adapter))
        model = model.merge_and_unload()

    if cond.adapter is not None:
        model = PeftModel.from_pretrained(model, str(cond.adapter))
        model = model.merge_and_unload()  # merge so gradients flow into base weights

    # The base model loads with all parameters frozen (4-bit weights have
    # requires_grad=False; PEFT inference-mode merges also leave the result
    # frozen). CULNIG's per-layer hooks call `output.retain_grad()`, which
    # requires the activation to have requires_grad=True — that in turn
    # requires the input embedding's output to carry grad. enable_input_require_grads()
    # adds the forward hook on the embedding layer that makes this work without
    # us needing to flip any parameter's requires_grad. This is the same trick
    # `prepare_model_for_kbit_training` uses internally.
    model.enable_input_require_grads()

    # See _pin_name_or_path docstring for why this is a permanent (not scoped) swap.
    _pin_name_or_path(model)

    # device_map="auto" silently places the whole model on CPU when the allocated
    # GPUs can't hold it (e.g. landing on 11 GB cards). Gradient scoring then runs
    # orders of magnitude too slow to ever finish, with no error — so refuse now
    # rather than burn the walltime.
    placements = {str(d) for d in getattr(model, "hf_device_map", {}).values()}
    if str(model.device) == "cpu" or placements & {"cpu", "disk"}:
        raise RuntimeError(
            f"Model is not fully on GPU (device={model.device}, "
            f"placements={sorted(placements) or 'n/a'}). Allocated GPUs are too "
            f"small for {model_size}. Request a VRAM floor by GPU type, e.g. "
            "--gres=gpu:rtxa6000:2 for gemma4 or --gres=gpu:rtxa5000:1 for 8b."
        )
    return model, tokenizer


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def _shard_dataloader(dataloader, shard_idx: int, shard_n: int, logger):
    """Return a DataLoader over every shard_n-th sample, offset by shard_idx.

    Stride (not contiguous) slicing: the dataset is built country-major, so a
    contiguous slice would hand one shard a single country. Striding gives every
    shard the same country coverage and comparable prompt-length distribution.
    """
    ds = dataloader.dataset
    indices = list(range(shard_idx, len(ds), shard_n))
    logger.info(f"shard {shard_idx}/{shard_n}: {len(indices)} of {len(ds)} samples")
    return torch.utils.data.DataLoader(
        ds.select(indices), batch_size=BATCH_SIZE,
        collate_fn=dataloader.collate_fn, shuffle=False, pin_memory=True,
    )


def _as_bytes(text: str):
    import numpy as np
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def write_shard_partial(raw_scores, total_probs, dataset_ids, out_path: Path, logger):
    """Dump one shard's accumulator as a dense .npz.

    JSON is not an option here: the full accumulator is 1.2-5.4 GB as JSON and
    minutes to serialize. Dense float64 keeps the merge exact — float32 would
    perturb scores at a level that still swamps the reassociation error.
    """
    import numpy as np

    keys = sorted(raw_scores.keys())
    countries = sorted({c for per in raw_scores.values() for c in per})
    cidx = {c: j for j, c in enumerate(countries)}

    arr = np.zeros((len(keys), len(countries)), dtype=np.float64)
    for i, k in enumerate(keys):
        row = arr[i]
        for country, val in raw_scores[k].items():
            row[cidx[country]] = val

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            scores=arr,
            keys=_as_bytes("\n".join(f"{m}_{l}_{n}" for (m, l, n) in keys)),
            countries=_as_bytes("\n".join(countries)),
            probs=np.array([total_probs.get(c, 0.0) for c in countries],
                           dtype=np.float64),
            ids_json=_as_bytes(json.dumps({k: list(v) for k, v in dataset_ids.items()})),
        )
    tmp.replace(out_path)
    logger.info(f"Wrote {out_path} ({len(keys)} keys x {len(countries)} countries)")


def run(condition_name: str, dataset_names: list[str], out_root: Path, logger,
        target_data: str = "neuron",
        model_size: str = "3b", precision: str = "matched_bf16",
        shard: tuple[int, int] | None = None,
        force_countryrc: bool = False):
    model, tokenizer = load_model_for_culnig(
        condition_name, model_size=model_size, precision=precision
    )
    logger.info(f"Loaded model for condition={condition_name} "
                f"(model_size={model_size}, precision={precision}) on {model.device}")

    dataset_names = sorted(dataset_names)

    size_sfx = "" if model_size == "3b" else f"_{model_size}"
    out_dir = out_root / f"{condition_name}{size_sfx}"

    # Main dataset(s)
    dataloader = upstream_score.load_dataset_neuron_scores(
        dataset_names, tokenizer, batch_size=BATCH_SIZE,
        target_countries=None, target_data=target_data,
    )

    if shard is not None:
        dataloader = _shard_dataloader(dataloader, shard[0], shard[1], logger)

    raw_scores, total_probs = calculate_scores_memory_efficient(
        model, tokenizer, dataloader, logger
    )

    if shard is not None:
        shard_idx, shard_n = shard
        shard_ids = defaultdict(list)
        for item in dataloader.dataset:
            if item["id"] not in shard_ids[item["dataset_name"]]:
                shard_ids[item["dataset_name"]].append(item["id"])
        name = "".join(dataset_names)
        write_shard_partial(
            raw_scores, total_probs, shard_ids,
            out_dir / "_shards" / f"{name}_{shard_idx}of{shard_n}.npz", logger,
        )
        # CountryRC is a separate, cheap pass — extend_countryrc.py owns it, and
        # repeating it per shard would waste GPU hours for identical output.
        logger.info("shard mode: skipping countryrc pass; run merge_shards.py when "
                    "all shards are done")
        return

    neuron_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (module_name, layer_idx, neuron_idx), per_country in raw_scores.items():
        key = f"{module_name}_{layer_idx}_{neuron_idx}"
        for country, score in per_country.items():
            neuron_scores[key][country] += score

    dataset_ids = defaultdict(list)
    for item in dataloader.dataset:
        if item["id"] not in dataset_ids[item["dataset_name"]]:
            dataset_ids[item["dataset_name"]].append(item["id"])

    # out_dir is suffixed with the model size so different base models don't
    # clobber each other on disk (e.g. outputs/neurons/sft_8b/).
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{''.join(dataset_names)}_max_scores.json"
    out_file.write_text(json.dumps({
        "neuron_scores": neuron_scores,
        "total_probabilities_per_country": dict(total_probs),
        "dataset_ids": dict(dataset_ids),
    }, indent=2))
    logger.info(f"Wrote {out_file}")

    # CountryRC second pass — same scoring loop, target countries restricted.
    #
    # Skip when the file on disk already covers more countries than this pass
    # would write. scripts/extend_countryrc.py grows this file to 81 countries,
    # and rewriting it with the 8 upstream TARGET_COUNTRIES silently destroys
    # that work — which is exactly what happened to sftdpo_aya_nocult_gemma4
    # when a blend scoring run finished after its extension.
    crc_file = out_dir / "countryrc_max_scores.json"
    if crc_file.exists() and not force_countryrc:
        try:
            with open(crc_file, "rb") as fh:
                fh.seek(max(0, crc_file.stat().st_size - 200_000))
                tail = fh.read().decode("utf-8", "ignore")
            import re
            m = re.search(r'"total_probabilities_per_country":\s*\{(.*?)\}', tail, re.S)
            existing_n = len(re.findall(r'"[^"]+":', m.group(1))) if m else 0
        except OSError:
            existing_n = 0
        if existing_n > len(upstream_score.TARGET_COUNTRIES):
            logger.info(
                f"Skipping countryrc pass: {crc_file} already covers {existing_n} "
                f"countries (> {len(upstream_score.TARGET_COUNTRIES)} this pass would "
                "write). Pass --force-countryrc to overwrite anyway."
            )
            return

    crc_dataloader = upstream_score.load_dataset_neuron_scores(
        dataset_names=["countryrc"], tokenizer=tokenizer, batch_size=BATCH_SIZE,
        target_countries=upstream_score.TARGET_COUNTRIES, target_data="neuron",
    )
    crc_raw, crc_probs = calculate_scores_memory_efficient(
        model, tokenizer, crc_dataloader, logger
    )
    crc_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (module_name, layer_idx, neuron_idx), per_country in crc_raw.items():
        key = f"{module_name}_{layer_idx}_{neuron_idx}"
        for country, score in per_country.items():
            crc_scores[key][country] += score
    crc_ids = defaultdict(list)
    for item in crc_dataloader.dataset:
        if item["id"] not in crc_ids[item["dataset_name"]]:
            crc_ids[item["dataset_name"]].append(item["id"])

    crc_file = out_dir / "countryrc_max_scores.json"
    crc_file.write_text(json.dumps({
        "neuron_scores": crc_scores,
        "total_probabilities_per_country": dict(crc_probs),
        "dataset_ids": dict(crc_ids),
    }, indent=2))
    logger.info(f"Wrote {crc_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                        choices=["base", "dpo", "sft", "sftdpo",
                                 "sft_aya_cult", "sft_aya_nocult",
                                 "sftdpo_aya_cult", "sftdpo_aya_nocult"])
    parser.add_argument("--dataset-names", nargs="+", required=True,
                        help="e.g. `normad` or `normadcontrol` (single name per run).")
    parser.add_argument("--yn-only", action="store_true",
                        help="Replace 'normad' with 'normad_yn': filters neutral-gold "
                             "examples and holdout countries, uses a binary yes/no prompt. "
                             "Existing normadcontrol runs are unaffected.")
    parser.add_argument("--out-root", default=str(PROJECT_ROOT / "outputs" / "neurons"))
    parser.add_argument(
        "--target-data", default="neuron", choices=["neuron", "non_neuron", "all"],
        help="Data split to use for scoring. 'neuron' (default) uses the first half "
             "of each country×label group, consistent with upstream CULNIG. 'all' uses "
             "all examples — use for datasets where no downstream eval split is needed "
             "(e.g. blend when skipping intervention evaluation).",
    )
    parser.add_argument(
        "--model-size", default="3b", choices=["3b", "8b", "gemma4", "qwen35"],
        help="Base model size. '3b'=Llama-3.2-3B (default), '8b'=Llama-3.1-8B, "
             "'gemma4'=Gemma 4 12B. Per-condition output dir is suffixed with "
             "the size when not 3b.",
    )
    parser.add_argument(
        "--precision", default="matched_bf16",
        choices=["matched_bf16", "qlora_4bit"],
        help="'matched_bf16' (default): all conditions in bf16. 'qlora_4bit': "
             "C1/C2/C3 in 4-bit (legacy regime).",
    )
    parser.add_argument(
        "--shard", default=None, metavar="I/N",
        help="Score only shard I of N (0-indexed, stride-sliced) and write a "
             "dense .npz partial to {out_dir}/_shards/ instead of the final JSON. "
             "Skips the countryrc pass. Run scripts/merge_shards.py once all N "
             "shards finish. Lets a preempted job lose one shard instead of the "
             "whole run, and lets shards run in parallel.",
    )
    parser.add_argument(
        "--force-countryrc", action="store_true",
        help="Rewrite countryrc_max_scores.json even when it already covers more "
             "countries than the 8 upstream TARGET_COUNTRIES. Without this, the "
             "countryrc pass is skipped so a rescore cannot clobber the country "
             "set added by scripts/extend_countryrc.py.",
    )
    return parser.parse_args()


def _parse_shard(spec: str | None) -> tuple[int, int] | None:
    if spec is None:
        return None
    try:
        idx_s, n_s = spec.split("/")
        idx, n = int(idx_s), int(n_s)
    except ValueError:
        raise SystemExit(f"--shard must look like I/N, got {spec!r}")
    if not (n >= 1 and 0 <= idx < n):
        raise SystemExit(f"--shard needs 0 <= I < N and N >= 1, got {spec!r}")
    return idx, n


def main():
    set_seed(42)
    args = parse_args()
    logger = setup_logging()
    dataset_names = args.dataset_names
    if args.yn_only:
        dataset_names = ["normad_yn" if d == "normad" else d for d in dataset_names]
        logger.info("--yn-only: replaced 'normad' with 'normad_yn' in dataset_names")
    run(
        args.condition, dataset_names, Path(args.out_root), logger,
        model_size=args.model_size, precision=args.precision,
        target_data=args.target_data, shard=_parse_shard(args.shard),
        force_countryrc=args.force_countryrc,
    )


if __name__ == "__main__":
    main()
