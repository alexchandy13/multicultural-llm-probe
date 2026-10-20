"""Measure prompt token lengths per benchmark, to convert items/sec into tokens/sec.

The evals score rather than generate -- one forward pass over the prompt, then
the label log-probs, max_new_tokens=0 -- so throughput is naturally items/sec
and that is what the slurm progress bars record. Reporting tokens/sec/GPU needs
the prompt length, which is what this prints. Tokenizer only: no GPU, no model
weights, a couple of minutes on a login node.

    python scripts/count_prompt_tokens.py --model-size 8b --items-per-sec 17
    python scripts/count_prompt_tokens.py --model-size gemma4 --items-per-sec 8

Each benchmark is reconstructed with that eval's own prompt builder, in the
default shot setting used for the reported results, so the counts match what
the GPU actually saw. A benchmark whose dataset is missing locally is skipped
rather than failing the run.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def normad_prompts():
    from evaluate.eval_normad import (YN_PROMPT_TEMPLATE, country, load_normad,
                                      scenario_text)
    ds = load_normad(PROJECT_ROOT / "data" / "NormAd")
    return [YN_PROMPT_TEMPLATE.format(country=country(ex), scenario=scenario_text(ex))
            for ex in ds]


def culturalbench_prompts():
    from evaluate.eval_culturalbench import (DATA_PATH, PROMPT_SUFFIX,
                                             build_neutral_fewshot_prefix,
                                             build_neutral_shots)
    rows = json.loads(DATA_PATH.read_text())
    usable = [r for r in rows if r.get("reformatted_prompt")]
    # same 4 gold-matched neutral shots the _nfs4 results were produced with
    gold_yes_rate = sum(bool(r["answer"]) for r in usable) / len(usable)
    prefix = build_neutral_fewshot_prefix(build_neutral_shots(4, gold_yes_rate))
    return [prefix + r["reformatted_prompt"] + PROMPT_SUFFIX for r in usable]


def blend_prompts():
    from evaluate.eval_blend import load_blend
    ds = load_blend(PROJECT_ROOT / "data" / "BLEnD")
    return [ex["prompt"] for ex in ds]


def nlu_prompts(task):
    from datasets import load_dataset
    from evaluate.eval_nlu import DATASET_CONFIGS, format_prompt
    cfg = DATASET_CONFIGS[task]
    ds = (load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])
          if cfg["hf_name"] else load_dataset(cfg["hf_path"], split=cfg["split"]))
    return [format_prompt(task, ex) for ex in ds]


BENCHES = {
    "normad_yn":          normad_prompts,
    "culturalbench_nfs4": culturalbench_prompts,
    "blend":              blend_prompts,
    "nlu_boolq":          lambda: nlu_prompts("boolq"),
    "nlu_csqa":           lambda: nlu_prompts("csqa"),
    "nlu_mrpc":           lambda: nlu_prompts("mrpc"),
    "nlu_qnli":           lambda: nlu_prompts("qnli"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", default="8b",
                    help="Which tokenizer to measure with; Llama and Gemma "
                         "vocabularies give different counts for the same text.")
    ap.add_argument("--items-per-sec", type=float, default=None,
                    help="Measured items/sec, to print the implied tokens/sec.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from evaluate._common import model_id_for
    model_id = model_id_for(args.model_size)
    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"tokenizer: {model_id}\n")

    header = f"{'benchmark':20s} {'items':>7s} {'mean':>6s} {'median':>7s} {'p95':>6s}"
    if args.items_per_sec:
        header += f" {'tok/s/GPU':>10s}"
    print(header)

    for name, fn in BENCHES.items():
        try:
            prompts = fn()
        except Exception as e:
            print(f"{name:20s}  skipped ({type(e).__name__}: {e})")
            continue
        lens = sorted(len(tok(p).input_ids) for p in prompts)
        mean = statistics.mean(lens)
        row = (f"{name:20s} {len(lens):7d} {mean:6.0f} "
               f"{statistics.median(lens):7.0f} {lens[int(0.95 * len(lens))]:6d}")
        if args.items_per_sec:
            row += f" {mean * args.items_per_sec:10,.0f}"
        print(row)

    print("\nSingle scoring pass. A us-probe run scores each item twice "
          "(real country + US swap), so double it for those.")


if __name__ == "__main__":
    main()
