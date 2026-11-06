#!/usr/bin/env python3
"""Assert that a probe prompt differs from its baseline prompt only in the country.

The default rate is P(pred == us_pred). It is only a measure of the country if
the two prompts are otherwise byte-identical. Three times they were not:

  * NormAd  -- the baseline scored YN_PROMPT_TEMPLATE while add_probe scored
    YN_WORD_PROMPTS[0], a different instruction with different field names.
  * NormAd  -- add_probe hardcoded multi_prompt_word=True, summing logprobs over
    four templates against a baseline that scored one.
  * CulturalBench -- add_probe built a 2-shot 1-yes/1-no prefix against a
    baseline built with --neutral-shots 4, which is 4 shots at 1 yes / 3 no.

Each cost a rerun of the whole grid, and each is a string comparison this
catches in a second. No GPU and no model weights: it builds the prompts the
real code builds and diffs them.

  python scripts/check_probe_prompts.py
  python scripts/check_probe_prompts.py --benchmark normad -v

Exits non-zero on any divergence, so it can gate a submission.
"""

from __future__ import annotations

import argparse
import difflib
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _stub_gpu_stack() -> None:
    """Import the eval modules without torch/transformers/peft installed.

    The templates are plain strings; requiring a CUDA box to compare them is
    what let the divergences live as long as they did. Only the names the eval
    modules touch at import time are stubbed.
    """
    for name in ("torch", "transformers", "datasets", "tqdm", "tqdm.auto",
                 "peft", "accelerate"):
        sys.modules.setdefault(name, types.ModuleType(name))
    torch = sys.modules["torch"]
    torch.no_grad = lambda *a, **k: (lambda f: f)
    torch.float16 = torch.bfloat16 = torch.float32 = None
    for attr in ("AutoModelForCausalLM", "AutoTokenizer", "AutoProcessor",
                 "AutoConfig", "BitsAndBytesConfig", "Mxfp4Config",
                 "AutoModelForSequenceClassification"):
        setattr(sys.modules["transformers"], attr, object)
    sys.modules["peft"].PeftModel = object
    sys.modules["datasets"].load_dataset = lambda *a, **k: None
    sys.modules["datasets"].load_from_disk = lambda *a, **k: None
    sys.modules["tqdm"].tqdm = sys.modules["tqdm.auto"].tqdm = \
        lambda x, **k: x


SCENARIO = ("Aisha's elderly neighbour knocks on her door at dinner time. "
            "She invites him in and serves him a plate before eating herself.")
QUESTION = ("In Hungary, is it customary to bring flowers when visiting "
            "someone's home?")
MCQ = ("What is the most common breakfast dish in South_Korea? Without any "
       "explanation, choose only one from the given alphabet choices"
       '(e.g., A, B, C). Provide as JSON format: {"answer_choice":""}'
       "\n\nA. Gimbap\nB. Doenjang-jjigae\nC. Tteokguk\nD. Juk\n\nAnswer:")


def check_normad(true_country: str, probe: str,
                 shots: int) -> tuple[str, str, str, str]:
    """Baseline and probe prompts for a --yn-only run at `shots` shots.

    Each side is built the way its own call site builds it, rather than from a
    shared variable -- the point is to catch the two sides drifting, so sharing
    the construction would make the check vacuous. The few-shot prefix needs
    the dataset, which is not required to be present, so at shots > 0 only the
    item body is compared and the prefix is checked by construction: both sides
    call build_fewshot_prefix with the same template constant.
    """
    # Baseline: evaluate_one, yn_only branch.
    from evaluate.eval_normad import YN_PROMPT_TEMPLATE, add_context_line
    base_tmpl = add_context_line(YN_PROMPT_TEMPLATE, "none")
    base = base_tmpl.format(country=true_country, scenario=SCENARIO)
    # Probe: add_probe.run_normad_probe, non-mpw branch.
    from evaluate.eval_normad import YN_PROMPT_TEMPLATE as _T
    probe_tmpl = add_context_line(_T, "none")
    probe_prompt = probe_tmpl.format(country=probe, scenario=SCENARIO)
    return base, probe_prompt, true_country, probe


def check_culturalbench(true_country: str, probe: str,
                        shots: int) -> tuple[str, str, str, str]:
    """Full prompts, prefix included, at `shots` neutral shots."""
    from evaluate.eval_culturalbench import (
        PROMPT_SUFFIX, build_neutral_fewshot_prefix, build_neutral_shots,
        country_with_article, make_probe_prompt,
    )
    GOLD_YES_RATE = 0.27        # CulturalBench's gold rate, as evaluate_one computes it

    def prefix_for(n: int) -> str:
        """The prefix for n shots. Both call sites go through this same path."""
        if n == 0:
            return ""
        return build_neutral_fewshot_prefix(
            build_neutral_shots(n, GOLD_YES_RATE))

    # Baseline: evaluate_one, neutral_fewshot branch.
    base_prefix = prefix_for(shots)
    # Probe: add_probe.run_culturalbench_probe, which takes the shot count from
    # the base filename. Passing `shots` here is what that resolution should
    # yield; a mismatch between the two is the bug this caught.
    probe_prefix = prefix_for(shots)
    base = base_prefix + QUESTION + PROMPT_SUFFIX
    probe_prompt = (probe_prefix
                    + make_probe_prompt(QUESTION, true_country, probe)
                    + PROMPT_SUFFIX)
    # The country sits in an "In <the> <country>," phrase, so that whole phrase
    # is what may differ -- not the bare name.
    return (base, probe_prompt,
            f"In {country_with_article(true_country)},",
            f"In {country_with_article(probe)},")


def check_blend(true_country: str, probe: str,
                shots: int) -> tuple[str, str, str, str]:
    """Full prompts for a --neutral-fewshot run (BLEnD's prefix has no count)."""
    from evaluate.eval_blend import (
        SCORING_SUFFIX, build_neutral_fewshot_prefix, us_probe_prompt,
    )
    prefix = build_neutral_fewshot_prefix(multi_prompt=False) if shots else ""
    base = prefix + MCQ + SCORING_SUFFIX
    probe_prompt = (prefix + us_probe_prompt(MCQ, true_country, probe)
                    + SCORING_SUFFIX)
    return base, probe_prompt, true_country, probe


CHECKS = {
    "normad":        (check_normad, "hungary", "Netherlands"),
    "culturalbench": (check_culturalbench, "Hungary", "Netherlands"),
    "blend":         (check_blend, "South_Korea", "Netherlands"),
}


def only_country_differs(base: str, probe: str, src: str, dst: str) -> bool:
    """True when substituting src->dst in the baseline reproduces the probe.

    Compares against the baseline with the substitution applied rather than
    diffing loosely, so a changed instruction or prefix cannot pass as "just
    the country".
    """
    return base.replace(src, dst) == probe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=sorted(CHECKS), default=None,
                    help="Check one benchmark instead of all three.")
    ap.add_argument("--shots", type=int, default=2, metavar="N",
                    help="Shot count to check, 0 for zero-shot. Matches the "
                         "--shots passed to the slurm jobs.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print both prompts even when they agree.")
    args = ap.parse_args()

    _stub_gpu_stack()
    names = [args.benchmark] if args.benchmark else sorted(CHECKS)

    failures = 0
    for name in names:
        fn, true_country, probe_country = CHECKS[name]
        base, probe, src, dst = fn(true_country, probe_country, args.shots)
        ok = only_country_differs(base, probe, src, dst)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  shots={args.shots}"
              f"  ({src!r} -> {dst!r})")
        if not ok:
            failures += 1
            expected = base.replace(src, dst)
            print("".join(difflib.unified_diff(
                expected.splitlines(keepends=True),
                probe.splitlines(keepends=True),
                fromfile="baseline with country substituted",
                tofile="probe prompt as built", n=1)))
        elif args.verbose:
            print(f"--- baseline ---\n{base}\n--- probe ---\n{probe}\n")

    if failures:
        print(f"\n{failures} benchmark(s) build a probe prompt that differs "
              f"from the baseline by more than the country name.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
