#!/usr/bin/env python3
"""Print the top next tokens at the position the scorer reads, per model.

score_choices() takes the log-prob of "yes"/"no" at the first position after the
prompt. That is only meaningful if the model would in fact put the answer there.
A chat template can leave the prompt somewhere else entirely -- inside a
reasoning block, or between channel markers -- and the scorer cannot tell: it
reads a log-prob either way and the number looks ordinary.

Gemma-4's non-thinking generation prompt ends with an opened-and-closed empty
thought channel, which is ambiguous from the template alone: the answer may
follow directly, or a content-channel marker may come first. This resolves it
empirically. If the top tokens are yes/no variants, the position is right; if
they are channel or newline tokens, every templated score for that model is
read at the wrong place.

  python scripts/check_scoring_position.py --models gemma4_instruct
  python scripts/check_scoring_position.py            # all chat-templated ones
  python scripts/check_scoring_position.py --raw      # the --raw-prompt arm

One model load each, no dataset, a few seconds of GPU per model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate._common import (  # noqa: E402
    NEUTRAL_CHAT_SYSTEM, build_chat_prompt, load_model_for_eval,
    resolve_condition,
)
from evaluate.eval_normad import (  # noqa: E402
    NEUTRAL_SHOTS_NORMAD, YN_PROMPT_TEMPLATE, build_fewshot_prefix, load_normad,
)

SCENARIO = ("Aisha's elderly neighbour knocks on her door at dinner time. "
            "She invites him in and serves him a plate before eating herself.")

DEFAULT_MODELS = ["8b_instruct", "gemma4_instruct", "olmo3_instruct",
                  "qwen35_instruct"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="Comma list of model_size labels.")
    ap.add_argument("--condition", default=None,
                    help="Override the condition; default is per-model.")
    ap.add_argument("--precision", default="matched_bf16")
    ap.add_argument("--raw", action="store_true",
                    help="Score the flat prompt instead of the chat template, "
                         "matching the --raw-prompt arm.")
    ap.add_argument("--shots", type=int, default=2, metavar="N",
                    help="Few-shot demonstrations, matching the eval's "
                         "--few-shot. 0 for zero-shot. This matters: without a "
                         "prefix the flat prompt does not elicit an answer at "
                         "all, so a 0-shot reading says nothing about an fs2 run.")
    ap.add_argument("--data-path", default=str(PROJECT_ROOT / "data" / "normad"))
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    body = YN_PROMPT_TEMPLATE.format(country="Hungary", scenario=SCENARIO)

    # Built from the same call the eval makes, so the shots are the ones the
    # real run demonstrates -- including their lowercase yes/no answers, which
    # is the thing that may or may not fix the casing.
    prefix = ""
    fewshot_turns = None
    if args.shots > 0:
        ds = load_normad(Path(args.data_path))
        prefix, _excluded, turns = build_fewshot_prefix(
            ds, args.shots, yn_only=True, multi_prompt_word=False)
        fewshot_turns = turns

    for label in [m.strip() for m in args.models.split(",") if m.strip()]:
        cond_name = args.condition or (
            "base" if label in ("8b_instruct", "gemma4_instruct") else "offshelf")
        cond = resolve_condition(cond_name, model_size=label)
        tokenizer, model = load_model_for_eval(cond, precision=args.precision)

        if args.raw:
            prompt = prefix + body
        else:
            prompt = build_chat_prompt(tokenizer, body, fewshot=fewshot_turns,
                                       system=NEUTRAL_CHAT_SYSTEM)

        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        top = torch.topk(logprobs, args.top)

        print("=" * 72)
        print(f"{label}  condition={cond_name}  "
              f"{'raw' if args.raw else 'chat-templated'}  shots={args.shots}")
        print(f"prompt tail: {prompt[-90:]!r}")
        for lp, tid in zip(top.values.tolist(), top.indices.tolist()):
            print(f"   {tokenizer.decode([tid])!r:24} {lp:+.3f}")

        # What the scorer would actually compare, for reference. leading_space
        # follows the eval scripts: True on the flat prompt, False under a chat
        # template, where the answer starts right after the turn opener.
        pre = " " if args.raw else ""
        for word in ("yes", "no"):
            ids = tokenizer.encode(pre + word, add_special_tokens=False)
            print(f"   scored {pre + word!r} -> first token "
                  f"{tokenizer.decode([ids[0]])!r} lp={logprobs[ids[0]]:+.3f}")

        # Case variants, to size the surface-form gap: if the scored spelling is
        # far below the aggregate, the comparison is happening in the tail and a
        # case-insensitive scorer would read a different answer.
        for word, variants in (("yes", ("yes", "Yes", "YES")),
                               ("no", ("no", "No", "NO"))):
            lps = []
            for v in variants:
                ids = tokenizer.encode(pre + v, add_special_tokens=False)
                lps.append(logprobs[ids[0]])
            agg = torch.logsumexp(torch.stack(lps), dim=0)
            scored = lps[0]
            print(f"   {word!r}: scored {scored:+.3f}  "
                  f"logsumexp over {list(variants)} {agg:+.3f}  "
                  f"(missing {agg - scored:+.3f})")

        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
