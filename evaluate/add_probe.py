"""Add a cluster-representative probe to an existing NormAd or BLEnD prediction file.

Loads an existing _usprobe.json (which already has pred/scores for the real
country) and runs only the probe-country scoring pass, writing a new file with
us_pred/us_scores filled in for the probe country.

This uses half the GPU memory compared to running eval_normad/blend.py from scratch,
because the model runs only once per example (probe pass only).

Usage (NormAd):
    python evaluate/add_probe.py \
        --base outputs/behavioral/normad_sftdpo_aya_cult_8b_nfs_mpw_usprobe.json \
        --probe-country iran --condition sftdpo_aya_cult --model-size 8b

Usage (BLEnD):
    python evaluate/add_probe.py \
        --base outputs/behavioral/blend_base_8b_nfs_usprobe.json \
        --probe-country taiwan --condition base --model-size 8b
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from tqdm import tqdm


def _in_sample(key, frac: float) -> bool:
    """Deterministic per-example membership in a subsample of size `frac`.

    Hashing the example id rather than slicing a shuffled list keeps the subset
    identical across probe countries, conditions and model sizes, which is what
    makes the agreement rates comparable. It needs no global state, so a resumed
    task picks the same examples.

    BLEnD is 23,032 examples per country against NormAd's 1,697, so a full probe
    runs ~37 min per country at the observed ~10 it/s and a 7-country grid is
    ~4h per condition. The probe metric is an agreement rate; at n=3000 its
    standard error is ~0.9 points, well below the 5-10 point differences being
    compared, so sampling is ~8x faster at negligible cost to precision.

    Unsampled examples keep us_pred=None, which every consumer already skips —
    the same path taken when the probe country is the example's own country.
    """
    import hashlib
    if frac >= 1.0:
        return True
    h = hashlib.md5(str(key).encode()).hexdigest()[:8]
    return int(h, 16) / 0x100000000 < frac

from evaluate._common import (
    NEUTRAL_CHAT_SYSTEM,
    PROJECT_ROOT,
    build_chat_prompt,
    is_instruct,
    load_model_for_eval,
    resolve_condition,
)
from evaluate.country_names import resolve_probe_country

BEHAVIORAL = PROJECT_ROOT / "outputs" / "behavioral"


def run_normad_probe(base_data: dict, probe: str, model, tokenizer,
                     instruct: bool, data_path: Path,
                     base_name: str = "",
                     system_prompt: str | None = None) -> list[dict]:
    from evaluate.eval_normad import (
        NEUTRAL_SHOTS_NORMAD, YN_WORD_CHOICES, YN_WORD_PROMPTS,
        YN_PROMPT_TEMPLATE, add_context_line,
        build_fewshot_prefix, build_neutral_fewshot_prefix, country,
        gold_label, load_normad, scenario_text, score_choices,
        HOLDOUT_COUNTRIES,
    )
    ds = load_normad(data_path)
    probe = resolve_probe_country(probe, {country(ex) for ex in ds})
    # The base run's flags are not recorded in its JSON, so they are read off
    # its filename. Hardcoding multi_prompt_word=True here meant us_scores was
    # a sum over 4 templates while the base pred came from one, so pred and
    # us_pred were never on the same scale -- the thing a default rate
    # compares. Same for the few-shot source: _nfs is the neutral pair, _fs<N>
    # is N examples drawn from the holdout pool, and they are different prompts.
    mpw = "_mpw" in base_name
    neutral_fs = "_nfs" in base_name
    if mpw:
        prefix = build_neutral_fewshot_prefix(multi_prompt_word=True)
    elif neutral_fs:
        prefix = build_neutral_fewshot_prefix(multi_prompt_word=False)
    else:
        m = re.search(r"_fs(\d+)", base_name)
        if m:
            # Same seed as the base run, and its shots come from the holdout
            # pool, which is already excluded below -- so rebuilding the prefix
            # here does not change which rows are iterated and the zip against
            # base_data["predictions"] stays aligned.
            prefix, _fs_excluded, _fs_turns = build_fewshot_prefix(
                ds, int(m.group(1)), yn_only=True, multi_prompt_word=False)
        else:
            # No _fs/_nfs/_mpw marker means a zero-shot base run, which
            # eval_normad gives an empty prefix.
            prefix = ""
    fewshot_turns = NEUTRAL_SHOTS_NORMAD if instruct else None
    choices = YN_WORD_CHOICES
    holdout_excluded = {i for i, ex in enumerate(ds) if country(ex) in HOLDOUT_COUNTRIES}

    pred_iter = iter(base_data["predictions"])
    new_predictions = []
    leading_space = not instruct

    for i, ex in enumerate(tqdm(ds, desc=f"add_probe(normad)/{probe}")):
        if i in holdout_excluded:
            continue
        if gold_label(ex) == "neutral":
            continue
        existing = next(pred_iter)
        c = country(ex)

        us_pred = us_raw_scores = None
        if c != probe:
            if mpw:
                acc = [0.0, 0.0]
                for tmpl, pfx in zip(YN_WORD_PROMPTS, prefix):
                    if instruct:
                        up = build_chat_prompt(tokenizer, tmpl.format(country=probe, scenario=scenario_text(ex)), fewshot=fewshot_turns, system=system_prompt)
                    else:
                        up = pfx + tmpl.format(country=probe, scenario=scenario_text(ex))
                    s = score_choices(model, tokenizer, up, choices, leading_space=leading_space)
                    acc[0] += s[0]; acc[1] += s[1]
                us_pred = "yes" if acc[0] > acc[1] else "no"
                us_raw_scores = acc
            else:
                # The baseline template itself, not a copy of it. This path used
                # YN_WORD_PROMPTS[0] while a --yn-only base run scored
                # YN_PROMPT_TEMPLATE, so pred and us_pred came from different
                # prompts and their agreement measured the template gap.
                tmpl = add_context_line(YN_PROMPT_TEMPLATE, "none")
                if instruct:
                    up = build_chat_prompt(tokenizer, tmpl.format(country=probe, scenario=scenario_text(ex)), fewshot=fewshot_turns, system=system_prompt)
                else:
                    up = prefix + tmpl.format(country=probe, scenario=scenario_text(ex))
                s = score_choices(model, tokenizer, up, choices, leading_space=leading_space)
                us_pred = "yes" if s[0] > s[1] else "no"
                us_raw_scores = list(s)

        new_predictions.append({
            "country": existing["country"], "group": existing["group"],
            "gold": existing["gold"], "pred": existing["pred"],
            "us_pred": us_pred, "scores": existing.get("scores"),
            "us_scores": us_raw_scores,
        })
    return new_predictions, probe


def run_blend_probe(base_data: dict, probe: str, model, tokenizer,
                    instruct: bool, data_path: Path,
                    sample_frac: float = 1.0,
                    system_prompt: str | None = None) -> list[dict]:
    from evaluate.eval_blend import (
        NEUTRAL_SHOTS_BLEND, build_neutral_fewshot_prefix, load_blend,
        score_choices, us_probe_prompt, SCORING_SUFFIX,
    )
    ds = load_blend(data_path)
    probe = resolve_probe_country(probe, {ex["country"] for ex in ds
                                          if ex.get("country")})
    prefix = build_neutral_fewshot_prefix(multi_prompt=False)
    fewshot_turns = NEUTRAL_SHOTS_BLEND if instruct else None

    pred_iter = iter(base_data["predictions"])
    new_predictions = []

    for ex in tqdm(ds, desc=f"add_probe(blend)/{probe}"):
        existing = next(pred_iter)
        c = ex["country"]

        us_pred = None
        if c != probe and _in_sample(ex.get("MCQID", ex.get("mcqid")), sample_frac):
            prompt = ex["prompt"]
            probe_prompt = us_probe_prompt(prompt, c, probe=probe)
            if instruct:
                up = build_chat_prompt(tokenizer, probe_prompt, fewshot=fewshot_turns, generation_suffix=SCORING_SUFFIX, system=system_prompt)
            else:
                up = prefix + probe_prompt + SCORING_SUFFIX
            us_scores = score_choices(model, tokenizer, up)
            from evaluate.eval_blend import CHOICES
            us_pred = CHOICES[max(range(4), key=us_scores.__getitem__)]

        new_predictions.append({
            "country": existing["country"], "group": existing["group"],
            "gold": existing["gold"], "pred": existing["pred"],
            "us_pred": us_pred, "mcqid": existing.get("mcqid"),
            "scores": existing.get("scores"),
        })
    return new_predictions, probe


def run_culturalbench_probe(base_data: dict, probe: str, model, tokenizer,
                            instruct: bool,
                            neutral_shots: int | None = 2,
                            system_prompt: str | None = None) -> list[dict]:
    from evaluate.eval_culturalbench import (
        CHOICES, build_neutral_fewshot_prefix, build_neutral_shots,
        build_prompt, make_probe_prompt, DATA_PATH,
    )
    import json
    rows = json.loads(DATA_PATH.read_text())
    probe = resolve_probe_country(probe, {r["country"] for r in rows
                                          if r.get("country")})
    # Built exactly as evaluate_one builds it, from the shot count the base run
    # used. This called build_neutral_fewshot_prefix() with no argument, which
    # falls back to the 2-shot 1-yes/1-no NEUTRAL_SHOTS -- so an --neutral-shots 4
    # baseline (4 shots, 1 yes / 3 no) was compared against a probe that saw a
    # different context and a different implied yes-rate.
    # None means the base run had no few-shot prefix at all, which is not the
    # same as "unknown" -- a zero-shot baseline must be probed zero-shot.
    if neutral_shots is None:
        prefix = ""
        fewshot_turns = None
        print("Neutral few-shot: none (zero-shot base run)")
    else:
        usable = [r for r in rows if r.get("reformatted_prompt")]
        gold_yes_rate = sum(bool(r["answer"]) for r in usable) / len(usable)
        shots = build_neutral_shots(neutral_shots, gold_yes_rate)
        prefix = build_neutral_fewshot_prefix(shots)
        fewshot_turns = shots if instruct else None
        n_yes = sum(a == "yes" for _, a in shots)
        print(f"Neutral few-shot: {len(shots)} shots, {n_yes} yes / "
              f"{len(shots) - n_yes} no (gold yes-rate {gold_yes_rate:.3f})")
    leading_space = not instruct

    from evaluate.eval_normad import score_choices
    pred_iter = iter(base_data["predictions"])
    new_predictions = []

    for row in tqdm(rows, desc=f"add_probe(culturalbench)/{probe}"):
        if not row.get("reformatted_prompt"):
            continue
        existing = next(pred_iter)
        country = row["country"]

        us_pred = us_raw_scores = None
        if country.lower().replace(" ", "_") != probe.lower().replace(" ", "_"):
            probe_q = make_probe_prompt(row["reformatted_prompt"], country, probe)
            probe_prompt = build_prompt(prefix, probe_q, instruct, tokenizer, fewshot_turns,
                                        system_prompt=system_prompt)
            us_scores = score_choices(model, tokenizer, probe_prompt, CHOICES, leading_space=leading_space)
            us_pred = CHOICES[0] if us_scores[0] > us_scores[1] else CHOICES[1]
            us_raw_scores = list(us_scores)

        new_predictions.append({
            "data_idx":     existing["data_idx"],
            "question_idx": existing["question_idx"],
            "country":      existing["country"],
            "group":        existing["group"],
            "gold":         existing["gold"],
            "pred":         existing["pred"],
            "us_pred":      us_pred,
            "scores":       existing.get("scores"),
            "us_scores":    us_raw_scores,
        })
    return new_predictions, probe


def run_culturalbench_normy_probe(base_data: dict, probe: str, model, tokenizer,
                                  instruct: bool,
                                  system_prompt: str | None = None) -> list[dict]:
    from evaluate.eval_culturalbench_normy import DATA_PATH
    from evaluate.eval_normad import (
        NEUTRAL_SHOTS_NORMAD, YN_WORD_CHOICES, YN_WORD_PROMPTS,
        build_neutral_fewshot_prefix, score_choices,
    )
    import json
    rows = json.loads(DATA_PATH.read_text())
    probe = resolve_probe_country(probe, {r["country"] for r in rows
                                          if r.get("country")})
    prefix = build_neutral_fewshot_prefix(multi_prompt_word=True)
    fewshot_turns = NEUTRAL_SHOTS_NORMAD if instruct else None
    choices = YN_WORD_CHOICES
    leading_space = not instruct

    pred_iter = iter(base_data["predictions"])
    new_predictions = []

    for row in tqdm(rows, desc=f"add_probe(culturalbench_normy)/{probe}"):
        story = row.get("prompt")
        if not story:
            continue
        existing = next(pred_iter)
        country = row["country"]

        us_pred = us_raw_scores = None
        if country.lower().replace(" ", "_") != probe.lower().replace(" ", "_"):
            acc = [0.0, 0.0]
            for tmpl, pfx in zip(YN_WORD_PROMPTS, prefix):
                if instruct:
                    up = build_chat_prompt(tokenizer, tmpl.format(country=probe, scenario=story), fewshot=fewshot_turns, system=system_prompt)
                else:
                    up = pfx + tmpl.format(country=probe, scenario=story)
                s = score_choices(model, tokenizer, up, choices, leading_space=leading_space)
                acc[0] += s[0]; acc[1] += s[1]
            us_pred = choices[0] if acc[0] > acc[1] else choices[1]
            us_raw_scores = acc

        new_predictions.append({
            "data_idx":     existing["data_idx"],
            "question_idx": existing["question_idx"],
            "country":      existing["country"],
            "group":        existing["group"],
            "gold":         existing["gold"],
            "pred":         existing["pred"],
            "us_pred":      us_pred,
            "scores":       existing.get("scores"),
            "us_scores":    us_raw_scores,
        })
    return new_predictions, probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True,
                        help="Baseline run to read pred/scores from: the eval "
                             "output with the TRUE country in the prompt and no "
                             "probe. A legacy _usprobe.json also works, since "
                             "it carries the same baseline pred. Pass 'us' to "
                             "--probe-country to score the US like any other "
                             "country.")
    parser.add_argument("--probe-country", required=True, nargs="+", metavar="COUNTRY",
                        help="One or more country names to substitute as probe (e.g. 'iran'). "
                             "Several share a single model load, which is most of the cost "
                             "for a one-pass job like this.")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model-size", default="8b")
    parser.add_argument("--data-path", default=None,
                        help="Dataset path. Default: data/normad or data/BLEnD based on base filename.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--precision", default="matched_bf16")
    parser.add_argument(
        "--raw-prompt", action="store_true",
        help="Bypass the chat template on an instruct checkpoint, matching the "
             "way our aya conditions and tulu3_* are scored. Must match how the "
             "--base file was produced, which is why a _raw tail on that file is "
             "carried into the output name rather than stripped.")
    parser.add_argument(
        "--sample-frac", type=float, default=1.0, metavar="F",
        help="Score only a deterministic fraction F of examples (BLEnD only). "
             "1.0 scores everything. BLEnD's 23k examples per country make a full "
             "7-country grid ~4h per condition; F=0.13 (~3000 examples) cuts that "
             "to ~30min with ~0.9 points of standard error on the agreement rate. "
             "The subset is chosen by hashing the example id, so it is identical "
             "across countries and conditions. Unsampled examples keep "
             "us_pred=None and are skipped by consumers. The chosen fraction is "
             "recorded as probe_sample_frac in the output.",
    )
    parser.add_argument(
        "--neutral-shots", type=int, default=None, metavar="N",
        help="CulturalBench: the --neutral-shots the base run used, so the "
             "probe prefix is the same context. Read from the base filename's "
             "_nfs<N> tail when omitted; a base whose name carries no shot "
             "count is an error rather than a silent default, because the "
             "wrong count silently changes the number of shots AND the "
             "yes-rate the prefix implies.")
    parser.add_argument(
        "--system-prompt", default=None, metavar="TEXT",
        help="Explicit system message for chat-templated checkpoints, which "
             "MUST match the --base run's: without it a template inserts its "
             "own default (Olmo-3-Instruct's is a function-calling prompt) on "
             "the probe side only, so probe and baseline differ by a system "
             "message as well as the country. Pass 'neutral' for "
             "'You are a helpful assistant.'. No effect under --raw-prompt.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip countries whose output file is already present, so a "
                             "preempted and requeued job resumes instead of redoing work.")
    args = parser.parse_args()

    if args.system_prompt == "neutral":
        args.system_prompt = NEUTRAL_CHAT_SYSTEM
    # A _sys tail on the base file means that run carried a system prompt, so
    # probing it without one would reintroduce the mismatch this flag exists to
    # prevent. out_for() preserves the tail, so the output would claim _sys too.
    if "_sys" in Path(args.base).name and not args.system_prompt:
        parser.error(
            f"{Path(args.base).name} was run with a system prompt; pass the "
            f"same --system-prompt (e.g. 'neutral') or the probe prompts will "
            f"differ from the baseline by more than the country")

    if args.out and len(args.probe_country) > 1:
        parser.error("--out names a single file; pass one --probe-country with it")

    base_path = Path(args.base)
    base_data = json.loads(base_path.read_text())

    if base_path.name.startswith("blend_"):
        benchmark = "blend"
    elif base_path.name.startswith("culturalbench_normy_"):
        benchmark = "culturalbench_normy"
    elif base_path.name.startswith("culturalbench_"):
        benchmark = "culturalbench"
    else:
        benchmark = "normad"

    def out_for(probe: str) -> Path:
        if args.out:
            return Path(args.out)
        slug = probe.lower().replace(" ", "_")
        tail = "_usprobe" if slug in ("us", "usa", "united_states") else f"_{slug}probe"
        # _raw and _sys say how the prompt was built, so they have to survive the
        # rename: dropping them would name a raw-prompted probe file as if it were
        # chat-templated and let the two collide at one path. Any other tail
        # (_batch_calibrated and friends) is stripped as before.
        #
        # The base may be a plain baseline file (no probe at all) or a legacy
        # _usprobe file that doubled as one. Both are handled: the probe tail
        # replaces an existing one if present, and is inserted before
        # _raw/_sys otherwise. "us" keeps the historical _usprobe spelling so
        # the analysis paths and every existing file still line up.
        m = re.search(r"_usprobe((?:_raw|_sys)*)", base_path.name)
        if m:
            return BEHAVIORAL / f"{base_path.name[:m.start()]}{tail}{m.group(1)}.json"
        stem = base_path.name[:-len(".json")] if base_path.name.endswith(".json") \
            else base_path.name
        m = re.search(r"((?:_raw|_sys)*)$", stem)
        return BEHAVIORAL / f"{stem[:m.start()]}{tail}{m.group(1)}.json"

    # CulturalBench only: the probe prefix has to be the base run's prefix, and
    # the base run's flags are not recorded in its JSON. Prefer the explicit
    # flag, fall back to the filename, and refuse to guess.
    neutral_shots = args.neutral_shots
    if benchmark == "culturalbench" and neutral_shots is None:
        m = re.search(r"_nfs(\d+)", base_path.name)
        if m:
            neutral_shots = int(m.group(1))
        elif "_nfs" in base_path.name:
            neutral_shots = 2      # bare _nfs predates the count in the name
        # else: no _nfs tail at all, so the base run was zero-shot and
        # neutral_shots stays None, which the runner reads as "no prefix".
    if benchmark == "culturalbench":
        src = "--neutral-shots" if args.neutral_shots is not None \
            else base_path.name
        print(f"Neutral shots : "
              f"{'none (zero-shot)' if neutral_shots is None else neutral_shots}"
              f" (from {src})")

    if args.data_path:
        data_path = Path(args.data_path)
    elif benchmark == "blend":
        data_path = PROJECT_ROOT / "data" / "BLEnD"
    else:
        data_path = PROJECT_ROOT / "data" / "normad"

    todo = []
    for probe in args.probe_country:
        out_path = out_for(probe)
        if args.skip_existing and out_path.exists():
            print(f"Skipping  : {probe} — {out_path.name} already present")
            continue
        todo.append((probe, out_path))

    print(f"Benchmark : {benchmark}")
    print(f"Base file : {base_path.name}  ({len(base_data['predictions'])} predictions)")
    print(f"Probes    : {', '.join(p for p, _ in todo) or '(none)'}")
    if not todo:
        print("Nothing to do.")
        return

    # One model load for every probe country, not one per country.
    cond = resolve_condition(args.condition, model_size=args.model_size)
    tokenizer, model = load_model_for_eval(cond, precision=args.precision)
    instruct = is_instruct(args.model_size) and not args.raw_prompt

    for probe, out_path in todo:
        print(f"--- probe={probe} -> {out_path.name}", flush=True)
        if benchmark == "blend":
            new_predictions, resolved = run_blend_probe(
                base_data, probe, model, tokenizer, instruct, data_path,
                sample_frac=args.sample_frac, system_prompt=args.system_prompt)
        elif benchmark == "culturalbench_normy":
            new_predictions, resolved = run_culturalbench_normy_probe(
                base_data, probe, model, tokenizer, instruct,
                system_prompt=args.system_prompt)
        elif benchmark == "culturalbench":
            new_predictions, resolved = run_culturalbench_probe(
                base_data, probe, model, tokenizer, instruct,
                neutral_shots=neutral_shots, system_prompt=args.system_prompt)
        else:
            new_predictions, resolved = run_normad_probe(
                base_data, probe, model, tokenizer, instruct, data_path,
                base_name=base_path.name, system_prompt=args.system_prompt)

        out_data = {k: v for k, v in base_data.items()
                    if k != "derived_from"}
        out_data["predictions"] = new_predictions
        out_data["probe_country"] = resolved
        # Recorded so a sampled file is never mistaken for a full one: with
        # --sample-frac < 1 most us_pred are None by design, and a consumer that
        # did not skip them would read that as disagreement.
        out_data["probe_sample_frac"] = args.sample_frac
        out_data["probe_n_scored"] = sum(
            1 for x in new_predictions if x.get("us_pred") is not None)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling temp file and rename, so a preempted or killed task
        # cannot leave a truncated JSON at out_path. write_text() truncates
        # first, which on a requeued scavenger job would destroy the previous
        # result before writing the new one. os.replace is atomic within a
        # filesystem, so out_path is always either the old file or the new one.
        tmp_path = out_path.with_name(out_path.name + f".tmp{os.getpid()}")
        tmp_path.write_text(json.dumps(out_data, indent=2))
        os.replace(tmp_path, out_path)
        print(f"Wrote {out_path}  ({len(new_predictions)} predictions, "
              f"substituted {resolved!r})", flush=True)


if __name__ == "__main__":
    main()
