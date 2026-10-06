#!/bin/bash
#SBATCH --job-name=bostest
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --time=0:40:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=legacygpu[00,02-07,09-42]
#SBATCH --output=slurm/bostest.%j.out
#SBATCH --error=slurm/bostest.%j.err

# Diagnose why CULNIG normad scores collapse for gemma4 sftdpo conditions.
#
# Two suspected defects, both in the CULNIG path and absent from the eval path:
#   1. No BOS. dataset.py sets add_special_tokens=False because apply_chat_template
#      succeeded — but calc_neuron_score replaces the template with a pass-through
#      that emits no special tokens, so nothing ever adds BOS.
#   2. Label lookup uses convert_tokens_to_ids('yes'), which wants the literal
#      token 'yes'; after "Answer:" a SentencePiece model predicts '_yes'.
#
# Compares a healthy condition against a broken one so the contrast is visible.

set -euo pipefail
source ./env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python3 - <<'PY'
import torch, torch.nn.functional as F
from culnig.calc_neuron_score import load_model_for_culnig

PROMPT = (
    "You are evaluating cultural acceptability.\n"
    "Country: Japan\n"
    "Scenario: A guest finishes all the food on their plate.\n"
    "Question: Is this action socially acceptable in this culture?\n"
    "Answer with exactly one of: yes, no.\n"
    "Answer:"
)

# BLEnD's shape: the label follows a bare quote, so convert_tokens_to_ids('A')
# is already correct there. What's untested is whether the missing BOS alone
# degrades it — if not, blend needs no rescoring (which would cost ~124 GPU-hours).
BLEND_PROMPT = (
    "Which dish is eaten at New Year in South Korea? Without any explanation, choose "
    "only one from the given alphabet choices(e.g., A, B, C). Provide as JSON format: "
    '{"answer_choice":""}\n\nA. tteokguk\nB. pizza\nC. tacos\nD. sushi\n\nAnswer:'
    '{"answer_choice":"'
)

for cond in ["sft_aya_cult", "sftdpo_aya_cult"]:
    print(f"\n{'='*70}\n{cond} (gemma4)\n{'='*70}", flush=True)
    model, tok = load_model_for_culnig(cond, model_size="gemma4")
    model.eval()

    print(f"convert_tokens_to_ids('yes')={tok.convert_tokens_to_ids('yes')} "
          f"('no')={tok.convert_tokens_to_ids('no')} unk_id={tok.unk_token_id} "
          f"bos_id={tok.bos_token_id}", flush=True)
    for s in ["yes", "no", " yes", " no"]:
        print(f"  encode({s!r}) = {tok.encode(s, add_special_tokens=False)}", flush=True)

    for label, asp in [("no BOS", False), ("with BOS", True)]:
        ids = tok(PROMPT, add_special_tokens=asp).input_ids
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids]).to(model.device)).logits
        pr = F.softmax(logits[0, -1, :].float(), dim=-1)
        # probability CULNIG actually scores
        culnig_yes = pr[tok.convert_tokens_to_ids("yes")].item()
        culnig_no = pr[tok.convert_tokens_to_ids("no")].item()
        # probability the eval path would score
        ev_yes = pr[tok.encode(" yes", add_special_tokens=False)[0]].item()
        ev_no = pr[tok.encode(" no", add_special_tokens=False)[0]].item()
        top = torch.topk(pr, 5)
        print(f"  {label:<9} first_id={ids[0]:<8} "
              f"culnig(yes+no)={culnig_yes + culnig_no:.3e}  "
              f"spaced(yes+no)={ev_yes + ev_no:.3e}", flush=True)
        print(f"            top5={[(tok.decode([i]), round(v.item(), 4)) for i, v in zip(top.indices, top.values)]}",
              flush=True)

    # BLEnD shape: does missing BOS alone hurt the A/B/C/D distribution?
    for label, asp in [("no BOS", False), ("with BOS", True)]:
        ids = tok(BLEND_PROMPT, add_special_tokens=asp).input_ids
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids]).to(model.device)).logits
        pr = F.softmax(logits[0, -1, :].float(), dim=-1)
        tot = sum(pr[tok.convert_tokens_to_ids(c)].item() for c in "ABCD")
        top = torch.topk(pr, 5)
        print(f"  BLEnD {label:<9} P(A..D)={tot:.4f}  "
              f"top5={[(tok.decode([i]), round(v.item(), 3)) for i, v in zip(top.indices, top.values)]}",
              flush=True)

    del model
    torch.cuda.empty_cache()
PY
