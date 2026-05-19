---
name: Multicultural LLM Probe — Project Overview
description: Course research project comparing SFT vs DPO effects on cultural norms in Llama 3.2 3B; extends CULNIG neuron analysis to NormAd/values
type: project
---

The repo is the implementation for a course research paper on cultural norm drift in LLMs.

**Research question:** Does SFT vs. DPO differentially erode non-Western cultural norms in LLMs, and can this be explained at the neuron level via a values-based extension of CULNIG?

**Three contributions:**
1. Behavioral: SFT vs. DPO comparison on cultural norms across Western/non-Western cultures.
2. Methodological: extend CULNIG neuron identification from cultural knowledge (BLEnD) to norms/values (NormAd); construct NormAdctrl control dataset.
3. Mechanistic: neuron-level analysis of how alignment training shifts culture-neuron attribution across SFT, DPO, and SFT+DPO.

**Setup:** Llama 3.2 3B, QLoRA 4-bit (r=16, bf16), 4 saved target modules (mlp.gate_proj + self_attn.{q,k,v}_proj). UMIACS Nexus class partition, RTX A5000 24GB. C4 (SFT+DPO) loads in bf16 instead of 4-bit because `merge_and_unload()` doesn't compose with bitsandbytes — fits comfortably in 24GB anyway. Class partition allocations: see show_qos for QoS-specific caps; we use `high` for SFT/DPO/SFT+DPO and `medium` for eval/CULNIG.

**Four conditions:**
- **C1 `base`** — Llama 3.2 3B base, no fine-tuning.
- **C2 `sft`** — base + SFT on `tatsu-lab/alpaca`, 3 epochs.
- **C3 `dpo`** — base + DPO on HH-RLHF, 2 epochs.
- **C4 `sftdpo`** — C2 LoRA merged into base, then DPO on HH-RLHF, 2 epochs.

C3 and the DPO stage of C4 share the same HH-RLHF subsample (same 30K, same seed), so contrasts within {C3, C4} isolate the effect of SFT initialization on subsequent DPO.

**Why:** Course project; pilot for later conference submission scaling to 70B/MoE.

**How to apply:** Code runs on Nexus class partition. Keep CULNIG core gradient scoring logic unchanged — only swap dataset loading. SFT+DPO job has a SLURM `--dependency=afterok:$SFT_ID` so it auto-runs after SFT completes. Analysis scripts default to the four-condition list defined in `env.sh`'s `$CONDITIONS`.
