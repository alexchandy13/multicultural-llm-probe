---
name: Multicultural LLM Probe — Project Overview
description: Course research project comparing SFT vs DPO effects on cultural norms in Llama 3.2 3B; extends CULNIG neuron analysis to NormAd/values
type: project
---

The repo is the implementation for a course research paper on cultural norm drift in LLMs.

**Research question:** Does SFT vs. DPO differentially erode non-Western cultural norms in LLMs, and can this be explained at the neuron level via a values-based extension of CULNIG?

**Three contributions:**
1. Behavioral: first SFT vs. DPO comparison on cultural norms across Western/non-Western cultures, with the two conditions trained on identical data so any contrast is attributable to the objective.
2. Methodological: extend CULNIG neuron identification from cultural knowledge (BLEnD) to norms/values (NormAd); construct NormAdctrl control dataset.
3. Mechanistic: neuron-level analysis of how alignment training suppresses culture neurons across SFT, DPO, and SFT+DPO.

**Setup:** Llama 3.2 3B, QLoRA 4-bit (r=16, bf16), 7 target modules. UMIACS Nexus class partition, RTX A5000 24GB. C4 (SFT+DPO) loads in bf16 instead of 4-bit because `merge_and_unload()` doesn't compose with bitsandbytes — fits comfortably in 24GB anyway. Class partition allocations: see show_qos for QoS-specific caps; we use `high` for SFT/DPO/SFT+DPO and `medium` for eval/CULNIG.

**Five conditions:** C1 base, C2 SFT (HH-RLHF chosen, 3ep), C3 DPO (HH-RLHF, 2ep), C4 SFT+DPO sequential (same HH-RLHF, 30k subsample shared with C2/C3), C5 Meta Instruct.

**Why HH-RLHF for SFT, not Alpaca:** original plan used Alpaca for SFT and HH-RLHF for DPO, which conflated training method with training data. Switching SFT to HH-RLHF chosen responses (with the same 30k subsample, same seed as DPO) isolates the method effect at the heart of the RQ. C4 was added to make C5 (Meta Instruct) interpretable — without a controlled SFT+DPO, we couldn't separate Meta's proprietary pipeline from the alignment recipe itself.

**Why:** Course project; pilot for later conference submission scaling to 70B/MoE.

**How to apply:** Code runs on Nexus class partition. Keep CULNIG core gradient scoring logic unchanged — only swap dataset loading. Always have BLEnD-only fallback path in case NormAd extension breaks. SFT+DPO job has a SLURM `--dependency=afterok:$SFT_ID` so it auto-runs after SFT completes.
