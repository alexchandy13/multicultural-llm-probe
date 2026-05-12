---
name: Multicultural LLM Probe — Project Overview
description: Course research project comparing SFT vs DPO effects on cultural norms in Llama 3.2 3B; extends CULNIG neuron analysis to NormAd/values
type: project
---

The repo is the implementation for a course research paper on cultural norm drift in LLMs.

**Research question:** Does SFT vs. DPO differentially erode non-Western cultural norms in LLMs, and can this be explained at the neuron level via a values-based extension of CULNIG?

**Three contributions:**
1. Behavioral: first SFT vs. DPO comparison on cultural norms across Western/non-Western cultures.
2. Methodological: extend CULNIG neuron identification from cultural knowledge (BLEnD) to norms/values (NormAd); construct NormAdctrl control dataset.
3. Mechanistic: neuron-level analysis of how alignment training suppresses culture neurons.

**Setup:** Llama 3.2 3B, QLoRA 4-bit (r=16, bf16), 7 target modules. UMIACS Nexus SLURM, RTX A5000 24GB, 30GB storage quota.

**Four conditions:** C1 base, C2 SFT (Alpaca 3ep), C3 DPO (HH-RLHF 2ep), C4 Meta Instruct.

**Why:** Course project; pilot for later conference submission scaling to 70B/MoE.

**How to apply:** Code must run on Nexus SLURM with strict 30GB storage. Keep CULNIG core gradient scoring logic unchanged — only swap dataset loading. Always have BLEnD-only fallback path in case NormAd extension breaks.
