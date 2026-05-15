---
name: C4 precision tradeoff — bf16 merge vs stacked adapters
description: We accepted that C4 (SFT+DPO) doesn't literally reproduce the C2 SFT model semantics; merge to bf16 introduces a small precision tax. Document as a limitation in the paper.
type: project
---

C4 (SFT+DPO) is trained and evaluated by loading base in bf16, applying the C2 SFT adapter, calling `merge_and_unload()`, then training/applying a fresh DPO LoRA. This is **not** the same model the C2 condition runs against at eval time, which uses base in 4-bit + LoRA adapter applied inline (no merge).

**The discrepancy:**
- C2 eval forward = `bf16(dequant(NF4(base))) + LoRA`
- C4 SFT-precondition = `bf16(merge(bf16(base), LoRA))`
- Different precision regime + different math (inline application vs. merged matrices).
- Empirical magnitude: ~0.1-1% perplexity-equivalent — probably below the W vs NW gap signal but not zero.

**Why we accepted it (option C in the May 15 design discussion):**
- The clean fix (option D, stacked adapters) requires either a custom DPOTrainer subclass or doubled GPU memory for an explicit ref_model. Engineering risk vs. value tradeoff didn't pencil out for a course paper pilot.
- Option A (merge → save → re-quantize via /tmp) was viable but still has its own precision discrepancy in the merge step itself, so it only partially fixes the issue.

**Why:** Course project; pilot for a later conference paper where option D would be implemented properly.

**How to apply:**
- In the paper's Limitations section, explicitly state: "C4's SFT precondition is a bf16-merged approximation of C2, not the exact C2 model. Cross-condition contrasts involving C4 may include a small precision-tax confound (estimated <1% perplexity)."
- Treat the C3 vs C4 contrast as the most reliable single comparison for the disentanglement story (both go through analogous DPO training; precision differs only in the C4 base, not C3 base).
- For the conference scale-up, switch to option D: load base in 4-bit, stack frozen SFT adapter + trainable DPO adapter, train with a custom DPOTrainer that calls `model.set_adapter(["sft"])` for reference forwards.
