# Prompt: Do the 95% CIs change any paper claims?

I'm writing a 4-page paper for the PlurVA-LLM workshop titled "Post-Training Alignment and Cultural Bias: How SFT and DPO Affect LLM Performance on Culturally Grounded Benchmarks." The paper makes several empirical claims. I've now computed 95% bootstrap confidence intervals for all key numbers and want to know whether any claims need to be softened, dropped, or reframed.

For each claim below, the CI is listed next to the point estimate. Please:
1. Identify which claims are **statistically robust** (CIs clearly support the direction).
2. Identify which claims are **marginal or fragile** (CIs overlap or barely separate), and suggest how to reframe them.
3. Flag any claim where the CI **contradicts** the direction of the stated conclusion.
4. Suggest specific revised wording where needed.

---

## Claims and numbers with 95% bootstrap CIs

### RQ1 — Post-training degrades cultural benchmark accuracy

**NormAd (LLaMA 3.1 8B, overall):**
- Base: 0.839 [0.821, 0.856]
- SFT-cult: 0.822 [0.803, 0.839]
- SFT+DPO-cult: 0.789 [0.770, 0.807]
- SFT-nocult: 0.834 [0.817, 0.851]
- SFT+DPO-nocult: 0.815 [0.796, 0.835]

**NormAd (Gemma 4 12B, overall):**
- Base: 0.866 [0.849, 0.881]
- SFT-cult: 0.776 [0.757, 0.795]
- SFT+DPO-cult: 0.775 [0.756, 0.795]
- SFT-nocult: 0.823 [0.806, 0.841]
- SFT+DPO-nocult: 0.833 [0.816, 0.850]

**BLEnD (LLaMA 3.1 8B, overall):**
- Base: 0.630 [0.624, 0.636]
- SFT-cult: 0.604 [0.598, 0.610]
- SFT+DPO-cult: 0.605 [0.599, 0.611]
- SFT-nocult: 0.595 [0.589, 0.602]
- SFT+DPO-nocult: 0.593 [0.587, 0.600]

**BLEnD (Gemma 4 12B, overall):**
- Base: 0.660 [0.654, 0.666]
- SFT-cult: 0.656 [0.650, 0.663]
- SFT+DPO-cult: 0.657 [0.651, 0.664]
- SFT-nocult: 0.648 [0.642, 0.654]
- SFT+DPO-nocult: 0.647 [0.640, 0.652]

---

### RQ2 — Western/US-Centric vs. Non-Western/US-Distant performance gap

**NormAd (LLaMA 3.1 8B):**
- Base: W=0.890 [0.860, 0.919], NW=0.828 [0.802, 0.852] → gap ~6.2pp
- SFT+DPO-cult: W=0.849 [0.809, 0.884], NW=0.763 [0.731, 0.789] → gap ~8.6pp
- SFT-nocult: W=0.866 [0.831, 0.898], NW=0.840 [0.814, 0.865] → gap ~2.6pp

**NormAd (Gemma 4 12B):**
- Base: W=0.911 [0.882, 0.938], NW=0.860 [0.837, 0.882] → gap ~5.1pp
- SFT+DPO-cult: W=0.769 [0.726, 0.812], NW=0.800 [0.771, 0.827] → NW > W (inversion)

**BLEnD (LLaMA 3.1 8B):**
- Base: W=0.699 [0.687, 0.711], NW=0.628 [0.618, 0.638] → gap ~7.1pp
- SFT+DPO-cult: W=0.680 [0.668, 0.693], NW=0.608 [0.599, 0.618] → gap ~7.2pp
- SFT+DPO-nocult: W=0.662 [0.650, 0.675], NW=0.597 [0.587, 0.607] → gap ~6.5pp

**BLEnD (Gemma 4 12B):**
- Base: W=0.733 [0.721, 0.745], NW=0.657 [0.647, 0.666] → gap ~7.6pp
- SFT+DPO-cult: W=0.742 [0.731, 0.754], NW=0.655 [0.645, 0.665] → gap ~8.7pp

---

### RQ3 — NLU controls are preserved

**BoolQ (LLaMA 3.1 8B):**
- Base: 0.839 [0.826, 0.853]
- SFT+DPO-cult: 0.828 [0.815, 0.841]
- SFT+DPO-nocult: 0.815 [0.802, 0.828]

**BoolQ (Gemma 4 12B):**
- Base: 0.835 [0.823, 0.848]
- SFT+DPO-cult: 0.868 [0.856, 0.880]
- SFT+DPO-nocult: 0.866 [0.854, 0.877]

**CSQA (LLaMA 3.1 8B):**
- Base: 0.710 [0.686, 0.734]
- SFT-cult: 0.674 [0.649, 0.699]
- SFT+DPO-cult: 0.678 [0.654, 0.704]
- SFT-nocult: 0.659 [0.636, 0.684]
- SFT+DPO-nocult: 0.661 [0.636, 0.686]

**CSQA (Gemma 4 12B):**
- Base: 0.701 [0.677, 0.726]
- SFT+DPO-cult: 0.726 [0.704, 0.750]
- SFT+DPO-nocult: 0.731 [0.706, 0.754]

---

## Context for evaluation

- **NormAd** is binary yes/no; n ≈ 400–500 per group per condition. CIs are wider due to smaller n.
- **BLEnD** is 4-choice MCQ; n ≈ 8,000–10,000 total. CIs are very tight.
- **NLU** (BoolQ ≈ 3270 examples, CSQA ≈ 1221 examples). CSQA CIs are wider.
- The paper's current framing: "post-training degrades cultural benchmark accuracy, with DPO amplifying the drop" (RQ1); "the W/NW gap persists or widens post-training" (RQ2); "NLU controls are largely preserved, ruling out general language degradation" (RQ3).
- The Gemma W/NW inversion at SFT+DPO-cult is currently described as "the only observed inversion, driven by Western degradation rather than Non-Western improvement."

---

### RQ3 — US-default rate among Non-Western errors

US-default rate = P(pred == us_pred | pred ≠ gold, group == Non-Western). Chance baseline is 0.50 for NormAd (binary) and 0.25 for BLEnD (4-choice). n is the number of NW errors available for bootstrapping — NormAd n is small (~120–200), BLEnD n is large (~3100–3700).

**NormAd (LLaMA 3.1 8B):**
- Base: 0.643 [0.566, 0.720]  (n=143)
- SFT-cult: 0.784 [0.719, 0.850]  (n=153)
- SFT+DPO-cult: 0.832 [0.777, 0.883]  (n=197)
- SFT-nocult: 0.782 [0.707, 0.850]  (n=133)
- SFT+DPO-nocult: 0.594 [0.516, 0.671]  (n=155)

**NormAd (Gemma 4 12B):**
- Base: 0.776 [0.698, 0.853]  (n=116)
- SFT-cult: 0.754 [0.693, 0.816]  (n=179)
- SFT+DPO-cult: 0.741 [0.675, 0.801]  (n=166)
- SFT-nocult: 0.667 [0.589, 0.745]  (n=141)
- SFT+DPO-nocult: 0.734 [0.656, 0.812]  (n=128)

**BLEnD (LLaMA 3.1 8B):**
- Base: 0.563 [0.546, 0.580]  (n=3402)
- SFT-cult: 0.563 [0.546, 0.579]  (n=3611)
- SFT+DPO-cult: 0.516 [0.500, 0.533]  (n=3584)
- SFT-nocult: 0.536 [0.519, 0.552]  (n=3717)
- SFT+DPO-nocult: 0.545 [0.529, 0.561]  (n=3681)

**BLEnD (Gemma 4 12B):**
- Base: 0.515 [0.498, 0.533]  (n=3137)
- SFT-cult: 0.542 [0.524, 0.558]  (n=3163)
- SFT+DPO-cult: 0.542 [0.524, 0.559]  (n=3154)
- SFT-nocult: 0.561 [0.545, 0.579]  (n=3172)
- SFT+DPO-nocult: 0.528 [0.511, 0.545]  (n=3236)

Current paper framing for RQ3: "NormAd NW error US-match rises from 0.643 (base) to 0.782–0.832 under SFT/SFT+DPO-cult; BLEnD NW error US-match is 0.563 (base), elevated above chance (0.25) and above the correct-answer US-match baseline (~0.38), indicating a genuine directional bias." The paper also caveats NormAd by noting ~83% of NormAd gold answers are US-aligned, making BLEnD the cleaner signal.

---

Please give specific revised claim language where a claim is fragile or needs qualification.
