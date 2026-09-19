# Paper Writing Prompt — PlurVA-LLM Workshop

**Task:** Write a 4-page conference paper (not including references) in LaTeX answering the three research questions below. Follow ACL/EMNLP short-paper style (two-column, 11pt). Use `\includegraphics` to reference figures by the filenames listed. Write actual prose — no placeholder text.

---

**Title:** *Post-Training Alignment and Cultural Bias: How SFT and DPO Affect LLM Performance on Culturally Grounded Benchmarks*

---

**Venue & framing:** This paper is submitted to the First Workshop on Pluralistic Value Alignment of LLMs (PlurVA-LLM). Frame the paper accordingly throughout — the central argument is that standard alignment procedures (SFT, DPO) are not culturally neutral: they encode and amplify a Western/US-centric value prior at the expense of pluralistic representation. The introduction should engage directly with the workshop's concern that value alignment research has overwhelmingly assumed a monolithic set of "human values," and position this work as an empirical audit of whether post-training pushes models toward or away from pluralistic cultural competence. The discussion should close by arguing that culturally diverse training data alone (the Aya-cult condition) is insufficient — the alignment *process* itself needs to be redesigned to support pluralism. Avoid framing cultural bias purely as a fairness or safety issue; the workshop audience is specifically interested in value pluralism as a first-class alignment objective.

---

**Research Questions:**
1. How do post-training processes (SFT and DPO) affect LLM performance on culturally grounded benchmarks?
2. Is there a performance gap between Western/US-centric and Non-Western questions? Does post-training affect this gap?
3. When the model gets Non-Western questions wrong, does it default to US-centric answers?

---

**Methods:**
- **Models:** LLaMA 3.1 8B (primary), Gemma 4 12B (replication). Five conditions: Base, SFT-cult, SFT+DPO-cult, SFT-nocult, SFT+DPO-nocult. SFT uses the Aya dataset; the cultural vs. non-cultural split tests whether cultural training data specifically drives the effects.
- **Benchmarks:**
  - *NormAd*: cultural norm adherence, yes/no questions. Countries grouped into Western (US, UK, Germany, Australia, etc.) and Non-Western (Japan, India, Egypt, Brazil, etc.). Evaluated with multi-prompt averaging and batch calibration.
  - *BLEnD*: 4-choice MCQ about everyday cultural practices across 16 countries. Same Western/Non-Western grouping.
  - *NLU controls*: BoolQ and CommonsenseQA, to verify alignment does not degrade general language understanding.
- **US-probe:** For each non-US example, the question is re-scored with the country name replaced by "US" to obtain the model's US-counterfactual prediction. **US-default rate** = `P(pred = us_pred | pred ≠ gold)` — the fraction of errors where the model predicted the same answer it would have given for the US version.

---

**Key Results:**

*RQ1 — Post-training effects:*
- NormAd (LLaMA 8B): Base 0.839 → SFT-cult 0.822 → SFT+DPO-cult 0.789. SFT-nocult 0.834 → SFT+DPO-nocult 0.815. Post-training degrades cultural norm accuracy, with DPO amplifying the drop.
- BLEnD (LLaMA 8B): Base 0.630 → modest monotonic degradation across all conditions (range 0.593–0.605). Effect is smaller but consistent.
- NLU controls largely preserved: BoolQ stable (0.848 base, 0.838–0.854 range); CSQA slight drop (0.731 base → 0.690–0.695). General NLU is not the source of cultural degradation.
- Gemma 4 12B replicates the NormAd degradation (base 0.866 → 0.775–0.833 post-training).

*RQ2 — Western vs. Non-Western gap:*
- NormAd 8B: W=0.890, NW=0.828 at base (gap=6.2pp). Gap persists or widens post-training: SFT+DPO-cult W=0.849, NW=0.763 (gap=8.6pp). SFT-nocult slightly narrows it (NW=0.840). Cultural training data does not preferentially help Non-Western performance.
- BLEnD 8B: W=0.699, NW=0.628 at base (gap=7.1pp). Gap is stable across all conditions (6.5–7.2pp range). Post-training neither closes nor widens it systematically.
- Gemma 4: Same pattern. One exception: SFT+DPO-cult shows W=0.769, NW=0.800 — the only observed inversion, driven by Western degradation rather than Non-Western improvement.

*RQ3 — US-default on errors:*
- NormAd NW error US-match: base 0.643 → rises to 0.782–0.832 under SFT/SFT+DPO-cult, drops to 0.594 under SFT+DPO-nocult. Cultural alignment paradoxically increases US-defaulting on errors. Caveat: ~83% of NormAd gold answers are US-aligned, so correct answers and US answers largely coincide — errors by definition disagree with both gold and US, limiting interpretability.
- BLEnD NW error US-match: 0.563 (base), range 0.516–0.563 across conditions. Elevated above chance (0.25 for 4-choice) and above the correct-answer US-match baseline (~0.38), indicating a genuine directional bias. BLEnD's more culturally diverse gold labels make this the cleaner signal.
- Interpretation: the model's errors on Non-Western questions are disproportionately US-aligned, particularly on BLEnD where this cannot be explained by dataset skew.

---

**Figures to include:**

| Filename | Caption content |
|---|---|
| `aya_cult_8b_normad_accuracy.pdf` | NormAd accuracy across alignment pipeline, Western vs. Non-Western (LLaMA 8B) |
| `aya_cult_8b_blend_accuracy.pdf` | BLEnD accuracy across pipeline, Western vs. Non-Western (LLaMA 8B) |
| `aya_cult_8b_nlu_accuracy.pdf` | BoolQ, CSQA, and BLEnD overall accuracy across pipeline — NLU and cultural MCQ controls (three-panel) |
| `aya_cult_8b_normad_us_default_rate_nw.pdf` | US-default rate on NW errors across pipeline (NormAd) |
| `aya_cult_8b_blend_us_default_rate_nw.pdf` | US-default rate on NW errors across pipeline (BLEnD) |

Place figures where they are most relevant. Keep captions concise and self-contained. You may omit figures if the page limit is tight — prioritize the accuracy and US-default figures.

---

**Tables to include — render as blank tables with label, caption, and column headers; leave all data cells as `---`:**

**Table 1 — Main accuracy results** (`\label{tab:main-results}`)
Caption: "Accuracy across alignment conditions and benchmarks. W = Western, NW = Non-Western. Bold = best per column."
Columns: Condition | NormAd (W) | NormAd (NW) | NormAd (Overall) | BLEnD (W) | BLEnD (NW) | BLEnD (Overall)
Rows: Base | SFT-cult | SFT+DPO-cult | SFT-nocult | SFT+DPO-nocult
Include a horizontal rule between Base and the SFT rows. Repeat for LLaMA 8B and Gemma 4 12B as separate panels with a model header row.

**Table 2 — NLU control results** (`\label{tab:nlu}`)
Caption: "BoolQ and CommonsenseQA (CSQA) accuracy across alignment conditions (LLaMA 8B). See also Figure~\ref{fig:nlu} for trends across the pipeline. NLU performance is largely preserved, ruling out general language degradation as the cause of cultural benchmark drops."
Columns: Condition | BoolQ | CSQA
Rows: Base | SFT-cult | SFT+DPO-cult | SFT-nocult | SFT+DPO-nocult

**Table 3 — US-default rate on Non-Western errors** (`\label{tab:us-default}`)
Caption: "Fraction of Non-Western errors where the model's prediction matches its US-counterfactual prediction. Chance baseline is 0.50 for NormAd (binary) and 0.25 for BLEnD (4-choice). NW-correct = US-match rate among correct predictions, serving as a baseline for the model's natural US alignment independent of being wrong."
Columns: Condition | NormAd NW-err | NormAd NW-correct | BLEnD NW-err | BLEnD NW-correct
Rows: Base | SFT-cult | SFT+DPO-cult | SFT-nocult | SFT+DPO-nocult
Include separate panels for LLaMA 8B and Gemma 4 12B.

---

**Structure:**
- **Abstract** (~100 words)
- **Introduction** — open with the premise that alignment research has implicitly assumed a monocultural "human values" standard; motivate culturally pluralistic evaluation; state three RQs
- **Experimental Setup** — models, benchmarks, US-probe methodology
- **Results** — three subsections, one per RQ; reference figures and tables inline
- **Discussion** — address the NormAd gold-label confound; contrast with BLEnD as a cleaner signal; argue that culturally diverse training data is necessary but not sufficient, and that the alignment process itself must be redesigned to support value pluralism
- **Conclusion** (~100 words)
- References (not counted toward page limit)

**Constraints:** 4 pages max excluding references. Tight, precise academic prose. Report exact numbers where cited. No hedging filler.
