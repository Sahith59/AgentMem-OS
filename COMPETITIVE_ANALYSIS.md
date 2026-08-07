# Competitive Analysis — Published Numbers vs. Reality

Research date: 2026-08-04. Every claim carries a source. This document exists because
AgentMem OS publishes benchmark numbers, and publishing numbers obliges you to know
exactly what the numbers you are compared against actually mean.

**Executive summary.** The headline scores in this field are not comparable to each
other and mostly not comparable to their own source papers. Three findings, verified
here against primary artifacts, break the standard leaderboard:

1. **The public LoCoMo is not the paper's LoCoMo.** The paper describes 50
   conversations / 7,512 questions; the only released file holds **10 conversations /
   1,986 questions**. Every vendor number is on the subset.
2. **The industry's LoCoMo category labels are swapped**, and everyone inherited it
   from one harness. Verified below by measuring evidence spans directly.
3. **LongMemEval numbers are meaningless without the split.** The benchmark's own
   authors measure GPT-4o at **0.924 on `oracle` vs 0.640 on `_s`** — a 28-point gap
   on identical questions.

---

## 1. What the original benchmark papers actually report

Vendors quote 90%+. The papers themselves report far lower, on different metrics.

### LoCoMo — Maharana et al., ACL 2024 ([arXiv 2402.17753](https://arxiv.org/abs/2402.17753))

Metric is **F1 partial match**, not LLM-judge accuracy.

| System | Overall F1 |
|---|---|
| **Human** | **87.9** |
| Best RAG (observation units, top-5), GPT-3.5-16k | **41.4** |
| GPT-4-turbo, 4K context | 32.1 |
| GPT-3.5-turbo-16K, 16K context | 37.8 |

**The authors' own best system scored 41.4 F1.** Any "90% on LoCoMo" is a different
metric (lenient LLM-judge) on a different subset.

### LongMemEval — Wu et al., ICLR 2025 ([arXiv 2410.10813](https://arxiv.org/abs/2410.10813))

Metric is LLM-judge accuracy, judge `gpt-4o-2024-08-06`, >97% human agreement.

| Setting | GPT-4o | GPT-4o + Chain-of-Note |
|---|---|---|
| **Oracle** (evidence sessions only) | 0.870 | **0.924** |
| **`_s`** (~40–50 sessions, ~115k tokens) | 0.606 | **0.640** |

The authors' own best *memory design* on `_m` reaches **0.714**. There is **no
per-category baseline table in the paper** — every per-category LongMemEval table in
circulation descends from Zep's paper, not the benchmark's authors.

---

## 2. Three verified errors in the standard leaderboard

### 2.1 The public LoCoMo is a subset — verified

`snap-research/locomo`'s only released file, `data/locomo10.json`, contains **10
conversations / 1,986 questions**, not the paper's 50 / 7,512. Paper baselines and
vendor numbers are therefore not comparable in either direction.

### 2.2 Category labels are swapped industry-wide — verified in this repo

The Mem0-derived harness (forked by Memobase, Backboard, Hindsight, Pam, MemMachine)
maps `1→single_hop, 2→temporal, 3→multi_hop, 4→open_domain`. The LoCoMo paper defines
multi-hop as requiring several sessions and single-hop as answerable from one. Measuring
evidence spans directly over the released data:

| Category | n | Mean evidence sessions | % spanning >1 session | Industry label | **Actually** |
|---|---|---|---|---|---|
| 1 | 282 | 2.68 | **95.4%** | single_hop | **multi-hop** |
| 2 | 321 | 1.10 | 8.8% | temporal | temporal ✓ |
| 3 | 96 | 1.75 | 34.8% | multi_hop | open-domain |
| 4 | 841 | 1.00 | **0.1%** | open_domain | **single-hop** |
| 5 | 446 | 1.00 | 0.0% | (excluded) | adversarial ✓ |

Reproduce with `python3 -c` over `locomo10.json`; the numbers above came from that pass,
not from any secondary source. Overall scores are unaffected (same question pool), but
**every per-category LoCoMo claim in that lineage is mislabeled.** Only temporal is right.

### 2.3 The adversarial category is dropped, and the remainder reported as "overall"

1,986 − 446 (adversarial) = **1,540**, the exact count Mem0, Memobase, Backboard and Pam
all report. Adversarial is where the LoCoMo paper's own GPT-3.5-16k baseline scored
**2.1 F1** — the category that punishes hallucination. Removing 22.5% of the questions
mechanically inflates every "overall" figure in that lineage.

**This repo's own past error, for the record:** our loader kept categories **1,2,3 only
= 699 questions**, silently excluding category 4 — the 841 easiest, single-session
questions. Our earlier LoCoMo numbers (AgentMem OS 30.0%, Mem0 26.7%) were therefore
measured on a **substantially harder subset** than any published 1,540-question number
and were never comparable to them. Now fixed: the loader defaults to categories 1–4 and
labels categories by what they actually contain.

---

## 3. What each vendor actually published

### Mem0
- **Paper** ([arXiv 2504.19413](https://arxiv.org/abs/2504.19413), ECAI 2025), GPT-4o-mini
  throughout, judge never named: LoCoMo overall **J = 66.88** (base) / **68.44** (graph).
- **In that same paper, full-context — pasting the whole conversation — scores 72.90**,
  beating both Mem0 variants and every other memory system tested. Not headlined.
- **2026 platform numbers**: LoCoMo **92.5**, LongMemEval **94.4**. Managed platform only
  ("proprietary optimizations not available in the open-source SDK"), **underlying LLM
  never disclosed**, **zero competitor comparisons**, top-200 retrieval budget, and
  Mem0's own docs contradict its own blog on per-category figures while reporting the
  same overall.
- Independent reproductions: MaxiMem **73.8%**, Bench'd (OSS) **32.4%**, this repo
  **56.7%**, a filed reproduction issue **≈0.20**.
- **2026-08-06 (verified live): Mem0's OSS graph module NO LONGER EXISTS.**
  `mem0/memory/graph_memory.py` and the whole typed-relation `mem0/graphs/`
  package were deleted in PR #4805 (v3 pipeline port); the OSS replacement
  is spaCy entity extraction used only for retrieval boosting — no edges,
  no graph. Open issue #6591 documents docs-vs-code drift over the removal.
  The hosted Platform still sells "Graph Memory" (co-occurrence-inferred,
  closed-source). **Consequence for our claims: the "Mem0 graph variant"
  (J=68.44 above) refers to CODE THAT IS GONE from OSS — any comparison we
  publish must say "Mem0 (paper, since-removed OSS graph)" or compare
  against the closed platform, and this is mid-deprecation: RE-VERIFY at
  publish time.**

### Zep / Graphiti ([arXiv 2501.13956](https://arxiv.org/abs/2501.13956))
- **LongMemEval `_s`**: Zep **71.2%** (GPT-4o) vs full-context 60.2%; **63.8%**
  (GPT-4o-mini) vs full-context 55.4%. 1.6k context tokens vs 115k, ~90% lower latency.
- **Zep loses to plain full-context on single-session-assistant** in both configs
  (−9.1%, −17.7%) and on knowledge-update with GPT-4o-mini. The aggregate win comes
  from preference and temporal categories.
- **DMR**: Zep 94.8% vs MemGPT 93.4% — but **full-conversation-in-context scores 94.4%**.
  A 1.4-point gap on 500 questions, no CIs. Zep's own paper calls DMR unfit for purpose.
- **Current best LoCoMo**: **80.32% ± 0.43** (GPT-4o-mini agent+judge, 10 runs, 30/30
  config), Zep-only, no competitors.
- Zep on LoCoMo's quality: "ambiguous questions and inconsistent ground truth."

### Letta / MemGPT
- **MemGPT DMR 93.4%** (GPT-4-turbo) — but **v1 of the same paper reported 82.4%**, and
  the GPT-4 baseline fell from 63.0% to 32.1% between revisions **with no explanation**.
  The whole DMR leaderboard rests on a number that moved 11 points between preprints.
  Judge prompt explicitly says *"be generous with your grading."*
- **Letta's own LoCoMo: 74.0%** using **nothing but a filesystem** — `grep`,
  `search_files`, `open`, `close` — with GPT-4o-mini, beating Mem0's graph variant
  (68.5%). Judge model and category split undisclosed.
- Letta's conclusion, verbatim: *"current memory benchmarks may not be very
  meaningful"*, and they were *"unable to determine a way to backfill LoCoMo data into
  MemGPT/Letta"*, with Mem0 *"did not respond to requests for clarification."*

### Everyone else on LongMemEval `_s` (QA accuracy, for scale)
Hindsight (Gemini-3) 91.4 · Honcho 90.4 · Mastra (gpt-5-mini) 94.9 / (GPT-4o) 84.2 ·
Supermemory 81.6–85.2 · **Zep 71.2** · **full-context GPT-4o 60.2**.
LangChain has published **no LangMem numbers at all**; every LangMem figure in
circulation was produced by its competitor Mem0.

### MemPalace — the cleanest illustration of the metric trap
58k GitHub stars, headline "100% on LongMemEval." Its own BENCHMARKS.md states the
numbers are **Recall@5 retrieval recall, not QA accuracy**, that the 100% was *"tuned on
3 specific wrong answers"*, and that a LoCoMo "100%" is *"structurally guaranteed
(top-k > sessions)."* Held-out honest figure: 98.4% recall.

---

## 4. The benchmarks are the weakest link

[Penfield Labs' LoCoMo audit](https://dev.to/penfieldlabs/we-audited-locomo-64-of-the-answer-key-is-wrong-and-the-judge-accepts-up-to-63-of-intentionally-33lg)
(repo [dial481/locomo-audit](https://github.com/dial481/locomo-audit)):

- **6.4% of the answer key is wrong** (99/1,540) — hallucinated gold answers, wrong
  temporal arithmetic, 24 speaker-attribution errors.
- **The GPT-4o-mini judge accepts 62.81% of deliberately wrong-but-topically-adjacent
  answers**, versus ~11% for specifically-wrong answers — a **6× leniency gap that
  rewards exactly the failure mode of weak retrieval**.
- **Theoretical ceiling ≈ 93.6%.** Mem0's 92.5 and Backboard's 90.0 sit inside the
  benchmark's noise floor.
- **56% of adjacent per-category comparisons are statistically indistinguishable.**

[LoCoMo-Refined](https://github.com/mem-eval-suite/LoCoMo_refined) measured the stock
judge at **43.67% agreement with humans**; a stricter prompt reached 86.33%. Re-scoring
under it: EverMemOS −22.1, MemOS −17.3, MemPalace −15.8, **Mem0 −15.6 points**. A judge
*prompt* change moves scores more than any claimed inter-system gap.

[MemDelta](https://arxiv.org/abs/2606.29914) isolates the confounds: swapping only the
**embedding model** moves LongMemEval **±6.2 points**; the **answerer model family**
spans a **45-point range**; and **Mem0 vs plain cloud-embedding RAG is statistically
indistinguishable (p=1.0)** while Mem0 spends **50× more write-path compute**.

**Aggregate sensitivity: judge prompt ±15–22 pts · answerer model up to 45 pts ·
embedding ±6–11 pts · split choice ~28 pts.** The protocol dominates the system.

---

## 4b. Consolidation-style systems (audited 2026-08-05 — the representation lane we are entering)

Systems that distill raw conversation into higher-level memory, audited at the
representation level (papers read directly, not abstracts):

| System | Distilled unit | Citations to source? | Supersession? | Cross-lingual? | Published score |
|---|---|---|---|---|---|
| **TiMem** (ACL Findings 2026) | hierarchical free-text summaries (turns→summaries→persona tree) | not documented | implicit tree order only | none mentioned | **76.88% LongMemEval `_s`**, 75.30% LoCoMo |
| Nemori (arXiv 2508.03341) | free-text "insights" (key,text) | none described | whole-insight replace | none mentioned | claims SOTA-at-publication on LoCoMo/LME |
| SeCom (ICLR 2025) | topic segments (not facts) | n/a | n/a | none | evidence vs turn/session/summary granularity |
| Generative Agents (UIST 2023) | narrative reflections | no | no | no | believability eval only, no QA |
| Zep/Graphiti | KG edges, bi-temporal valid_at/invalid_at | YES (episode lists, blogged) | YES (invalidate-don't-delete) | none documented | 71.2% `_s` |
| Mem0 (OSS v3) | write-time facts | UNVERIFIED | **regressed to ADD-only** (issues #4896/#4956/#5867 open) | none documented | self-reported 94.4/66.88 paper |
| Letta sleep-time | free-text memory blocks | UNVERIFIED | rethink_memory() rewrite | none documented | — |

**Honest takes we bind ourselves to:**
- **TiMem beats our current 66.0% on the identical `_s` split (76.88%).** It exists,
  it's open source, and any consolidation work we publish gets compared to it by us
  first. Consolidation-as-a-category is established literature (Nemori, SeCom,
  TiMem, Honcho's "dream-time") — we claim no priority on the category.
- What no surveyed system does: structured, individually-dated, individually-cited
  ATOMIC facts (all use narrative text or graph edges); a separate, functioning
  bi-temporal per-attribute profile tier (Zep: same graph substrate; Letta:
  free-text; Mem0: designed it, currently broken in shipped OSS); and write-time
  cross-lingual canonicalization (no publicly documented claimant — re-verified
  against non-English docs before we publish that sentence).
- Never claimable: "first bi-temporal memory" (Zep's paper title), "first with
  provenance" (Zep ships + blogs it), "first per-attribute supersession" (Mem0's
  design intent).

## 5. Where AgentMem OS actually stands

**Current measured position** (LongMemEval **`oracle`**, n=30, seed 42, identical
answerer/judge/top-k for every system, this repo's harness):

| System | QA accuracy |
|---|---|
| **AgentMem OS** | **76.7%** |
| Letta | 66.7% |
| Mem0 | 56.7% |
| LangMem | 36.7% |
| Recent-only floor | 33.3% |
| *Oracle ceiling (retrieval off)* | *83.3%* |

**Honest reading:**
- These are **`oracle`-split** numbers. They are **not** comparable to Zep's 71.2% or
  Mem0's 94.4%, both of which are `_s` or unnamed. The right reference point for an
  oracle number is the paper's own oracle full-context GPT-4o at **0.870**.
- The `_s` run is the milestone that produces a comparable number. The loader now
  supports it (`--lme-split s`, 47.5 sessions/question haystack). **Not yet run.**
- Our LoCoMo numbers were measured on a harder 699-question subset and are being
  re-run on the standard 1,540-question pool.

**What is genuinely defensible today**, and is rarer than any score in this field:
one harness, every system run identically, published per-question outputs, a fixed
seed, an oracle ceiling published beside the results, and an open invitation to
correct our adapter configurations. Note that our Mem0 number (56.7%) is **24 points
more generous** than Bench'd's independent measurement of the same library (32.4%).

**What to chase, from our own per-category diagnostic** (ours vs ceiling): knowledge-update
0.00 vs 1.00 — the category our Temporal KG supersession exists for, and it is not
firing in the benchmark path; multi-session 0.67 vs 1.00. Not the headline average.

**What not to chase.** 92% on LoCoMo is one point under a ceiling set by a broken answer
key, measured by a judge that accepts 63% of wrong answers. Reaching it would mean
optimizing for judge leniency.
