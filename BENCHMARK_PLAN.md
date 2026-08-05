# Benchmark Plan — Getting to a Publishable, Defensible Baseline

**Status: AWAITING FOUNDER APPROVAL. Nothing below runs without an explicit go.**
Written 2026-08-04. This file is the single source of truth for what we run, in what
order, at what cost, and what each result would mean. Read this before spending money.

Companion docs: `COMPETITIVE_ANALYSIS.md` (what competitors actually published and how
much of it survives scrutiny), `RUNNING_NOTES.md` (day-to-day state).

---

## The rule this plan is built on

**Every paid run must be preceded by a free or cheap check that tells us whether the
paid run is worth doing.** This is not caution for its own sake — it is the discipline
that has already paid for itself three times:

- The oracle-ceiling diagnostic found the harness scoring *above its own ceiling*,
  which exposed three real bugs and moved LongMemEval 46.7% → 76.7% for $0.
- A free evidence-coverage check killed a dense-retrieval upgrade that measured worse,
  saving a paid run that would have produced a worse number.
- An audit of what competitors publish found our own LoCoMo loader was excluding the
  841 easiest questions, making every past LoCoMo number non-comparable.

Nothing here is "run it and see."

---

## Where we stand right now (all measured, all committed)

| | Result |
|---|---|
| LongMemEval **oracle**, n=30, our harness | AgentMem OS **76.7%** · Letta 66.7 · Mem0 56.7 · LangMem 36.7 · floor 33.3 · ceiling 83.3 |
| LoCoMo, n=30, **old 699-question subset** | AgentMem OS 30.0 · Mem0 26.7 · Letta 6.7 · LangMem 6.7 · floor 0.0 — **not comparable to anything published**, superseded |
| Per-category gaps (ours vs ceiling) | knowledge-update **0.00 vs 1.00** · multi-session 0.67 vs 1.00 · temporal 0.67 vs 0.83 |

**Why these are not yet publishable as competitive claims:** they are on the `oracle`
split, and every vendor number is on `_s` or unnamed. The benchmark's own authors
measure GPT-4o at **0.924 oracle vs 0.640 `_s`** — a 28-point gap on identical
questions. Comparing across splits is the single most common error in this field and we
will not make it.

---

## The plan, in order

### GATE 0 — Oracle ceiling on `_s` *(FIRST ATTEMPT FAILED — see below; ~$3.50 to redo)*

**Result of the first attempt (2026-08-04): the gate worked.** 4 of 150 questions
completed; **146 died on OpenAI 429 rate limits**, and the harness printed a confident
`ORACLE CEILING 0.750` computed from the 4 survivors without flagging that 97% of the
run had failed. Two real defects, both now fixed and free to fix:

1. **No retry logic in any of the three harnesses.** This account's gpt-4o limit is
   **30,000 TPM** (gpt-4o-mini is 200,000). A ~6k-token answerer call × 4 workers bursts
   past it immediately. X-MemoryArch hit the identical wall and solved it with backoff;
   we ported its prompt but not its retry. Now all three use exponential backoff
   (8 attempts) and auto-throttle to 2 workers for any ≤30k-TPM model.
2. **Silent partial results.** A degraded run reported a score as if complete. Runs now
   print `!! INCOMPLETE RUN: n/attempted` and record `"complete": false` in the result
   file, so a partial run can never be mistaken for a finished one.

**This is the second time a result that looked valid wasn't** — the first was scoring
above our own ceiling. It is also why the earlier gpt-4o oracle number ("0.500, 11/22")
was unreliable and should be discarded: 8 of its 30 questions had silently failed.

Re-run command (unchanged):
`benchmarks/oracle_ceiling_eval.py --dataset longmemeval --lme-split s --n 150 --gen-model gpt-4o`

`benchmarks/oracle_ceiling_eval.py --dataset longmemeval --lme-split s --n 150 --gen-model gpt-4o`

Retrieval OFF, gold sessions handed to the answerer. Establishes the maximum score
anything can reach on `_s` with this answerer and judge.

**Decision rule:**
- **Ceiling ≥ 80%** → the benchmark is winnable; any gap below it is our retrieval to
  fix. Proceed to Phase 1.
- **Ceiling 65–80%** → normal; proceed to Phase 1 with realistic expectations.
- **Ceiling < 65%** → the answerer/judge is the binding constraint, not memory.
  **STOP.** Do not spend on the 500-question runs; fix the answer layer first, because
  no retrieval work can move a number that is already at its ceiling.

### PHASE 1 — The comparable number *(~$12.50, needs approval)*

The one run that lets us stand next to Zep, Supermemory, Mastra and Honcho honestly.

| Run | Config | Cost |
|---|---|---|
| 1a | LongMemEval `_s`, **n=150**, GPT-4o answerer, GPT-4o judge | ~$3.50 |
| 1b | *(only if 1a is sane)* same at **n=500** | ~$9 |

**Checkpoint between 1a and 1b:** if 1a lands within ~10 points of the Gate-0 ceiling,
retrieval is healthy and the full run is worth paying for. If it lands far below, stop
and diagnose per-category first — a 500-question run of a broken configuration is $9
spent to measure the same bug 500 times.

⚠️ **Practical, non-money cost:** `_s` has 19,195 sessions to ingest locally. Free, but
slow — expect hours. Must be `nohup`'d (see RUNNING_NOTES: a session exit already
killed one long run).

### PHASE 2 — The cheap-model column *(~$1, needs approval)*

Same `_s` run with **gpt-4o-mini**. Two reasons this is not optional padding:
1. Zep publishes a GPT-4o-mini row (63.8%); this makes us directly comparable to it.
2. Reporting two models is what the honest actors do (Supermemory, Mastra, Hindsight all
   publish 3+ model columns). It shows how much of a score is the memory system versus
   the answerer — Mastra gains **+10.6 points** from a model swap alone.

### PHASE 3 — LoCoMo on the corrected pool *(~$3.50, needs approval)*

`--dataset locomo`, categories **1–4 = 1,540 questions**, gpt-4o-mini (matching Mem0's
and Letta's published setup). Replaces the superseded 699-question numbers and makes us
comparable to Mem0 (66.88 paper / 92.5 platform) and Letta (74.0 filesystem).

### PHASE 4 — Publish *(free)*

Rewrite the README results section as **two clearly separated tables**:
- *Same-harness table* — every system run by us, identically, with the ceiling.
- *Published-numbers table* — each vendor's own claim, each carrying its split, its
  answerer model, its judge, and whether it was self-reported.

Never merge those two tables. That merge is the field's original sin.

---

## Improvements deliberately NOT in this plan yet

Evidence-backed, but they change the system while we are trying to measure it. Measure
first, then improve, then re-measure — one variable at a time.

| Lever | Measured value | Why deferred |
|---|---|---|
| Fact-augmented retrieval keys | +9.4% recall, +5.4% accuracy (LongMemEval authors) | We only hold extracted facts for the 940 oracle sessions; `_s` has 19,195. Extraction would cost real money. |
| Time-aware query expansion | +11.3% recall (LongMemEval authors) | ~$0.50/run; add after the baseline exists |
| Knowledge-update retrieval fix | ours 0.00 vs ceiling 1.00 | **Highest-value engineering target.** It's the category our Temporal KG supersession was built for and it is not firing. Fix, then re-measure. |
| Multi-session aggregation | ours 0.67 vs ceiling 1.00 | Retrieval doesn't gather across sessions |

---

## Budget

| Item | Cost |
|---|---|
| Gate 0 (running) | ~$3.50 |
| Phase 1a | ~$3.50 |
| Phase 1b | ~$9 |
| Phase 2 | ~$1 |
| Phase 3 | ~$3.50 |
| **Total if every gate passes** | **~$20.50** |
| Spent to date on this whole arc | ~$8 |

Re-running competitors inside our harness on `_s` would add **$60–90** (Mem0 and LangMem
must LLM-extract 19,195 sessions each). **Not planned** — we already have the
same-harness comparison on the oracle split, and their published numbers exist.

---

## What "solid baseline" means, so we know when we're done

1. A LongMemEval `_s` number, on the standard split, with the answerer and judge named.
2. A LoCoMo number on the standard 1,540-question pool.
3. Both with the oracle ceiling published beside them.
4. Both with per-question outputs committed and a fixed seed.
5. Two separate tables — ours-measured vs vendor-published — never merged.

Landing in the **55–70%** band on `_s` is a good, honest result: near Zep (71.2%) and
well above bare full-context GPT-4o (60.2%). It is not 94%, and 94% is not a number
anyone has reproduced outside the vendor that published it.

**Only after all five hold does the Sarvam / cross-lingual track begin.** The
differentiator needs a credible foundation under it, or it reads as a feature on top of
numbers nobody respects.
