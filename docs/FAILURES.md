# What Failed, What It Cost, and What Each Failure Bought

Nobody in the agent-memory space publishes their negative results. This page
is ours: every approach we tried that did not work, what it cost, and the
lesson each one paid for. We keep it public for two reasons. First, negative
results are results: several of these refutations are as informative as the
wins. Second, you can calibrate how much to trust our positive numbers by
seeing how hard we tried to break them.

Total evaluation API spend across the whole English-baseline campaign:
**≈ $100.** Extraction and infrastructure: **$0** (local models, and a
university SLURM cluster for the big parallel jobs). Every paid run is
logged against what it bought.

---

## The one that changed everything: our own extraction was destroying answers

The single most important (and most painful) measurement of the project. A
run configured to answer from extracted fact summaries instead of verbatim
conversation turns scored **0.287, below the no-memory baseline of 0.602.**

It came with a natural control group: 22 questions whose sessions were stored
identically in both runs scored identically (11/22 in both), while 128
questions whose storage form changed collapsed from 68.8% to 25.0%. Same
answerer, same judge, same questions. The only variable was storage form.
**Replacing verbatim turns with lossy summaries cost 43.8 accuracy points.**
A concrete instance: for "How many playlists do I have?", the extracted fact
kept the topic and dropped the number.

What it bought: the architecture rule that extraction must augment verbatim
evidence, never replace it, plus an answer-survival preflight that now
refuses to start any paid run if the stored form loses more than 20% of
recoverable answers. This finding also explains why several earlier
"improvements" measured as zero: they were tuned on top of evidence the
storage layer had already destroyed.

## Harness bugs that made us look worse than we were

Ceiling-testing our own harness (hand the answerer perfect evidence and see
what it scores) exposed three defects in our loader: the benchmark's
per-question reference dates were silently dropped, per-session dates were
dropped, and sessions were truncated. Every "how many days ago" question,
27% of the dataset, was unanswerable by construction. Fixing the loader
moved the measured ceiling from 46.7% to 83.3% with zero changes to the
memory system.

Later, two more: the eval recorded the memory-source argument while the code
silently stored something else, and runs shared one mutating database so
they were not reproducible. Both fixed with config-scoped databases and
write-time verification of what was actually stored.

The lesson, stated as a rule we now operate by: **harness defects moved our
numbers more than most architecture changes did.** If a benchmark harness
has never been ceiling-tested, its numbers are unvalidated.

## Retrieval ideas that measured as noise or worse

Each of these was implemented, measured under a pre-registered bar, and
refuted. They are listed so nobody (including us) rebuilds them.

| Idea | Result | Verdict |
|---|---|---|
| Pseudo-relevance feedback (query expansion from top hits) | Coverage unchanged on systematic failures | Refuted |
| Session round-robin packing | Traded wins for losses, net zero | Refuted |
| Date-boosted retrieval for temporal questions | No coverage gain | Refuted |
| Dropping the raw-turn tail entirely | 75.3% vs 76.0 baseline | Refuted |
| Breadth-then-depth retrieval as default | 76.7%, inside noise; useful shape, not a default | Ships opt-in only |
| Aggregation-intent routing as default | Passed its gate 9/10, then failed its probe bar | Ships opt-in only |
| A sharper evidence-coverage proxy metric | Predicted worse than the metric it replaced | Discarded, kept as a record |

## The structured answerer: two designs, two honest failures

The systematic failures cluster in set construction and date arithmetic, and
the research literature (PAL, TReMu, Test of Time) says code should do that
arithmetic. We agreed, and built it. Version 1 (code does set membership and
window filtering): **57.3% against a 76.0% baseline.** Version 2 (model
selects the final set, code only computes): **65.3%.** Root causes were
autopsied per question: v1 gave code the semantic judgment work our own
earlier experiments had already proven code cannot do; v2's unit
verbalization misfired on formats the gold answers wanted bare.

Both versions were killed under their pre-registered bars. The default
answer path was never touched. What it bought: a documented negative result
that the obvious "LLM enumerates, code computes" architecture loses to free
reasoning on this benchmark, plus eight unit-tested deterministic components
(date windows, dedup, unit selection) that survive for future use.

## The noise floor discovery (why we stopped chasing 2-point wins)

Running a byte-identical configuration twice: totals moved by 1 question,
but **13 of 150 individual answers flipped.** GPT-4o at temperature 0 is not
deterministic. Consequences we now operate by: single-run deltas under ~5
questions are unattributable, per-category swings of ±2 are weather, and an
evening was once spent chasing a temporal-reasoning "regression" that this
measurement later proved was noise. All published numbers are now means over
at least 3 runs with spread.

## Predictions we got wrong (kept on the record)

We pre-register a predicted range before every paid run. The tally so far:
**7 wrong, 1 right.** The 7 misses are all logged with what we believed and
why it was wrong. The single hit was the 40k-context run (predicted 117 to
122 correct, got 120), and it was the first prediction derived from a
measured mechanism instead of hope. That is the difference between the two.

## Infrastructure failures worth knowing about

- **Graphiti ingestion, 21+ hours, killed.** The same 30-question haystack
  every other system ingested in minutes. Their per-message LLM extraction
  is the bottleneck. A parallelized re-run is planned; their accuracy cell
  stays empty until it exists.
- **A $0.29 unauthorized spend.** A broad `pytest` sweep collected a test
  file that makes real API calls at collection time. It is now permanently
  excluded from broad sweeps. Small number, important habit: every dollar is
  authorized before it is spent, or it is an incident.
- **An early proxy benchmark was retired entirely.** Simulation-based
  competitor scripts were quarantined into `benchmarks/deprecated_proxy_sim/`
  and are cited nowhere. Real libraries or nothing.

---

*The full internal log behind this page runs to several hundred entries with
per-run artifacts. This page is the distilled, public account. If you want a
specific number's full provenance, open an issue and we will publish the
trail for it.*
