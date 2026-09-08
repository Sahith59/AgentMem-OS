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

## The 800-character photocopier (found by the n=500 autopsy)

The most recent entry in this ledger, and the fourth harness defect that
made us look worse than we are. Our benchmark loader capped every
conversation turn at 800 characters when building the local dataset cache
(a leftover from an earlier fix that raised the cap from 300 instead of
removing it). Measured against the raw benchmark source: **42.8% of all
turns exceed 800 characters** and lost their tails. Several questions ask
about details that sit past the cut, deep inside long assistant messages.
Our system searched, honestly answered "not mentioned", and was scored
wrong for it: the answer had been deleted by our own tooling before the
memory system ever saw it.

How it was found: a 35-question autopsy of the first full-set run rebuilt
every failed question's context packet through the exact evaluation code
path, probed for the specific evidence each question needs, and traced 11
of 12 single-session-assistant failures upstream, past retrieval, past
the database, into the cache itself. The fix delivers the benchmark
verbatim, and a $0 end-to-end check confirmed all 12 previously-destroyed
details now reach the packet. Re-runs on the fixed harness are in
progress; all previously published numbers remain labelled as measured on
the truncated harness until then.

What it bought, beyond the score: the strongest evidence yet for this
page's core claim. Every harness defect we have found made us look
*worse*, and none of them could have been found without ceiling tests,
per-question artifacts, and autopsies that refuse to stop at the first
plausible explanation.

## The overstuffed suitcase (what fixing the photocopier revealed)

Fixing the 800-character truncation uncovered a second, subtler defect —
this one in our own retrieval, and invisible for the entire truncated
era. With turns delivered in full, one retrieved turn can cost 3,000 to
17,000 characters, so a fixed context budget suddenly held **half** the
distinct sessions it used to (measured: ~10 sessions per packet fell to
~5.4 at a constant ~36k chars). Categories that need *breadth* — did the
value change in a later session? which session holds the user's
preference? — paid immediately: knowledge-update fell from 87.2% to
80.8% on the honest harness, and preference questions' evidence-session
coverage fell from 93% to 67%. The truncation had been quietly
subsidizing breadth by amputating depth.

Two things worth recording about the diagnosis. First, the dual-stack
audit rebuilt all 108 affected packets on *both* harnesses and proved
the newly shipped update-resolution feature innocent — zero of its
annotations appeared in any broken packet; suspicion of the newest code
is a reflex, evidence is a discipline. Second, an earlier probe had
concluded "the packing lever is exhausted" — and re-reading it before
acting showed it swept *neighbor count* on truncated turns, where turn
*length* could not yet be a cost. The lever that was exhausted and the
lever that was broken were different levers wearing the same name.

The fix (snippet packing): turns longer than 800 characters contribute
their query-relevant region, elisions marked, disable switch shipped,
cap swept at $0 before adoption. Knowledge-update recovered to 87.2%
exactly; assistant-recall kept its 94.6%. Cost of the whole episode:
about $6.70 of smokes — which caught, diagnosed, and verified a
regression that would otherwise have landed unexplained inside a $27
headline run.

## The prediction that missed by five points (and what fixed it)

Before the first full-evidence run of all 500 questions we predicted
78 to 81 percent, reasoning from category smoke tests that had gained
+27 questions. The run landed at 73.2. The prediction error is on the
record: two of the three smokes had been run before the snippet change
existed, so we were adding apples to a forecast about oranges. The
forensics then split the real damage three ways: snippet elisions had
cut answer sentences out of top-ranked evidence (assistant-recall) and
countable items out of counting questions (multi-session); a further
"protect the top hits" variant restored only 3 of 8 lost answers while
collapsing breadth from 9.4 to 6.0 sessions per packet — packing
levers trade, they do not add, now proven at a second operating point;
and the remainder was answerer non-determinism, proven by
knowledge-update packets that were byte-identical between smoke and
run yet scored differently.

The fix that survived measurement is F-19, query-adaptive packing:
each question is routed by embedding similarity (no keyword rules) to
the packing its intent needs — lookback questions get deep, intact
passages; aggregate questions get breadth. Verified on rebuilt packets,
then paid: assistant-recall 96.4% (54/56), breadth guard intact. The
answerer-column measurement that followed (same memory, same judge,
gpt-5.6-luna) moved the full-set number to 80.0% ± 0.5 pooled — and the
gains landed almost exactly where the autopsies said reasoning, not
memory, was the binding constraint.

## Four probes, one wall: the counting tail is not a memory problem

Across the whole campaign, 46 questions were wrong in every run we ever
made, on every corpus, under both answerer models. Most are
multi-session or temporal counting questions, and 21 of them had their
complete gold evidence sitting in the assembled packet. Four
independent attempts to fix them from the memory side:

1. **Span-width and breadth-then-depth packing** (2026-08): traded
   coverage for depth, never added.
2. **Aggregation routing** (2026-08): failed its pre-registered bar,
   shipped opt-in only.
3. **Structured answerers v1/v2** (2026-08): 57.3% and 65.3% against a
   ~79% free-reasoning baseline. Forcing reasoning through a schema
   loses more than the arithmetic it buys.
4. **F-22 cross-session activity ledgers** (2026-08-17): a local model
   read each scope's facts, question-blind, and authored recurring
   topic ledgers ("Concerts: 7 distinct instances, dates ..."), which
   were injected as their own packet section. Result on the 43 chronic
   counting failures: **8 correct against ~10 expected from
   flip-history alone.** Refuted. The forensics are the interesting
   part: in the 20 cases where a ledger contained the gold count, the
   answerer used it 4 times and ignored it 16 times, and it was mostly
   right to ignore it. Question-blind ledgers cannot know a question's
   window ("in March", "in the past two weeks", "before making an
   offer"), so an unscoped tally is a distractor, not an answer. Our
   own prototype metric had also been too loose: it scored a hit
   whenever *any* ledger's count equalled the gold number, which
   inflated the go-signal that justified the test.

The convergent conclusion, now paid for four times: **this tail is
bound by the answerer's set construction under question constraints,
not by memory delivery.** The one intervention that ever moved it was
a stronger answerer (+13 multi-session questions from GPT-4o to
GPT-5.6 Luna). We publish this because "we tried four times and the
memory layer was never the problem" is a result, and because it tells
anyone building in this space where not to spend their month.

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
