
## 2026-08-06 — Run a flagship "proof" test N times before believing it (Critics, Stage 1 R4)
Evidence: the round-3 N1 proof test passed for the team and was written into the build log as
"Proof: 3-OS-process test — mention_count exactly 60". On re-run it failed 6/10 in isolation and
3/5 in the full file. A concurrency test that passes once proves nothing; it must be looped.
**Enforced check:** any test offered as proof of a CONCURRENCY fix must be run >=10x and the pass
rate reported. A single green run is not evidence.

## 2026-08-06 — Optimistic CAS across two transactions is not atomic (Critics, Stage 1 R4)
Evidence: read via autocommit SELECT + write via a new BEGIN leaves a lost-update window; measured
1-3% retry exhaustion on a hot row at only 3-8 writers. **Enforced check:** a compare-and-set must
occur inside ONE write transaction (BEGIN IMMEDIATE) or be expressed as a single SQL statement.

## 2026-08-06 — Coverage percentage without characterization hides the fired branches (Critics, R4)
Evidence: third repeat (R2 M1, R3 N6, now R4-2). "98%" was accurate; the 6 uncovered lines were the
error paths that actually fire. **Enforced check:** every uncovered line must be named and justified
in the build log, and any branch demonstrated to fire under test must be covered.

## 2026-08-06 — A broad `except` that swallows the cause misdiagnoses ops failures (Critics, R4)
Evidence: read-only DB reported as "lost 8 version races", __cause__ and __context__ both None.
**Enforced check:** narrow the exception to the specific recoverable condition, and always
`raise ... from e` when converting an exception at the end of a retry loop.

## 2026-08-06 — A store must never call expire_all() on a session it does not own (Critics, Stage 1 R5)
Evidence: `_reaffirm`'s `session.expire_all()` silently discarded a caller's unflushed edits
(turn.content, fact.extraction_model) in a caller-owned batch on the production session config
(autoflush=False), and the store returned success. A targeted `session.expire(obj)` was measured
to give the identical fresh re-read with no collateral damage. **Enforced check:** in any code
that accepts a caller-supplied session, session-WIDE operations (expire_all, rollback, commit,
close, flush-of-everything) are forbidden on the not-owned path; only object-targeted operations
are allowed. Any `db=`/caller-session API needs at least one test that stages caller state
BEFORE the call and asserts it survives.

## 2026-08-06 — A structural removal must delete the mechanism's VOCABULARY too (Critics, R5)
Evidence: 4th repeat of the stale-doc class (N6-N8, R4-5, now R5-2/R5-4). After the retry/CAS
removal, the rewritten function's own docstring still promised "version-guarded", "retry up to
_REAFFIRM_RETRIES" (constant deleted), and a test-file comment still credited "NullPool" two
rounds after the QueuePool switch. **Enforced check:** when a mechanism is removed, grep the WHOLE
repo for every identifier and every noun that named it (constant, class, pool name, "retry",
"version") and show the grep as evidence in the build log; a rewritten function's docstring is
part of the diff, not documentation debt.

## 2026-08-06 — The evidence doc must record the round that falsified it (Critics, R5)
Evidence: the build log still presents the round-3 optimistic-CAS resolution and its falsified
proof ("mention_count exactly 60") as current, with no round-4 record and stale "98% coverage"
artifacts. **Enforced check:** a fix round is not complete until the log entry that the round
falsified is annotated in place (struck/marked superseded) and the new round's findings +
resolutions + freshly measured artifacts are recorded — the log is the founder's evidence, and a
stale row in it is an over-claim.

## 2026-08-06 — Mutation-test the fix, not just the feature (Critics, Stage 1 R6)
Evidence: two fixes shipped in R5. Swapping `expire(fact)`→`expire_all()` flipped its regression
test red (real tripwire), but deleting the `coalesce(mention_count,1)` left the suite 51/51 GREEN
while a legacy NULL row silently returned to mention_count=None forever. A green suite proves the
fix is present, never that it is DEFENDED. **Enforced check:** for every fix, mutate the fixed line
away and re-run; if the suite stays green, the fix has no tripwire and needs one before the gate.

## 2026-08-06 — Model-level column constraints never reach an existing table (Critics, R6)
Evidence: `mention_count` gained `nullable=False, server_default=1`, but `create_all` skips
existing tables and `_migrate_semantic_tier` verifies indexes/constraints only — the founder's real
DB still reads `mention_count INTEGER` and the verifier still reports "verified". The defense that
actually holds on live data was the SQL-side `coalesce`, not the model change. **Enforced check:**
when a column definition changes, state explicitly which DBs it reaches (fresh only vs migrated),
verify the claim against a real existing DB read-only, and keep a runtime defense for the old shape.

## 2026-08-06 — Say PASS plainly when it is earned (Critics, R6)
Evidence: rounds 1-5 all blocked; round 6 found no blocker and no major. Manufacturing a finding to
justify the round would have cost the team a day and taught them that the gate never opens.
**Enforced check:** a round that finds only notes reports PASS-WITH-NOTES with the notes named and
the verified fixes stated plainly — the verdict follows the evidence, not the round number.
