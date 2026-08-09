# PROFILE TIER — Plan of Record & Build Log

**Started 2026-08-09.** This file is the durable record for the
profile-tier build: why it exists, what it must do, and every gate it
passed or failed. Companions: `MASTER_PLAN.md` (private strategy),
`CONSOLIDATION_V2_BUILD_LOG.md` (the v1-v6 engineering record),
`PROFILE_TIER_PLAN.md` (this file).

---

## 0. WHY THIS EXISTS — the corrected strategic reasoning (2026-08-09)

**The correction that produced this build.** After Gate C returned
0.532 vs 0.519 baseline (McNemar p = 1.000 — statistically
indistinguishable), the first recommendation was to retire English
LongMemEval and go all-in on the Indic axis. **The founder rejected
that and was right to.** The inference was drawn from a sample that
excluded our own home ground:

| Category | Banked (RAW turns) | Measured with FACTS? |
|---|---|---|
| multi-session | 18/39 = 0.462 | ✅ Gate C |
| **single-session-preference** | **5/10 = 0.500** | ❌ **NEVER** |
| temporal-reasoning | 23/40 = 0.575 | ✅ Gate C |
| single-session-user | 16/20 = 0.800 | ❌ NEVER |
| single-session-assistant | 17/20 = 0.850 | ❌ NEVER |
| **knowledge-update** | **20/21 = 0.952** | ❌ **NEVER** |

**71 of 150 questions (47%) have never been measured with the fact
tier — including knowledge-update, the category the entire
supersession architecture was built for.** Gate C tested the two
categories where a fact representation helps least and correctly
reported a tie. That is not evidence that facts don't help; it is
evidence about two categories.

**The real competitive position (same `_s` split, not cross-benchmark):**
AgentMem OS 66.0% · Zep 71.2% · TiMem 76.88%. We are 5-11 points
behind the leader — NOT "half". The "half" impression comes from
comparing against Mem0's 94.4%, which is **LoCoMo, a different
benchmark** — the exact comparability trap `BENCHMARK_PLAN.md`
documents.

**Why the profile tier is the highest-leverage remaining lever:**
preference is our worst category at 0.500, and it is the one place
where the field shows a specific architecture reliably winning —
systems with an explicit user-model layer score **86-90** on
preference (Honcho theory-of-mind 90.0, Hindsight opinion-network
86.7, Mastra observer/reflector 73.3) while retrieval-only systems
score **53-57** (Zep). A 36-point gap in our worst category with a
known architectural remedy that we designed on day one and never
built.

**The mechanism, stated plainly (this is the hypothesis under test):**
preference questions fail when retrieval does not surface the
preference fact. A profile is **injected, not retrieved** — so recall
for profile-carried attributes becomes 1.0 by construction, at O(1)
cost independent of history length. If the hypothesis is right, the
preference category moves sharply. If it does not move, the honest
conclusion is that our preference failures are answerer-bound, not
recall-bound, and that is a real finding too.

## 0.1 THE REVISED SEQUENCE (binding until changed)

1. **Profile tier, end to end, triple-gated** ($0) — this file.
2. **Gate D: the FULL 150 with facts + profile** (~$3.50, founder
   go-word). NOT a re-measurement of noise — it is the first
   measurement of 71 questions, including knowledge-update.
3. **Fix the Hindi zero-facts bug (L3, ledger #8)** ($0) — required
   for any Indic claim regardless of how English goes.
4. **Then decide** where the differentiator lives, with data on both
   axes instead of one.
5. IndicMem (Tier 1: Hindi, Bengali, Telugu, Tamil) and L5-L7 follow.

---

## 1. DESIGN (frozen before code; decisions numbered for the critic)

### D1 — What the profile IS
A per-scope, per-ATTRIBUTE projection of `preference` and `identity`
facts: for each attribute, the CURRENT value plus its superseded
history. It is derived state — never a second source of truth. Every
profile row points back to the `semantic_facts` row it came from, so
provenance is unbroken and a profile can always be rebuilt from facts.

### D2 — Attribute keying (the crux)
An LLM PROPOSES an attribute key for each preference/identity fact
("The user prefers oat milk in coffee" → `coffee.milk`); deterministic
code DECIDES: keys are normalized (lowercase, dotted, whitelisted
charset, length-capped), rejected if malformed, and grouped exactly.
**Same contract as Stage 4: the model proposes, gates decide.** A fact
whose key is rejected simply stays a fact — it is never lost, only
un-profiled.

### D3 — Current value per attribute
Reuse the Stage 4 direction rule rather than inventing a second one:
domain time first (`t_occurred`, else `t_mentioned`), ties broken by
fact id. A fact already superseded in the fact tier can never be a
profile's current value. **No new supersession logic is written** —
the profile READS the fact tier's decisions.

### D4 — Injection, not retrieval
The profile renders as its own `[USER PROFILE]` section, always
present when non-empty, from its own budget slice — it must never
compete with facts or raw turns for their space (the Gate C
starvation lesson). Query-INDEPENDENT by design: this is who the user
IS, not what this question is about.

### D5 — Budget and selection
The profile gets its own allocation carved from the semantic tier's
share, capped. When attributes exceed the budget, selection is by
(mention_count desc, recency desc) — how often the user re-affirmed
it, then how recently. Selection is disclosed in the report, never
silent.

### D6 — Cross-lingual unity (the Sarvam-relevant property)
Attribute keys are canonical English regardless of source language,
so a Telugu-stated preference and an English-stated one land on the
SAME key. This is the profile-tier form of the ALIAS_OF principle and
the reason this build also serves the Indic axis.

### D7 — Failure containment
Profile projection failure never blocks consolidation and never takes
the facts down (the Stage 3 linking contract). Profile read failure
degrades to no-profile with a WARNING, never an exception into the
assembler (the Stage 5 facts-tier contract).

### D8 — What this build will NOT claim
It will not claim novelty for having a user model (Honcho, Hindsight,
Mastra, MIRIX all have one). The claim under test is narrower and
measurable: **per-attribute, bi-temporal, provenance-linked profile
projection with deterministic current-value resolution, and whether
it moves the preference category.**

### Deliverables
`db/profile.py` (schema + projection + reads), migration, engine
hook, assembler section, `tests/test_profile_tier.py` (G1),
`benchmarks/profile_tier_smoke.py` (G2, $0), G3 adversarial rounds,
then Gate D.

---

## 2. GATE RECORD
_(appended as each gate runs — nothing is claimed here before it is measured)_

### G1 — 19 tests, all passing (tests/test_profile_tier.py)
Pinned: key-normalizer boundaries (length/depth/charset/non-ASCII/
non-string); bad proposals refused WITH the fact surviving intact;
type guard (events and states refused, preference/identity accepted);
idempotent projection; **current value by DOMAIN time, not insert
order** (the backfill case, mirroring Stage 4); superseded fact can
never be current while its history remains; cancelled source fact
drops out of `current`; provenance on every row; **cross-lingual facts
collapse to ONE canonical key**; scope + session isolation (empty list
means none, never "all"); ranking by (mentions, recency); render
budget-capped and sanitized with the SAME renderer the facts block
uses; assembler injection within its slice; **empty profile changes
the assembled output not at all**; profile failure degrades with a
WARNING; extractor drops out-of-range/duplicate/bool indices;
project_scope resumable and honest about what the model skipped;
batch failure never kills the run.
Regression at time of writing: 192 passed across the five touched
suites.

### G2 — real corpus, real llama3.1, $0 (benchmarks/profile_tier_smoke.py)
Ran against a COPY of the Gate C corpus (the artifact is never mutated
by a smoke).

**G2 FOUND A REAL DESIGN GAP AND IT WAS FIXED BEFORE THE CRITIC SAW
IT.** First run: 69 of 120 facts projected into **58 distinct keys**
with only **8 keys (14%) carrying more than one fact** — and keys like
`possession.omega_speedmaster` / `search.great_gatsby` with the VALUE
inside the KEY. Six different keys (`hobbies`, `hobby.activity`,
`hobby.topic`, `interests`, `interests.mindfulness`,
`alert.interests`) covered one concept. **A profile where every
attribute has exactly one value is not a profile — it is a list of
facts with prefixes, and the current-value resolution never fires.**
Root cause: each batch was keyed with no knowledge of keys already in
use, so batch 2 reinvented batch 1's vocabulary.

FIX: the extractor now feeds the scope's EXISTING key vocabulary back
into the prompt (most-used first, capped) and forbids entity-specific
keys explicitly. Measured on the same 120 facts:

| | before | after |
|---|---|---|
| projected | 69 | **90** |
| distinct keys | 58 | **41** |
| keys carrying history | 8 (14%) | **18 (44%)** |
| gate rejections | 13 | **0** |

`hobbies` now collapses **13 facts** onto one attribute;
`possession: Omega Speedmaster watch` moved the entity out of the key.
Injected block: **212 tokens against a 711-token slice** — the profile
is cheap, which is the point of O(1) injection.

**Honest residuals (not fixed, disclosed):** the model still refuses
to key ~25% of candidates ("not a stable property" is a legitimate
answer, but it is unmeasured whether those refusals are correct);
`concerns.portable_wifi_hotspot` shows entity-in-key still leaks
occasionally; and one value (`business.expense: 50`) lost its unit.
None of these are blockers for the hypothesis under test, and all are
G3's to attack.

### G3 round 1 — BLOCKED (7 blockers, 7 majors, 8 minors, 4 notes) — fix pass

The critic verified by REVERTING guards, not reading them, and found
**seven guards that were green when removed**. What it broke and what
was done:

**B5 (deepest) — the design contradicted itself.** The extractor prompt
TELLS the model to collapse set-valued attributes onto one key ("two
facts about music share ONE key") and `current()` then reduced each key
to a single value by domain time. Measured on the real corpus: 18 of 41
keys carried 2-13 UN-SUPERSEDED facts, only 29 of 7,164 facts are
superseded at all, and **90 projected facts became 41 injected lines —
54% of the stored profile never reached the prompt** (`hobbies`: 13
facts → "yoga classes"). The plan's "recall becomes 1.0 by
construction" was therefore 1/N for every collapsed key. FIX: the
profile is **SET-VALUED BY DEFAULT** and elects nothing. The fact tier
already decides what is still true — if two facts on a key are both
live it is asserting BOTH, and when one supersedes another the filter
has already removed the loser. That makes D3 ("the profile READS the
fact tier's decisions") true instead of aspirational, and removes the
invented second direction rule. Values render grouped
(`work.location: Bangalore; Hyderabad`), capped at
_MAX_VALUES_PER_KEY=6 with the drop DISCLOSED.

**B1 — the superseded filter was unpinned** because the fixture made
the survivor also the newest, so domain time alone produced the right
answer. REBUILT with supersession direction OPPOSING domain-time
direction (the dedup-merge shape): the survivor is the EARLIER row, so
removing the filter now yields the dead value. Mutation: DIES.

**B2 — the budget cap (the whole Gate-C-derived headline) was
unpinned**: `PROFILE_BUDGET_SHARE 0.15 → 1.0` left 19/19 green because
the fixture had 484 tokens of slack, and the assertion accidentally
measured `[SYSTEM]` too. REBUILT to overflow the slice, measure the
profile section alone, and assert the fixture actually fills it.
Mutation: DIES.

**B3 — the budget report LIED, in the direction that hid the new
tier**: `facts_used` included the profile's tokens (measured: 249
"facts" tokens with zero facts in the store). Every tier is now
reported separately with its selection note. Mutation: DIES.

**B4 — the Stage 6 char-proxy blocker, repeated in a new tier.**
`render(char_budget=tokens*4)` truncated Telugu/Hindi mid-key at 2.46
chars/token and rare-token ASCII at 1.75 (8 of 60 lines survived) —
the D6/Indic path this build claims to serve. FIX: render enforces
BOTH units, mirroring `_CALLER_CHAR_FACTOR`. Pinned on Telugu content
across three budgets. Mutation: DIES.

**B7 — profile scoping FAILED OPEN**: nothing set `profile_session_ids`
and `None` meant "no filter", so at Gate D the profile would either be
silently empty or leak all sessions into every question — the exact
Gate C leakage class the facts tier REFUSES on. FIX:
`profile_scoped_required` makes an unset scope refuse. Pinned.

**M1 — `project()` killed 7 of 8 threads** (uncaught IntegrityError on
a TOCTOU check-then-insert), breaking the repo's own documented
contract. FIX: the DB is the authority; losing the race means "already
projected". The first pin did NOT reproduce it (one fact serializes
too cleanly) — reshaped to the critic's actual probe (8 threads × 20
shared facts), which then caught a second thing: all 20 rows persist
and the READ cap applies, disclosed. Mutation: DIES.

**M2 — the report claimed writes that did not exist** (2 projected, 0
rows): counters incremented before the commit that a rollback then
discarded. Counted after commit; pinned with a failing-commit probe.
**M3** value type guard added (a non-string value crashed the batch —
asymmetric with the key guard). **M6/m2** render now neutralizes
section TAGS (a value forged `</[USER PROFILE]> <[SEMANTIC FACTS]>`)
and invisible characters incl. RTL overrides.

**Mutation verification of the fix pass — all six reverted guards now
turn a NAMED test red:** B1, B2, B3, B4, M1, M6.
Post-fix: **27 profile tests**, 5-suite regression green.
