# Smaller Sarvam comparison prepared — no live score

Started October 10; completed preparation October 11, 2026 (America/New_York).
Implementation commit `e79bb5f`. No API calls, credential-file reads or paid spend in this turn.

## What is ready

A separate `benchmarks.sarvam_comparison_v2` path prepares, validates, runs and scores
12 existing development renderings across 4 family groups and 2 repeats. It selects
D01 recall, D06 current/future dates, D07 conditional permission and D09 backend
confirmation/ordering, each in English, Hindi and Telugu. The original 228-call
pilot and its stopped checkpoint are untouched.

Both arms use sarvam-105b with explicit JSON null reasoning, temperature 0, JSON
output and maximum 4,096 completion tokens. Prompts, sources, labels and evaluation
rules are unchanged. SQL retains the same original evidence as full history plus
actual extracted facts, after close/reopen. Counterbalanced answer order; no earlier
probe response is reused. SQL includes extra extraction compute; no total-compute
matching or architectural superiority claim.

**Planned maximum 72 calls: 24 extractions + 48 answers.** Frozen package:
`07cb99155472ad97d49234351870726f8df314c6c5a51b6db1212b8ecae84744`. Conservative reservation ₹297.90830592 (displayed ceiling ₹297.91),
using the original higher rates and whole-context input reservation. This is not
expected spend or a verified invoice. No retries, fallback or second run. Stop on
first failure, retain raw receipts and known/unknown usage, refuse pending/error
resends. The approval template is **false**; no live directory exists.

Current official [pricing](https://docs.sarvam.ai/api/getting-started/pricing) remains
₹15 input / ₹60 output per million, lower than the retained reserve rates. The
[model documentation](https://docs.sarvam.ai/api/getting-started/models/sarvam-105b)
still documents 128K context and optional reasoning consuming the output allowance.
Recheck before a future approved dispatch; this turn did not inspect account balance.

## Verification

- 84 focused tests pass: 38 for the new path and 46 legacy tests. Network disabled.
- Ruff and diff checks pass. Worktree-name-only N999 warning ignored; no lint setting changed.
- A separate 72-call synthetic simulation completes, with 24 closed/reopened SQLite stores
 and 24 answer pairs. All requests differ from their original builders only in
 reasoning_effort=null; all authorized originals are retained with empty/nonempty facts.
- Durable reservations precede each fake dispatch, and stage/arm accounting balances.
- Historical package/code/checkpoint and successful diagnostic checkpoint remain unchanged.
- Clean Git export without site packages or credential environment reproduces the exact manifest.
- The evaluator retains field accuracy and source coverage, and adds source/fact dossiers
 with six explicit UNREVIEWED semantic checks. Schema validity never completes that review.

Synthetic outcomes are marked SIMULATED_TEST_ONLY and say nothing about Sarvam's
accuracy. The former 84.6% English result is unchanged; it is not a Sarvam score.
The 84 passing software tests are likewise not an 84% model score.

## Decision after a separately approved run

Inspect every extraction and answer. If both arms work, retain the simple control
for these requirements. If SQL harms, inspect augmentation before adding architecture.
If SQL helps, use fresh cases and a matched-total-compute control before claiming a
causal architectural improvement. If both fail, review source/labels/language before
calling it retrieval failure. Keep dates, conditions and requested-vs-confirmed state
visible. Do not change prompts during a frozen run to make failures disappear.

This is assistant-reviewed public development data with correlated language views,
not independently native-validated or held out. It uses short records and supplied
backend receipts, not actual tools, full erasure or a production engine. Broader
languages, native Sarvam/framework controls and public benchmark coverage remain
required follow-up work; product requirements and novelty are not established.

Runtime/checkpoint and scoring orchestration are separately versioned to preserve
historical replay; shared provider/parsers/prompt/storage/grade functions are reused.
No global monkeypatch sets production model parameters. Source and evidence are
committed separately. Git CI and merge completion are recorded in local receipts.
