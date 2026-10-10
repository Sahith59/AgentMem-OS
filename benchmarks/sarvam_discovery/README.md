# Sarvam discovery comparison v1

Status: runnable exploratory preparation, **no live results**. This measures whether
ordinary stored facts help or harm a fixed Sarvam model on a small, inspectable set.
It does not evaluate the AgentMem production engine or establish a new product gap.

There are 38 synthetic English/Hindi/Telugu renderings, 12 scenario templates and
9 underlying family groups. Translations, contrasts and repeats are correlated.
All cases and labels were authored and reviewed by assistants, including a separate
GPT-6-sol peer review. Independent native human validation is pending. These are
public development examples, never a future held-out benchmark set. Other relevant
Sarvam-supported languages remain intended scope; none gains a quality claim from
this first three-language slice.

## What is compared

| Arm | Evidence | Model calls per case and repeat |
| --- | --- | --- |
| Full history | Every authorized original record through the cutoff | One answer |
| Ordinary SQLite | The same original records plus extracted facts; store is closed and reopened | One extraction, one answer |

Both arms use `sarvam-105b`, temperature 0, reasoning effort `low`, JSON output and
maximum 4,096 completion tokens. No stronger model, external judge or fallback is
used. Temperature 0 does not promise determinism. Two repeats alternate arm order;
extraction happens first. Original records always remain available in SQLite even
if extraction misses a fact. The two arms collapse to identical requests if
extraction returns no facts, which is appropriate for that null intervention.

This is **not matched total compute**: SQLite adds extraction and a longer answer
input. All extraction cost is assigned to SQLite. It tests fact augmentation under
equal original evidence, not a sophisticated retrieval architecture or long-history
compression. The longest fixture has 17 short records. Both prompts describe the
same source and answer policy. The extractor never sees the question, requested
answer fields, family label or gold. It extracts from original records only.

## Cases and known limits

| Template | Test | Limit |
| --- | --- | --- |
| D01 | Short native recall | Easy control |
| D02 | Native source, English question | Shares D01 English anchor; no duplicate D02-en |
| D03 | Romanized Hindi/Telugu source | English is the Latin-script control |
| D04 | Verified identity vs missing alias context | Three added ambiguity contrasts require clarification |
| D05 | Common word vs company referent | Explicit contextual disambiguation |
| D06 | Current vs future-effective desk | Preserves date boundary |
| D07 | Conditional authorization | Depends on recorded repair approval, not merely a future date |
| D08 | Relevant fact among distractors | Same pickup family as D01–D03, not an independent sample |
| D09 | Failed/accepted/delayed backend receipts | Supplied receipts; no actual tool execution |
| D10 | Contact preference across sessions | Every SQLite arm reopens a DB; no production persistence claim |
| D11 | Customer/tenant scope | Supplied authorized ID filtering, not identity authentication |
| D12 | Withdrawn address, retained notebook preference | Answer suppression only; audit retains originals, no erasure claim |

Sources, expected canonical fields and review records are versioned in
[`../fixtures/sarvam_discovery_v1`](../fixtures/sarvam_discovery_v1).
Gold is an evaluator-only file. Its hash is bound by the review and package, but its
contents are not read by the inference runner. All labels specify the source IDs
needed for interpretation. Records and prompts are synthetic and contain no account
credentials or private customer data.

## Runbook

Use Python 3.11+ from the repository root. Runtime uses the standard library and the
existing atomic JSON writer; it does not import the application DB or read `.env`.

```sh
python -m pytest tests/test_sarvam_discovery.py -q
python -m benchmarks.sarvam_discovery prepare --output-dir /absolute/new/package-dir
python -m benchmarks.sarvam_discovery check --package /absolute/new/package-dir/package.json
```

Preparation is offline. It writes `package.json`, `preflight.json` and an unapproved
`approval-template.json`, refusing to replace a directory. All runtime source files,
fixture files and exact known requests are hashed. SQLite requests depend on the
preceding extraction, so the derivation code and bounds are frozen; each exact
derived request and hash are durably recorded before dispatch. A source, setting,
review or label change invalidates the prepared package.

The live package contains **228 calls maximum**: 76 extractions and 152 answers.
There are no retries. It stops on the first failed HTTP call, unexpected returned
model, missing/invalid usage, incomplete or malformed final answer, duplicate receipt
ID, source-scope violation, or store mismatch. Reasoning text is never used as the
answer. A timeout or interruption retains its entire reservation and cannot be
automatically resent. Inspect the stopped run before proposing a new package;
do not delete a pending attempt to resume spending.

After the founder approves this exact package, copy the template to an approval
record, set `approved` to true and record the actual authorization text. The package
hash, output path, maximum attempts, review level and budget must match. Credentials
belong only in the local `SARVAM_API_KEY` or `SARVAM_API_SUBSCRIPTION_KEY` environment;
do not put them in fixtures, approval records, command arguments or chat.

```sh
python -m benchmarks.sarvam_discovery run \
  --package /absolute/new/package-dir/package.json \
  --approval /absolute/new/package-dir/approval.json \
  --run-dir /absolute/new/package-dir/live-run
python -m benchmarks.sarvam_discovery score \
  --package /absolute/new/package-dir/package.json \
  --run-dir /absolute/new/package-dir/live-run \
  --output /absolute/new/package-dir/score.json
```

The exact HTTP configuration is checked against the
[chat API contract](https://docs.sarvam.ai/api-reference/chat/chat-completions-v1).
Provider behavior and account access are **not yet live-verified**. The first
scheduled request is also the transport/response-contract check; incompatibility
stops the package without an automatic fallback.

## Money and evidence

Frozen October 9, 2026 [published rates](https://docs.sarvam.ai/api/getting-started/pricing):
₹29.28 per million input tokens and ₹73.20 per million output tokens. No cached-input
discount is assumed. Each attempt reserves the entire documented 128K context
(131,072 tokens) plus 4,096 output tokens, rather than trusting a guessed tokenizer.
The whole-package maximum reservation is **₹943.37630208**. This is deliberately
conservative, not an expected bill. It bounds this runner at the frozen rates and
documented token limits; it is not an account-wide cap, tax estimate or guarantee
about changed provider billing. Recheck rates and the
[model contract](https://docs.sarvam.ai/api/getting-started/models) before activation.

The checkpoint persists a reservation before each call, then its raw response,
final content, model/receipt ID, finish reason, usage and elapsed time. Known usage
is retained even when final JSON is invalid; unknown usage remains visibly unknown.
Reports show prompt/completion tokens, frozen-rate cost and reservations by stage
and arm, charging extraction to SQLite. Neither failed calls nor repeated extraction
are free. Keep raw checkpoints and SQLite files locally; review credentials and
account identifiers before intentionally releasing live artifacts.

The evaluator uses exact canonical field values plus a separate clarification check.
Required source coverage is reported separately; citing all records can satisfy
coverage, so it is not a proof of citation precision or explanation quality. Invalid
and unattempted answers stay in the intended counts. Accuracy among valid answers
has an explicit denominator; an incomplete live run is never a headline score.
Reports include per-case paired wins/losses, each repeat, language direction and
family. No confidence interval treats translated views as independent cases.
Simulated runs are stamped `SIMULATED_TEST_ONLY`; their scores are not model results.

## How findings determine the next step

1. **Both arms succeed:** retain the simple approach for these requirements. These
   cases do not establish a product gap. Seek independently sourced, consequential
   cases before adding architecture.
2. **Full history succeeds, SQLite fails:** inspect unsupported/conflicting extracted
   facts and whether the answerer ignored originals. Fix or remove augmentation.
3. **Both fail:** inspect labels, protocol and language interpretation first. All
   original evidence was present, so this alone is not a retrieval-miss diagnosis.
4. **SQLite improves:** repeat-specific wins/losses and costs are exploratory leads.
   A matched-compute control and new family-held-out examples are required before
   attributing gains to architecture or publishing superiority.

Every apparent failure gets a trace review against originals, not just an aggregate
percentage. Compare matched language views and repeats; classify uncertain labels
separately. Native Sarvam CRM/hooks, existing memory frameworks, larger histories,
actual tool effects and the remaining endpoint-supported languages are still needed
before choosing a product or claiming benchmark coverage. This package does not
promise that a novel gap will appear.
