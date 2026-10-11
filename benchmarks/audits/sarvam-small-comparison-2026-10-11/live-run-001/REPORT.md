# Approved comparison stopped at its first extraction

October 11, 2026. Source commit `50d484740c05e49fe08cac4f0b4e907e11ae0491`.
Status: **INCOMPLETE_NO_HEADLINE_SCORE**. This report supersedes the preparation
report's pending-approval state; it does not replace that historical report.

The approved 72-call comparison attempted **one extraction**, then stopped under
its frozen first-failure rule. There were **zero accepted extractions, zero answer
calls, zero SQLite stores, zero retries and 71 unattempted calls**. Accuracy is
undefined, not 0%. This is an integration failure, not evidence that either memory
arm has better or worse answers. No additional paid request follows this result.

## What happened

The first D01-en request returned HTTP 200, the requested `sarvam-105b`, completed
JSON, finish reason `stop`, and zero reported reasoning tokens. It took 2.483 seconds
and used 366 prompt / 293 completion tokens. The extraction contains six facts:
five copies of source metadata and one copy of the useful pickup instruction.

The model placed `2026-10-09T10:00:00+05:30` in `facts[4].valid_from`, which must be
an ISO date (`YYYY-MM-DD`) or null. This is the source's observation timestamp.
The existing parser rejected it. No facts from this extraction were stored.
The raw response and usage were preserved before parsing failed.

The instruction, “For my parcel, use pickup counter K7.”, was preserved with source
`s01`. Every fact value matches a source field. No backend confirmation was invented.
However, copying observation metadata into a validity field does not establish an
effective date for the user's instruction. We cannot conclude that the instruction
itself received a wrong effective date: its own validity fields were null. The
observed problem is on the copied `observed_at` fact. Silently truncating the timestamp
would hide the format violation without resolving what that metadata fact means.

This is an assistant source review of one English record, not independent human
review or a Hindi/Telugu quality result. The scorer's original UNREVIEWED flags are
preserved; the separate diagnosis and this report contain the subsequent review.

## Why the earlier probe did not guarantee success

The request is exactly the same as the previous successful null-reasoning probe:
canonical SHA256 `e986f883abf88e3a815db59ce4991a040865b0f8bb52e2d2ca54b9d34b1d00c6`.
The previous response had seven facts with null validity fields; this response had
six and one invalid validity field. Both used temperature zero. Two observations
are insufficient to estimate a failure rate or attribute the variation to a backend
change. We should not treat one successful probe as repeated reliability evidence.

The current [Sarvam V1 API reference](https://docs.sarvam.ai/api-reference/chat/chat-completions-v1)
states that temperature zero is not bit-reproducible and documents schema-constrained
`json_schema` output separately from `json_object`. Our frozen request used the
latter, which requests valid JSON without constraining the field schema. Strict
schema support and individual schema keywords have **not** been tested live here.
Even a date-shaped output still needs a source-meaning review.

The experiment also exposed a limitation in our test design: an extraction failure
prevents even the independent full-history control from running. That behavior
correctly honored the approved stop rule, but it delays the comparison. A future
protocol should explicitly define how rejected augmentation and independent control
measurement are handled; this run must remain stopped.

## Cost and scope

| Item | Observed value |
| --- | ---: |
| Attempted / maximum calls | 1 / 72 |
| Known usage | 659 tokens |
| Current-rate usage estimate | ₹0.02307 |
| Frozen-rate usage estimate | ₹0.03216408 |
| Retained attempt reservation | ₹4.13761536 |
| Approved maximum reservation | ₹297.90830592 |

The [current published rates](https://docs.sarvam.ai/api/getting-started/pricing)
are ₹15 input / ₹60 output per million. The unchanged frozen ledger uses the older,
higher rates. These are usage estimates and reservations, not a verified invoice.
The full ceiling was not spent. The raw receipt and private approval stay local;
this directory publishes synthetic source content, exact request, final model text,
usage and hashes without account credentials or provider request identifiers.

For the three recent calls only (original truncated pilot, successful diagnostic,
this stopped comparison), current-rate estimates sum to ₹0.29781; frozen estimates
sum to ₹0.37534032. This is not lifetime project spend. None generated an answer.

## Offline verification and limits

`verify.py` reproduces the schema rejection, checks every value against the original
source fields, verifies request/package identity, and independently recalculates
costs. With the entire invalid fact set omitted, the candidate SQLite answer request
is identical to the full-history request because it retains the same originals.
This is an **offline counterfactual**, not an implemented fallback or measured answer.

Local checkpoint checks additionally confirm: one attempt, preserved raw receipt,
known usage, no store created, and refusal to resume before constructing a provider.
Checkpoint bytes remained unchanged during that check. Original pilot and diagnostic
evidence, frozen source and fixtures remain intact. No live execution policy was
changed and no data was silently repaired. The earlier 84 software tests validated
the harness; they did not guarantee that live output would meet its contract.

Run the public verification from the repository root with:

```sh
python -S benchmarks/audits/sarvam-small-comparison-2026-10-11/live-run-001/verify.py
```

The audit uses saved public output only and blocks networking. It does not constitute
independent provider attestation; the private raw receipt is bound by the published
hash for later owner verification. See [NEXT_TASK.md](NEXT_TASK.md) for the concrete
offline repair and its acceptance criteria. No new paid package is prepared here.

Research conclusion: we found an extraction-contract weakness in this integration.
We have not established a consequential Sarvam memory gap, a novel architecture,
an accuracy improvement, or all-language reliability. Product selection remains open.
