# Extraction admission v3: offline repair complete, live result pending

October 11, 2026. Implementation commit `045f255`. No paid calls or credential-file
reads in this turn. Model accuracy is **not measured**.

The previous stopped comparison could not produce answers after an extraction put
an observation timestamp into a date-only validity field. The separately versioned
v3 runner now rejects malformed augmentation while retaining the same original
evidence for answering. It records the rejection and its cost. Historical v1/v2
source, fixtures, prompts, raw receipts and scores remain unchanged.

## Completed engineering

1. Source metadata and generated facts are separate. Tenant/customer authorization,
   session and observation timestamp come from original input records. Scope and time
   filtering precede extraction, persistence and serving. The extractor receives
   source ID, role, observation timestamp and text, but no questions or answer labels.
2. Extraction uses a bounded JSON Schema and a prompt that distinguishes observation
   time from effective dates. Local checks independently reject malformed types,
   impossible/reversed dates, invalid citations, and explicit metadata attributes.
   No timestamp is silently trimmed and no model fact is hand-corrected.
3. Completed known-usage invalid extractions are `rejected`, with no admitted facts.
   The SQLite arm persists/reopens authorized originals and can answer using them.
   Empty valid extraction is labeled separately. All extraction cost remains charged
   to SQLite. Answer requests without admitted facts exactly match the control.
4. Completed malformed answers remain `invalid_answer` in the intended denominator.
   Transport, model, usage, truncated/missing final content, duplicate receipts,
   budget and storage failures still stop the run. There are no retries or automatic
   model/schema downgrades. Reserved slots and started API requests are counted
   separately, including a storage failure before dispatch.
5. Scoring separates schema admission, source review, fallback, exact answer fields,
   evidence coverage, valid-only accuracy and all-intended accuracy. Interrupted runs
   retain undefined all-intended accuracy; simulations have an explicit watermark.

These are changes to the research harness. They do not yet establish a better
production memory architecture or a Sarvam product gap.

## Verification evidence

- **138 focused tests pass:** 54 new and 84 historical regression tests. New tests
  block network access; selected suites use isolated temporary storage. Ruff and
  whitespace checks pass. Initial lint issues were corrected before the passing run.
  Disabling pytest plugin autoload emits an `asyncio_mode` configuration warning;
  these selected tests are synchronous and none was skipped for that warning.
- The saved live failure is rejected without editing it. A 72-call all-rejected
  simulation still completes both answer arms, retains every original and charges
  all extraction cost. This is a software-policy check, not fresh model output.
- A separate network-blocked mixed simulation resolves 72 synthetic calls with
  8 rejected, 8 empty and 8 nonempty extractions plus 1 invalid answer. All 24 SQLite
  stores reopen; 60 original-record occurrences are retained. Sixteen empty/rejected
  answer packets exactly match their controls. Semantic reviews remain UNREVIEWED.
- The 12 generated schemas pass a local JSON Schema Draft 2020-12 validator. That
  does not establish provider keyword support. No new runtime dependency was added.
- Historical runtime/fixture hashes, three prior live checkpoint hashes and English
  CURRENT_STATE are unchanged. The earlier English 84.6% is not a Sarvam score.

An important negative test deliberately supplies a false but structurally valid
fact. It passes structural admission and remains **UNREVIEWED** for meaning. Thus
the implementation does not guarantee entailment, complete conditions, or correct
effective dates merely because the output is well formed. Explicit metadata-name
rejection also cannot catch every paraphrase. Actual extracted facts and original
sources must be reviewed after a live run; the original-evidence answer instruction
is not an automatic correctness certificate.

The current [Sarvam API reference](https://docs.sarvam.ai/api-reference/chat/chat-completions-v1)
and [chat guide](https://docs.sarvam.ai/api/api-guides-tutorials/chat-completion/overview)
document schema-constrained output and name/schema/strict configuration. The exact
schema and its keywords have not been tested against the live service here. API
rejection will stop the proposed run rather than trigger a weaker schema silently.

## Frozen next comparison — not approved or executed

Package SHA256: `bf04d07dbe2a9b2fe2e77be5ce08cdc19eed511de6af2af075cf182ef5fa0b65`.

Same `sarvam-105b`, explicit null reasoning, temperature 0, 4,096 completion cap,
unchanged answer prompts and gold labels. Twelve existing EN/HI/TE development
renderings (four family groups), two repeats. **72 calls maximum: 24 extractions
and 48 answers**, no retries or additional smoke call. Conservative maximum
reservation **₹297.90830592 (ceiling ₹297.91)**, not predicted spending or an invoice.
The approval template is false, and no live-run directory exists.

The answer order remains counterbalanced. For the first pair, full-history precedes
extraction and SQLite; alternate pairs use extraction/SQLite/full-history. Control
does not consume extraction data. Rejected augmentation is handled under the explicit
new policy; infrastructure failures still halt globally. The first schema-mode call
is part of the 72, not a hidden extra probe.

Both arms see the same authorized originals. SQLite adds extraction compute and
possibly admitted facts: total compute is not matched. Extraction prompt, input
projection, schema and rejection policy change together, so this is a prospective
system comparison, not a causal single-lever ablation. A measured gain would need
fresh families and matched-compute/ablation checks before a stronger claim.

If ordinary full history already succeeds, retain that simple baseline. If adding
facts hurts, inspect augmentation. If it helps, verify source meaning before scaling.
Do not claim improvement just because fallback preserves a control answer.

Cases remain assistant-reviewed development data; translated views and repeats are
correlated, with no independent native human validation. Real tools, long histories,
native Sarvam/framework controls, full erasure and wider endpoint-supported languages
remain pending. Public benchmark coverage and product novelty remain open requirements.

## Reproduce offline

```sh
python -m pytest tests/test_sarvam_comparison_v3.py tests/test_sarvam_comparison_v2.py tests/test_sarvam_discovery.py -q
python -S benchmarks/audits/sarvam-admission-v3-2026-10-11/verify.py --output-dir /absolute/new/simulation
python -m benchmarks.sarvam_comparison_v3 check --package benchmarks/audits/sarvam-admission-v3-2026-10-11/package.json
```

The simulation refuses an existing output directory, blocks sockets, uses synthetic
receipts and needs no credentials. Execution and scoring instructions are in the
[v3 README](../../sarvam_comparison_v3/README.md). Only a newly approved, exact bound
scope can activate live execution. Prior stopped-run approvals cannot transfer.
