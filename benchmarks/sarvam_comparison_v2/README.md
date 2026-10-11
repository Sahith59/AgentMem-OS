# Small Sarvam comparison v2

Preparation only. No live answer results. This separate runner preserves the
original stopped pilot and its exact source/package bindings. It tests whether
adding model-extracted facts to the same original evidence helps or harms answers.
It does not evaluate the production AgentMem engine or prove a novel product gap.

Twelve existing development renderings: D01 recall, D06 current/future dates,
D07 conditional permission and D09 requested/rejected/confirmed/delayed delivery
state, each in English, Hindi and Telugu. Four family groups, two repeats; translated
views and repeats are correlated. Fixtures, labels, prompts and review records are
unchanged. They have assistant peer review, not independent native human validation.

Both arms use `sarvam-105b`, temperature 0, `reasoning_effort: null`, JSON output and
4,096 maximum completion tokens. Full history receives all authorized originals.
SQLite receives the same originals plus the actual model extraction, after closing
and reopening the database. Never replace that extraction with hand-authored facts.
SQL has extra extraction compute and input overhead: this is not total-compute-matched.
The two answer arms alternate order across cases/repeats. The successful earlier
single extraction is not reused or counted as a free trial.

The schedule has 24 extractions and 48 answers: **72 maximum calls**, no retries,
stopping on the first invalid response or failed invariant. The original conservative
rates (₹29.28 input / ₹73.20 output per million) reserve ₹4.13761536 per attempt,
or **₹297.90830592 total** (displayed ceiling ₹297.91). This reserves the full documented
128K input window plus the output allowance on every call; it is not expected spend
or an account-wide cap. Current prices must be checked before any approved dispatch.
Raw usage is counted even on malformed final output; timeouts keep unknown usage
visible and retain their reservations. Pending/error attempts are never resent.

## Offline preparation

From a checkout with Python 3.11+, without credentials:

```sh
python -m pytest tests/test_sarvam_comparison_v2.py tests/test_sarvam_discovery.py -q
python -m benchmarks.sarvam_comparison_v2 prepare --output-dir /absolute/new/package
python -m benchmarks.sarvam_comparison_v2 check --package /absolute/new/package/package.json
```

Preparation refuses an existing directory and leaves approval false. It binds all
old/new runtime and evaluator source, fixtures, review, subset, schedule and known
requests. Dependent SQLite requests are derived only from prior final extraction and
originals; each exact request is saved before dispatch. Inference never parses gold
labels. The evaluator validates all labels then selects this fixed subset.

## Separately authorized execution

Obtain approval of the exact package hash, 72-call/no-retry scope, review limitations,
reservation and output directory. Record it in `approval.json`; never flip the
template merely because a key exists or an earlier pilot was approved. Credentials
use the existing environment-only provider. Do not put a key in this repository.

```sh
python -m benchmarks.sarvam_comparison_v2 run \
  --package /absolute/new/package/package.json \
  --approval /absolute/new/package/approval.json \
  --run-dir /absolute/new/package/live-run
python -m benchmarks.sarvam_comparison_v2 score \
  --package /absolute/new/package/package.json \
  --run-dir /absolute/new/package/live-run \
  --output /absolute/new/package/score.json
```

Scoring preserves exact-field correctness, clarification and required source coverage
as distinct checks. It includes all intended/error/unattempted counts and paired
outcomes per repeat/language/family, charging every extraction to SQL. Simulations
are labeled `SIMULATED_TEST_ONLY`; incomplete live runs have no headline score.
`extraction_review` pairs original evidence with actual extracted facts and leaves
meaning, metadata, dates, conditions, action status and scope explicitly UNREVIEWED.
An assistant source review or human review must be recorded separately; schema
success does not complete this worksheet. No model judge or extra hidden call exists.

If both arms work, prefer the simple control for these requirements. If SQL harms,
inspect augmentation before adding architecture. If SQL helps, test fresh families
and matched total compute before claiming superiority. If both fail, examine source,
label and language issues before attributing the result to retrieval. This slice has
short histories and supplied backend receipts, not real tool execution, full erasure,
native platform/framework benchmarking or all-language quality coverage. Those remain
necessary follow-up work before broader product claims.

Implementation deliberately versions the execution/checkpoint and scoring orchestration
while reusing the provider, parsers, storage, prompts and grade functions. It never
patches the old module's global settings, so historical replay remains possible.
