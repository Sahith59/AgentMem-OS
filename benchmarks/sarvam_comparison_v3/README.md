# Sarvam comparison v3: extraction admission with original evidence retained

This is a development experiment, not the production memory engine. It keeps the
historical v1/v2 runners, fixtures, stopped runs and answer prompts unchanged.
No live v3 result or accuracy gain exists at preparation time.

The previous run stopped when a generated metadata fact put an observation timestamp
in a date-only validity field. V3 separates three outcomes: structurally admitted
facts, rejected augmentation, and infrastructure failure. Generated facts never
set tenant/customer authorization or overwrite original source metadata.

## What changes

- Extraction gets authorized source ID, role, observation timestamp and original text,
  plus the query cutoff. Customer/tenant/session envelope values stay in the original
  store; questions, requested answer fields and gold labels never enter extraction.
- The prompt distinguishes observation time from effective dates and asks for semantic
  facts without metadata copies. A schema bounds fact count, keys, types, statuses,
  date shapes and allowed source IDs. Local checks additionally reject impossible or
  reversed dates, duplicate/unknown citations and reserved metadata attribute names.
- The [V1 API reference](https://docs.sarvam.ai/api-reference/chat/chat-completions-v1)
  documents `json_schema`, with name/schema/strict configuration. Support for this
  exact schema and its keywords remains **untested live**. An API rejection stops;
  there is no automatic downgrade to JSON-object mode or alternate model.
- A completed, known-usage extraction that fails admission is recorded as `rejected`.
  Its entire fact set is omitted; original authorized records remain. The SQLite arm
  closes/reopens its store and can answer from those originals. Its extraction cost
  remains charged to SQLite. This is a visible source-only fallback, not successful
  extraction or an architectural gain. Empty valid memory has a separate label.
- A completed, known-usage malformed answer is `invalid_answer`; the schedule may
  continue without retry. All-intended accuracy counts it as unsuccessful, alongside
  conditional accuracy among valid answers. Incomplete runs have no headline score.
- Non-200 response, wrong model, incomplete/missing final content, bad/unknown usage,
  transport failure, duplicate receipt, budget or storage failure stops the run.
  Pending/error attempts are never resent. Reserve before persistence/dispatch;
  distinguish reserved slots, started requests, and unknown usage.

Dates and citations passing a parser **do not prove source meaning**. A fabricated
but well-formed fact can pass structural admission; a regression test demonstrates
this limit. Every extraction retains UNREVIEWED checks for meaning, dates, conditions,
identity, metadata duplication and requested versus confirmed status. Reserved-name
rejection catches explicit metadata copies, not every paraphrase or semantic error.
The unchanged answer prompt tells the model that originals govern conflicts; that
instruction is not a verified entailment guarantee.

## Prospective comparison

Same `sarvam-105b`, explicit null reasoning, temperature 0, 4,096 completion cap,
12 existing EN/HI/TE development renderings, 4 family groups, 2 repeats. Maximum
72 calls: 24 fresh extractions, 24 full-history answers, 24 SQLite answers.
No extra smoke call, retry, model upgrade or reuse of a previous extraction.

Order alternates full-history/extract/SQLite and extract/SQLite/full-history across
case/repeat pairs. Full-history needs no extraction result. Schema rejection allows
both answer arms to be measured under this new policy; infrastructure failures still
halt globally. The first full-history answer precedes the first schema-mode request.

Both arms receive exactly the same eligible originals. SQLite additionally receives
structurally admitted facts when available. Extra extraction compute is counted;
this is **not matched total compute**. Extraction prompt, input projection, schema,
admission and execution policy change together, so results cannot isolate one cause.
If there is a gain, use ablations, fresh cases and matched compute before claiming
architectural superiority. Success of the ordinary control is a valid research result.

The conservative old rates remain ₹29.28 input / ₹73.20 output per million, reserving
the entire 128K context plus 4,096 output per request: ₹4.13761536 per attempt,
₹297.90830592 total (ceiling ₹297.91). Recheck actual published prices before any
approved dispatch; this reservation is not predicted spend or an account-wide cap.
Report all usage and unknown amounts; schema/API failures are not free successes.

## Offline use

```sh
python -m pytest tests/test_sarvam_comparison_v3.py tests/test_sarvam_comparison_v2.py tests/test_sarvam_discovery.py -q
python -m benchmarks.sarvam_comparison_v3 prepare --output-dir /absolute/new/package
python -m benchmarks.sarvam_comparison_v3 check --package /absolute/new/package/package.json
```

Preparation refuses an existing output directory and creates an approval template
with `approved: false`. Source hashes include all shared v1/v2 and new v3 modules;
fixtures/review hashes and the exact prospective policy/schedule are bound.

## Separately authorized execution and scoring

After approval of the exact package, call cap, cost, review limitations and changed
failure policy, save a new bound approval. Old stopped-run approvals cannot transfer.
Use environment-only credentials; never put a key in Git, prompts or reports.

```sh
python -m benchmarks.sarvam_comparison_v3 run --package /absolute/package/package.json --approval /absolute/package/approval.json --run-dir /absolute/package/live-run
python -m benchmarks.sarvam_comparison_v3 score --package /absolute/package/package.json --run-dir /absolute/package/live-run --output /absolute/package/score.json
```

Scoring retains exact-field correctness and required citation coverage separately;
neither certifies every cited source's meaning. Extraction reviews include raw final
text, admitted facts, rejection status and originals. Answer rows identify admitted
facts, empty extraction, or originals after rejection. Software simulations carry
`SIMULATED_TEST_ONLY`; they are never Sarvam model results.

All cases are assistant-reviewed development data. Language views/repeats are
correlated, with no independent native validation. Long histories, real tools,
native platform/framework controls, erasure and broader endpoint-supported languages
remain future work. A public full benchmark and product novelty are not established.
