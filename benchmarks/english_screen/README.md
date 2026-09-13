# Frozen English source-supplement screen

This opt-in, serial runner compares 150 selected questions with and without a source supplement. It does not change application defaults. The population includes 72 persistent misses, 20 disagreements, 58 stable-pass controls and all 30 abstention questions. It is development-exposed; its accuracy is not an estimate of full500 accuracy.

The original answer prompt, generator settings (`gpt-5.6-luna`, 4,200 completion tokens), official judge template/settings (`gpt-4o`, five output tokens, temperature zero), and baseline packets are preserved. Only the delivered context changes. The previous source verifier establishes that every added passage is a complete, attributed source turn in the question's permitted scope.

## Prepare and verify

Run modules from the Git root. `build --help` lists required immutable source artifacts. Build refuses an existing output directory. No command makes model calls unless `packet` receives `--execute-paid` and a matching founder approval record.

```sh
python -m benchmarks.english_screen.packet /absolute/path/to/package.json
python -m benchmarks.english_screen.verification /absolute/path/to/package.json
python -m benchmarks.english_screen.offline /absolute/path/to/package.json /new/offline/output
python -m pytest tests/test_english_screen.py -q
```

The offline exercise denies socket connections and runs both distinct-answer and identical-answer scenarios, followed by zero-call completed resumes. Its answers and grades are synthetic. They provide no accuracy evidence.

## Paid execution, only after exact approval

The approval JSON must contain `status: FOUNDER_APPROVED`, the canonical `package_sha256`, integer `budget_nusd`, the exact absolute `run_directory`, and the actual `founder_message`. Do not invent an approval from this example. The current reviewed package and cap are documented in the adjacent audit report. The provider reads an already configured `OPENAI_API_KEY`; never paste keys in a command, approval, report or chat.

```sh
python -m benchmarks.english_screen.packet /absolute/path/to/package.json \
  --execute-paid --output /approved/output --budget-usd APPROVED_CAP \
  --approval-record /absolute/path/to/approval.json
python -m benchmarks.english_screen.verification /absolute/path/to/package.json \
  --checkpoint /approved/output/checkpoint.json \
  --approval-record /absolute/path/to/approval.json
```

Runtime and source hashes are checked before dispatch. Reservations use integer nanoUSD, full output caps, a UTF-8 input bound and the Luna cache-write premium. They are written and fsynced before each call and never automatically refunded. This is a per-run dispatch bound at frozen prices, not an account-wide billing cap. Usage estimates account for reported cache writes and conservatively ignore read discounts; reconcile invoices separately.

Arms are counterbalanced by question hash. Exact identical judge requests share one verdict only within the same question. SDK retries are disabled. Errors, model-version drift, truncation, ambiguous verdicts and unresolved attempts halt the run. No manual retry/reconciliation path is included. A clean schedule-prefix checkpoint resumes; a pending/error checkpoint needs a new reviewed decision. The independent checker reconstructs requests, parsed answers, grades, sharing provenance, receipts, ledger and gates before resume or acceptance.

`packet` temporarily binds the paired-context dispatcher to the isolated runner; it is deliberately single-process, serial, and protected by a file lock. Do not invoke multiple packet operations concurrently in one interpreter. The independent verifier does not import the dispatcher or project database. The original historical runners are preserved.

## Decision

All 150 pairs must complete with no unresolved work. Require at least 10 net correct gains, at most two paired losses on stable controls, at least 56/58 controls correct in the candidate, and no net abstention decline across all 30 abstentions. The absolute control floor was added before new paid outputs to catch regressions affecting both arms.

A passing screen permits consideration of a separately approved full500 evaluation. A failing screen blocks broad promotion. Never selectively regrade, omit losses, retune mid-run, present this selected cohort as held out, or claim the supplement fixes reasoning, abstention or reference-label issues.
