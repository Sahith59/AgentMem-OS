# Fresh Sarvam workflow confirmation — stopped, corrected package unexecuted

Read [REPORT.md](REPORT.md), [FAILURE_REVIEW.md](FAILURE_REVIEW.md) and [NEXT_PHASE.md](NEXT_PHASE.md). This is a failed/incomplete confirmation, not a new accuracy headline. The assistant found ambiguity in our question/gold interpretation and stopped after32/240episodes. All111calls are preserved, no retries. The32started workers completed before exit; execution.json was reconstructed transparently from immutable results. Fourteen frozen-contract misses include six outputs in the ambiguous family. No candidate or memory architecture is promoted.

The archive contains `phase/` (stopped v1), including original inputs, labels, reviews, all calls, meter and SQLite state, plus `revised-v2/` (corrected, frozen, unexecuted). V2 changes only12questions and runtime bindings and retains all120cases; it is development-exposed, not a holdout. Assistant language review is not independent native review. Historical snapshots and provisional review inputs in the archive describe their dated stage; REPORT.md and the completed audits take precedence.

Check archive SHA256 and every extracted file against artifact-manifest.json. Use an empty directory and new output filenames. The following operations are offline and read no credentials:

```sh
mkdir /tmp/sarvam-fresh-evidence
tar -xzf benchmarks/audits/sarvam-fresh-confirmation-2026-10-11/evidence.tar.gz -C /tmp/sarvam-fresh-evidence
python -m benchmarks.sarvam_fresh_confirmation.audit /tmp/sarvam-fresh-evidence/phase --output /tmp/sarvam-fresh-replay.json
python /tmp/sarvam-fresh-evidence/phase/analysis/independent_check.py /tmp/sarvam-fresh-evidence/phase /tmp/sarvam-fresh-independent.json
python -m benchmarks.sarvam_fresh_confirmation.study check /tmp/sarvam-fresh-evidence/revised-v2
python -m benchmarks.sarvam_fresh_cases_v2 /tmp/sarvam-fresh-regenerated-v2
```

The first two outputs must exactly match phase/verification.json and phase/independent-aggregation.json. Regenerated runtime.json/gold.json must match revised-v2/ byte-independent JSON values. Do not run a paid command as an audit, resume the stopped manifest or overwrite evidence. Independent parser implementation means a separate code path by the same assistant, not external validation. Protocols, budget carry, interruption receipts and known limitations are preserved.
