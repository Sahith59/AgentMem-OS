# Sarvam final reporting instruction comparison

Read [REPORT.md](REPORT.md) for the new result and [SARVAM_HISTORY.md](SARVAM_HISTORY.md) for the complete recorded research history. [FAILURE_REVIEW.md](FAILURE_REVIEW.md) preserves the failures and paired changes. [NEXT_PHASE.md](NEXT_PHASE.md) is a proposal, not a completed result.

This is a development comparison on 144 saved histories (72 related English/Hindi/Telugu inputs across two models), each answered once per instruction. Only the first sentence differs. Models, reasoning settings, evidence, schema and 8,192-token caps remain fixed. No tool actions are rerun. Equal caps do not imply equal actual token consumption. No new memory engine, independent holdout, all-language reliability or English benchmark gain is claimed.

Verify `artifact-manifest.json` and every extracted byte hash. Extract into an empty directory and use new output paths:

```sh
mkdir /tmp/sarvam-wording-evidence
tar -xzf benchmarks/audits/sarvam-reporter-wording-2026-10-11/evidence.tar.gz -C /tmp/sarvam-wording-evidence
python -m benchmarks.sarvam_reporter_wording audit /tmp/sarvam-wording-evidence/phase --output /tmp/sarvam-wording-replay.json
python /tmp/sarvam-wording-evidence/phase/analysis/independent_check.py /tmp/sarvam-wording-evidence/phase /tmp/sarvam-wording-independent.json
```

These commands disable networking and do not read credentials or call a model. The outputs must match `phase/verification.json` and `phase/independent-aggregation.json`. Never restart the completed paid manifest. Source snapshots and expected labels are preserved separately from the generation inputs. Parent artifacts preserve the original actions and database evidence. Historical planning/status snapshots inside the archive are dated records, not the current state.

Assistant review is not independent native-language review. Unsupported-date findings, missing output, question echoes and reporting-status changes remain separate. The frozen screen and any negative outcome are retained. Publication/CI receipts in the dedicated workspace memory are completed after the evidence commit.
