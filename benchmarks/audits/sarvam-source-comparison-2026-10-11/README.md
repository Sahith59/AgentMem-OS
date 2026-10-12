# Sarvam ordinary source comparison — evidence

Read [REPORT.md](REPORT.md) for measured outcomes, reviewed failures, costs and limitations. The frozen comparison uses two model profiles, 72 English/Hindi/Telugu renderings of 24 related scenarios, and ordinary SQLite with approval enforcement off/on. No memory engine or English benchmark is changed. Translations are correlated and independent native-language review is absent.

`evidence.tar.gz` preserves the original phase (including frozen setup and comparison requests, receipts, real SQLite databases, ledger, source, separate labels and evaluation contracts) and the new comparison report/analyses. Earlier preparation documents inside `phase/` are dated historical snapshots, including their formerly unstarted status. The completed execution and `report/` provide this run's result; never resume a completed manifest.

Verify the archive and all byte hashes in `artifact-manifest.json`, then extract to an empty directory. From this checkout, use new output paths:

```sh
mkdir /tmp/sarvam-comparison-evidence
tar -xzf benchmarks/audits/sarvam-source-comparison-2026-10-11/evidence.tar.gz -C /tmp/sarvam-comparison-evidence
python -m benchmarks.sarvam_ordinary_audit /tmp/sarvam-comparison-evidence/phase comparison --output /tmp/sarvam-comparison-replay.json
python -m benchmarks.sarvam_source_attribution audit /tmp/sarvam-comparison-evidence/phase --output /tmp/sarvam-attribution-replay.json
python -m benchmarks.sarvam_source_results /tmp/sarvam-comparison-evidence/phase /tmp/sarvam-comparison-replay.json /tmp/sarvam-attribution-replay.json --output /tmp/sarvam-summary-replay.json
python -m benchmarks.sarvam_source_action_replay /tmp/sarvam-comparison-evidence/phase /tmp/sarvam-comparison-replay.json --output /tmp/sarvam-backend-replay.json
```

Outputs must respectively match `report/verification.json`, `report/attribution.json`, `report/summary.json` and `report/backend-action-replay.json`. Paid-request replay and backend replay disable networking. No command above reads a credential or makes a model call. The last analysis replays observed tool calls under both approval policies; it does not observe new model answers or replace any original grade. Missing output, valid but incorrect answers, status-contract errors, desired state and actual unsafe effects remain separate.
