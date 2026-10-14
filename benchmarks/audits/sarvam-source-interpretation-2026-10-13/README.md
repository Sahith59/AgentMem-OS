# All-history source-interpretation comparison

Completed720 reporter calls on240 saved histories. C increases date-plus-clarification counts from104→105/120 base and96→103/120 Conversations versus chronological A, but fails the predefined reliability gate and is not promoted. Read [REPORT.md](REPORT.md), [decision.json](decision.json), [the frozen protocol](PROTOCOL.md), and [the next offline proposal](NEXT_STEP.md).

The study is closed to inference. Do not remove started markers or rerun paid dispatch. No actor/tool/backend action was repeated; older studies remain unchanged. Models, temperature, instructions and output ceilings were fixed, but actual token use differs. Cases are synthetic, related and development-exposed, with assistant-only language review.

## Offline replay

Verify `evidence.tar.gz` against `artifact-manifest.json`, then check every extracted file against its recorded SHA256. The archive contains3459 files under `phase/`, including frozen source, labels, all720 request/response receipts, results, settled meter, reviews and analysis. It excludes credentials and the private founder-file inventory.

Use an empty directory and new output paths. These commands send no API request and need no key:

```sh
mkdir /tmp/sarvam-source-evidence
tar -xzf benchmarks/audits/sarvam-source-interpretation-2026-10-13/evidence.tar.gz -C /tmp/sarvam-source-evidence
python -m benchmarks.sarvam_source_interpretation audit /tmp/sarvam-source-evidence/phase --output /tmp/sarvam-source-primary.json
python /tmp/sarvam-source-evidence/phase/analysis/independent_raw_audit.py /tmp/sarvam-source-evidence/phase > /tmp/sarvam-source-independent.json
python /tmp/sarvam-source-evidence/phase/analysis/verify_comparison.py /tmp/sarvam-source-evidence/phase /tmp/sarvam-source-comparison.json
```

The first two JSON outputs must equal the archived primary and independent audits. The final command checks the two archived audits against one another. The separate semantic-review verifier checks all720 reviewed job IDs, receipt/result bindings and agreement with the audit; its timestamp is expected to differ on replay. Review is not independent human/native-language validation.

The earlier complete844-call native study is published in the [v3 archive](../sarvam-fresh-confirmation-v3-2026-10-13/README.md). This package includes all original paired histories and the exact reporter requests or capped-actor reconstruction needed for the new comparison. Its hashes support local integrity and reproduction; they do not provide a provider signature.

The new module's local six-test suite passed before dispatch. One test depends on the prior local study fixture and self-skips outside that environment; public artifact replay is a separate requirement, not replaced by that fixture-dependent test. Publication verification is recorded in `VERIFICATION.md` after clean export.
