# Corrected fresh Sarvam comparison v3

Read [VERIFICATION.md](VERIFICATION.md) for the completed clean-export check. Read [REPORT.md](REPORT.md), [FAILURE_REVIEW.md](FAILURE_REVIEW.md) and [NEXT_STEP.md](NEXT_STEP.md). This archive preserves the completed or stopped outcome, all attempted calls and missing outputs; never restart its manifest. The primary gate is unchanged by supplementary metrics. The bank has120related EN/HI/TE renderings of40scenarios and assistant-only language review. It is development research, not a held-out industry benchmark or a demonstrated memory-product advantage.

Verify the archive SHA256 and every extracted file against artifact-manifest.json. Extract to an empty directory. The following commands are offline; they neither read credentials nor call a model. Use new output paths.

```sh
mkdir /tmp/sarvam-v3-evidence
tar -xzf benchmarks/audits/sarvam-fresh-confirmation-v3-2026-10-13/evidence.tar.gz -C /tmp/sarvam-v3-evidence
python -m benchmarks.sarvam_fresh_confirmation.audit /tmp/sarvam-v3-evidence/phase --output /tmp/sarvam-v3-replay.json
python /tmp/sarvam-v3-evidence/phase/analysis/independent_check.py /tmp/sarvam-v3-evidence/phase /tmp/sarvam-v3-independent.json
python -m benchmarks.sarvam_fresh_cases_v3 /tmp/sarvam-v3-regenerated
```

The two audit outputs must equal phase/verification.json and phase/independent-aggregation.json. Generated runtime/gold JSON values must equal the frozen bank. Raw grade changes or interpreting assistant review as independent human validation are not part of this replay. The `unexecuted-screen/` directory preserves the blocked prospective comparison; it contains no live request manifest or results. Supplemental research/prototypes are offline only; no architectural gain follows from software tests.
