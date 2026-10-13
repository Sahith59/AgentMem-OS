# Observed capped-actor recovery diagnostic

**One observed English recovery, two reporter variants, both correct.** No actor or backend action was repeated. Added token-cost estimate ₹0.566850; the earlier 240-episode scores and failed gate remain unchanged. Read [REPORT.md](REPORT.md), [the frozen protocol](PROTOCOL.md), [independent assistant review](semantic-review.json) and [the next unexecuted proposal](NEXT_STEP.md).

The diagnostic is closed to inference. Its durable started/claim markers must never be removed to restart it. Offline replay uses no key and sends no request. Software test transports are explicitly labelled as injected, separate from the actual normal provider dispatch recorded here.

Verify `evidence.tar.gz` and every extracted file against `artifact-manifest.json`, then extract into an empty directory. Use new output paths:

```sh
mkdir /tmp/sarvam-cap-evidence
tar -xzf benchmarks/audits/sarvam-capped-recovery-2026-10-13/evidence.tar.gz -C /tmp/sarvam-cap-evidence
python -m benchmarks.sarvam_handoff_diagnostic audit /tmp/sarvam-cap-evidence/phase --output /tmp/sarvam-cap-audit.json
python /tmp/sarvam-cap-evidence/phase/analysis/independent_check.py /tmp/sarvam-cap-evidence/phase /tmp/sarvam-cap-independent.json
```

The outputs must equal `phase/audit.json` and `phase/independent-aggregation.json`. The snapshot supports all 240 eligibility/routing decisions and exact replay of the selected native receipt plus both new reporter calls. To replay all 844 historical parent calls, use the separately published [v3 archive](../sarvam-fresh-confirmation-v3-2026-10-13/README.md); that full replay was verified before this diagnostic.

The raw source and labels are synthetic and development-exposed. Review is assistant-only; independent human/native review is absent. No broad accuracy, multilingual memory, latency distribution, production readiness or novel-product claim follows. Caller-provided artifact hashes are integrity links, not cryptographic proof of a provider's identity.

[Clean-export verification](VERIFICATION.md) reproduces both audits, all315artifact hashes and240focused tests.

The current [NEXT_STEP.md](NEXT_STEP.md) clarifies one inference in the original proposal retained inside the archive: failure of both interventions would not identify the cause of the remaining errors. This documentation correction changes no requests, labels, results or archive bytes.
