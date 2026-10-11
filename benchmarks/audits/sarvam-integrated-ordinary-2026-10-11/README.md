# Integrated ordinary Sarvam baseline — evidence

Read [REPORT.md](REPORT.md). Twenty-four live setup episodes preserve every current date and stored state; raw reports are correct in 21/24 and strict tasks in 19/24. The frozen gate fails. A separate offline receipt-status projection produces 24/24 correct application reports and 22/24 strict tasks on the same histories; this is not a fresh model score. No larger run started.

`evidence.tar.gz` contains frozen source, protocol, runtime cases and separate labels, immutable API requests/receipts, real SQLite databases, usage ledger, original verification, and separate projection output. `artifact-manifest.json` records raw byte hashes/sizes and archive hash. Six synthetic situations are not a language benchmark. Source review and every strict miss are documented.

From a clean checkout, extract to an empty directory and use new output paths:

```sh
mkdir /tmp/sarvam-integrated-evidence
 tar -xzf benchmarks/audits/sarvam-integrated-ordinary-2026-10-11/evidence.tar.gz -C /tmp/sarvam-integrated-evidence
python -m benchmarks.sarvam_native_controls.integrated_audit /tmp/sarvam-integrated-evidence --output /tmp/sarvam-integrated-recheck.json
python -m benchmarks.sarvam_receipt_reporting /tmp/sarvam-integrated-evidence --output /tmp/sarvam-projection-recheck.json
```

Compare those outputs exactly with `verification.json` and `receipt-projection.json` inside the archive. Both audits forbid networking and never read a credential. Do not resume the completed paid manifest. Original raw results and failed gate remain authoritative for the live phase.
