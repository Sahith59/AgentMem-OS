# Native Sarvam structured-output control

The candidate failed its action compatibility gate. Original288episode comparison is closed with zero calls.20diagnostic requests are preserved; ordinary control executed both action tests, strict schema executed neither. This is one Telugu situation across two models, not a general accuracy claim. Read [REPORT.md](REPORT.md).

Implementation:9d09819. Evidence archive SHA256:`6859f8b783800549f5cf3cd6c26d6534c02c90be2ffa30e8793a459603a2cbc8`. 154 files are listed in artifact-manifest.json. They include both frozen protocols, raw requests/replies, actual SQLite files, labels, failed gate, unattempted comparison manifest, code snapshots and offline verification. No credentials, real customer data or account observations are included.

From a checkout containing the implementation, unpack into a fresh directory and replay without API calls:

```sh
mkdir /tmp/sarvam-format-evidence
 tar -xzf benchmarks/audits/sarvam-structured-control-2026-10-11/evidence.tar.gz -C /tmp/sarvam-format-evidence
PYTHONPATH=. python /tmp/sarvam-format-evidence/audit_control.py /tmp/sarvam-format-evidence /tmp/sarvam-format-replay.json
```

Use unused output paths. Networking is disabled by the audit. Check every extracted file against artifact-manifest.json. Same-code replay verifies integrity; it does not independently validate language semantics. Prior native-study budget accounting is carried forward; prior receipts are published in sarvam-native-study-2026-10-11.

Do not rerun the frozen paid manifests. The larger comparison's gate failed. The next design is in the archive's NEXT_PHASE.md and is not implemented or measured here. No English benchmark score or default runtime changed.
