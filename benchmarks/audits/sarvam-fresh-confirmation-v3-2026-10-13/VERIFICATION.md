# Clean-export verification

Verified from a fresh `git archive` of commit `058c39f21cb6f6031fd2dc3dd7e431399ae6a15a`, without depending on the original experiment directories. [Machine-readable record](clean-export-verification.json).

- Archive SHA-256: `b0378b68360ca06901e9c0b0d864d1fc4151e739dcc1b44a6c3dfa5545f7dfc4`; all 3,348 extracted file hashes match.
- Primary request/receipt/SQLite replay exactly reproduces `verification.json`, including the failed gate and missing reporter slots.
- Separate raw-response parser and accounting reproduce `independent-aggregation.json`.
- Committed v3 builder regenerates the frozen runtime and gold values.
- Capped-actor handoff preparation reproduces the saved prospective request hashes without dispatch. The original failed result remains unchanged.
- All 234 focused Sarvam tests pass from the clean export. No provider request or credential read is part of these checks.

These checks establish reproducibility and software behavior. They do not turn assistant semantic review into independent human review, promote the candidate, establish a recovery accuracy gain, or validate a novel memory product. The subsequent publication commit adds this verification record only; CI checks the final PR head separately.
