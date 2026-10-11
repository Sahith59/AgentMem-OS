# Native Sarvam study — October 11, 2026

This is an evidence-first model/API investigation using synthetic records and ordinary SQLite. It does not use AgentMem extraction or a new memory architecture. Read [REPORT.md](REPORT.md) for the completed results. It distinguishes strict task success, formatting failures, actual tool outcomes and known-rate cost estimates.

The bank has700renderings of200parameter rows across10capability families. An input audit finds637distinct visible requests; the reserved split has6inputs also in discovery. A separate sensitivity view removes repeats without changing the primary results. English/Hindi/Telugu and mixed-source text are covered. These are correlated templates, not700independent customer workflows or a general Sarvam accuracy benchmark. The reserved split shares constructions with discovery. Native text is assistant-reviewed, not independently reviewed by qualified annotators.

## Offline reproduction

`evidence.tar.gz` contains synthetic inputs, separate gold, frozen configurations/source, exact requests and raw provider responses, actual test databases, scores, usage ledger and source-review notes. `artifact-manifest.json` binds its SHA256 and every contained file. No key, authorization header or private dashboard report is included. Extraction creates a new directory; never extract over an existing run.

From the repository root, after extracting into a new path named `native-study`:

```sh
python -m benchmarks.sarvam_native_study.verify native-study native-study/bank discovery discovery-continuation-001 attribution confirmation
python native-study/audit_receipts.py native-study
```

Both commands block socket operations. They verify saved evidence and make no model calls. The first rebuilds tool execution in temporary SQLite stores; the second checks the complete dispatch inventory, accounting and excluded setup. They are mechanical evidence checks, not independent native-language validation or provider-signed attestations.

To recompute aggregate scores without overwriting saved artifacts:

```python
import json
from pathlib import Path
from benchmarks.sarvam_native_study.analyze import analyze
root = Path("native-study")
replayed = analyze(root)
assert replayed == json.loads((root / "analysis.json").read_text())
```

The paid setup, runner, diagnostic and paced commands are deliberately separate. Running them consumes API credits and requires a new concrete authorized scope. An old manifest or published approval is not permission for another dispatch.

## Interpretation

Original failures remain in denominators. Continuation runs only previously unattempted jobs. Null/low profiles in the diagnostic have the same8192output ceiling but different realized compute. Main baseline uses null/4096. Tools are ordinary read/update operations, with real local state verification; no customer or production backend is touched. SQL adds a read roundtrip and uses prompt-based JSON with tools, so arm differences are end-to-end setup differences, not isolated storage effects. See the complete protocol, reports and follow-up criteria in the archive.

Model IDs and request settings are recorded, but the provider does not supply an immutable model-weight snapshot in these artifacts. A new live run may differ; offline replay verifies the saved run. No claim that the benchmark is independent of all prior model training or representative of real deployments is made.

Completed: 2,800 main episodes, 264 diagnostic episodes, 12 excluded setup requests; 4,928 total API requests, zero retries, five original rate-limit failures preserved. Known usage estimate INR 58.280115; retained accounting including headroom/unknown usage INR 86.675931. The live campaign has ended.
