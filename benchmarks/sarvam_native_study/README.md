# Native Sarvam capability study

Research harness only: no AgentMem extraction or memory architecture. Two native model IDs,
full original records versus ordinary SQLite tools. Strict state-based grading, durable
cost reservations and immutable requests/receipts; no provider retries.

```sh
python -m benchmarks.sarvam_native_study.setup /path/to/campaign
python -m benchmarks.sarvam_native_study.dataset /path/to/campaign/bank
python -m benchmarks.sarvam_native_study.runner /path/to/campaign /path/to/campaign/bank discovery
python -m benchmarks.sarvam_native_study.score /path/to/campaign /path/to/campaign/bank discovery
```

Setup and runner make paid calls. Only run under an explicitly authorized scope; code ceilings
are not permission. Credential is privately parsed from `~/.config/agentmem-os/sarvam.env`.
No automatic retries or checkpoint resends. Existing output directories cause refusal.
The scorer is offline and never sends gold to inference. Raw final responses are preserved.

Bank: 700 renderings, 200 parameterized cases, ten capability families. Split grouping IDs
are not independent templates. Native sentences are assistant-reviewed only. No real-user,
all-language, full-context-window, voice, independent-native-review or product claim.
Test-backend success is distinct from final-answer correctness. Costs differ across arms.
The baseline uses null reasoning; reasonable reasoning-enabled diagnostics remain necessary
before general claims about model limitations. See the campaign protocol and complete results.
