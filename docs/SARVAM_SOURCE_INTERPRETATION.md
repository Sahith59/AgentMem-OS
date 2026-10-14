# Sarvam saved-history source interpretation screen

Completed once:720/720 calls. The candidate increases some factual counts but fails reliability gates and is not promoted. [Results and replay package](../benchmarks/audits/sarvam-source-interpretation-2026-10-13/README.md). Historical studies are unchanged.

`benchmarks/sarvam_source_interpretation.py` prepares a prospective, reporter-only comparison over all 240 frozen v3 native histories. It sends three fixed reporting views per history: chronological A, lossless grouped B, and grouped C with only nonempty actor prose withheld. The same `sarvam_evidence_ablation` renderer, model profiles, reporter instruction, schema and 8,192-token cap are used for every arm. No actor, tool or backend action is executed.

Preparation takes the frozen v3 parent, the completed two-call diagnostic's `completion.json`, and an existing new study directory containing reviewed `PROTOCOL.md`, `budget.json`, `pacing-control.json` and `source-preflight.json`:

```text
python -m benchmarks.sarvam_source_interpretation prepare STUDY_ROOT --parent V3_ROOT --prior-completion DIAGNOSTIC_COMPLETION
python -m benchmarks.sarvam_source_interpretation check STUDY_ROOT
python -m benchmarks.sarvam_source_interpretation run STUDY_ROOT --allow-paid
python -m benchmarks.sarvam_source_interpretation audit STUDY_ROOT --output NEW_AUDIT_JSON
```

Only the final two commands can follow preparation. `run` has no restart or retry path: it writes an exclusive start marker, uses fixed four-request batches and four workers with 90-RPM pacing, and stops before a new batch after two cumulative infrastructure errors or one fatal validation/budget error. The old v3 and blocked 108-call screen remain unchanged. Gold is copied and hashed as opaque bytes during preparation and execution; only offline audit parses it. Preparation saves all 240 source records, paired parent records, 239 original candidate requests, the capped actor's original request and receipt, omission ledgers, and hashes for each of 720 planned requests. The capped history is reconstructed from that actor receipt, never from the later diagnostic answers.

Audit replays recorded requests, responses, per-call usage and meter charges. It retains all 720 planned slots if execution stops early. It reports raw and receipt-projected status separately, date-plus-clarification accuracy, wrong definite dates, false abstentions, invalid finals, cost, latency, observed dispatch order, and paired A/B/C contrasts. C versus B on the predeclared 184 actor-prose-treated histories is primary; 56 B=C no-treatment histories are reported separately. Scenario-cluster bootstrap intervals resample all 40 source scenarios. The mechanical forward gate stays distinct from independent semantic review, and injected offline transport cannot pass the live-transport gate.

The study is a development-exposed synthetic reporting comparison with unequal actual input-token counts. A gain would support this reporting-boundary intervention on the fixed histories; it would not establish a new memory architecture, a production workflow result, or the unique cause of an actor-prose effect. The frozen protocol and preserved raw receipts are the source of truth for the actual phase.
