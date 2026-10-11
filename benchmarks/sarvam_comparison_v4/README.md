# Sarvam comparison v4 — minimal provider schema compatibility

V3 live campaign run001 stopped on its second request: HTTP500 wraps upstream400 `Unimplemented keys: ["uniqueItems"]`. Receipt has no usage; no claim of zero cost.

V4 removes **only uniqueItems from the transmitted extraction JSON Schema**. Strict schema mode, schema name, model, prompts, source projection, schedules, labels, answer construction, local validation, rejection policy, cost accounting and no-retry stops remain identical to v3. Local duplicate-source rejection is retained and tested. Historical v1/v2/v3 files and checkpoints remain unchanged. The runner/scorer snapshot preserves historical package hash validation; this is a compatibility profile, not a new memory architecture.

72 maximum calls, two repeats over12 development renderings/four families, fixed sarvam-105b/null reasoning/temperature0/4096 cap. Reservation₹297.90830592. v3 README documents all research limitations. Provider acceptance of remaining keywords is untested until this run. Known-usage rejected extractions retain originals; infrastructure failures stop.

Commands: `python -m benchmarks.sarvam_comparison_v4 prepare|check|run|score --help`. Preparation is offline, approval defaults false. Runs require bound authorization; the founder's five-run campaign approves the next slot within its unchanged model/call/reservation limits. No retry of run001.
