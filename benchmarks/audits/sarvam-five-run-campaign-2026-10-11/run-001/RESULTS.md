# Campaign run 001 — schema compatibility failure

Fixed frozen v3 package bf04d07d… attempted two calls and stopped: first full-history answer completed, first extraction HTTP500 contains upstream400: `Grammar error: Unimplemented keys: ["uniqueItems"]`. One completed answer, zero extractions, zero paired comparisons; incomplete, no headline accuracy. Second receipt has no usage; retain its full reservation and do not claim it cost zero. Raw response remains in live-run/checkpoint.json. No retry or resume.

Decision before run002: version a minimal provider-schema compatibility change removing uniqueItems from transmitted schema only. Preserve strict schema mode, prompts, sources, answer settings, schedules and all local validation, including duplicate citation rejection. New frozen package, fresh independent run, uses second authorized slot. Do not treat API syntax repair as model accuracy gain.
