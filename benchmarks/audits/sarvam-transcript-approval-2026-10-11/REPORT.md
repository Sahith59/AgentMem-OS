# Transcript finalization and ordinary approval controls — October 11, 2026

**This phase has two positive, limited results.** Both fixed Sarvam models returned usable correct final answers in the four-call transcript diagnostic. A separate ordinary SQLite approval guard blocked deliberately unapproved writes while preserving legitimate updates in offline tests. This is engineering and compatibility progress; no new broad accuracy score, memory advantage or industry novelty is established.

## Live final-answer result

The two saved successful action histories from the prior transport repair were preserved completely: original records, questions, assistant claims, native tool calls, actual tool receipts and tool definitions. They are now represented as JSON data with role labels in a system/user request that exposes no executable tools. The reporting instruction changes with this boundary. Source data round-trips exactly; no expected answer or hidden database field is inserted into the request.

| Model and unchanged profile | Prior native-history repair | New transcript JSON | New transcript schema |
|---|---|---|---|
| sarvam-105b, low reasoning | Empty final content in both formats | Correct date and confirmed status | Correct date and confirmed status |
| sarvam-105b-conversations, null reasoning | Correct in both formats | Correct date and confirmed status | Correct date and confirmed status |

All four new calls returned HTTP200 with valid usage and exact raw JSON answers matching the saved actual receipts. There were no tool requests, retries, rewritten outputs or repeated actions. The saved source databases stayed byte-identical. Labels, inputs, code, request hashes, budget and pass criteria were frozen before dispatch. Model IDs, profiles, temperature and8192output caps are unchanged.

These are two selected histories, each tested in two formats, not four independent workflows. The historical comparison is separated in time and changes the representation/reporting boundary. It cannot isolate an internal provider cause or establish a causal benchmark gain. The current within-diagnostic requests differ only by response_format; both formats succeed, so there is no measured schema accuracy advantage. The finalizer is currently scoped to complete native tool histories; read-only and no-tool refusal paths still need integration coverage before a broader run.

## Actual compute tradeoff

Base105B now uses2990/1700completion tokens, including2960/1670reported reasoning tokens, versus32for each prior empty reply. Its new requests take13.48/8.98seconds. Conversations uses41/40completion tokens and approximately0.93/0.96seconds. New prompt counts are1075base and931Conversations, versus864/733on the earlier path. Equal caps do not mean equal consumed compute or a cost-normalized gain. Reasoning details omitted by Conversations remain unknown, not zero. Full telemetry: compute-comparison.json.

## Ordinary approval backend

The research-only SQLite backend takes caller identity, time, trusted issuers and approvals from the application, not from model-generated text or grading labels. A new write must have a grant matching the tenant, customer, booking, operation, exact date and expected version, within its effective window and without revocation. Scope, version, approval, mutation and operation receipt are checked/written in one transaction. Tool arguments cannot override caller, clock or enforcement.

A previous identical completed request returns the prior receipt without another write, even after expiry/revocation; current state is reported separately from the original receipt. A conflicting payload with the same key is rejected. Both baseline and guarded controls retain identity, closed-booking, version and idempotency checks; only approval enforcement differs.

Frozen offline comparison:22cases×2controls=44real SQLite variants.

-14permission contrasts plus the reconstructed observed pending-approval attempt: unenforced control accepts all15; guarded control rejects all15 and leaves state unchanged.
-Three legitimate first writes succeed in both controls, including the two idempotency scenarios. The duplicate causes no extra write; conflicting key reuse is rejected.
-Closed/stale state, cross-customer access and read-only checks retain their correct behavior.
-34backend tests additionally check malformed/injected arguments, eight concurrent requests, current versus original receipt state, and rollback if saving the receipt fails.75focused tests pass in total.

The observed-pending reconstruction copies the prior model-selected booking/date/version/key and adds explicit synthetic approval metadata. It demonstrates that the backend can stop that kind of request; it is not an exact rerun of the original pipeline or proof the model avoids unauthorized attempts. There were no new model calls for this backend comparison. The guard is standard application authorization, not a novel memory system. The test application supplies identity and approval records; production authentication/issuer provisioning and a deployed customer workflow are not implemented.

## Evidence and spending

Four new API calls, zero errors or retries. Known usage estimate INR0.346440; retained accounting INR0.433053. Cumulative native research:4978calls, known INR59.691990, retained INR106.872785 within the standing INR6000ceiling. Historical unknown-usage reservations remain debited. These estimates are not verified invoices.

Raw requests/replies, usage, source histories, real SQLite, frozen labels/protocols and code are preserved. Offline audits reproduce all4model grades and44backend variants without networking or credential access. Ruff/diff checks pass. No English/default runtime changes. No independent native-language review or larger held-out model evaluation is claimed.

## Next decision

This removes two concrete blockers within the tested scope, but does not reopen either historical failed288-answer manifest. Next integrate the action phase, server guard and finalizer with explicit read-only/refusal paths and outcome semantics. Verify ordinary positive/negative flows before freezing a fresh larger matched comparison. Keep the same models and count all actual compute. Only residual failures against that stronger ordinary baseline can motivate a memory contribution. See NEXT_PHASE.md.
