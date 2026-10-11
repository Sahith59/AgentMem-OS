# Next task: make extraction failure measurable without losing original evidence

Status: design for the next offline engineering task, **not an implemented fix or
paid-run authorization**. Keep the stopped v1/v2 packages and receipts immutable.

The immediate research question remains: does extracted memory improve answers over
the same model with all authorized original records? We have not answered it. Do not
start a custom memory engine from this format failure alone.

1. Specify the boundary between source metadata and semantic facts. Store source
   IDs, tenant/customer authorization, sessions and observation timestamps directly
   from input records. The model must not regenerate the security scope. Define
   effective dates from source meaning separately; observation time is not a default
   effective date. Preserve conditions and requested versus confirmed status.
2. Prepare a schema-constrained request using the currently documented `json_schema`
   mode, retaining the same model. Keep the local parser as an independent gate for
   real dates, intervals, citations and allowed keys. Verify provider-supported
   keywords separately; do not silently weaken the schema or fall back to JSON mode.
   Schema correctness is not semantic correctness.
3. Build a separate admission result: accepted facts, rejected extraction, or
   infrastructure failure. A rejected augmentation must never be promoted as a
   successful extraction or silently repaired. Retain all authorized original records
   and the failed request's cost. Derive authorization/cutoffs before any serving path.
4. Test a source-only answer path with empty admitted facts after extraction rejection.
   It should use the exact originals the control sees, report that augmentation was
   rejected, and count its extra extraction cost. This is a reliability baseline,
   not a memory-gain claim. Transport/auth/budget/unknown-usage failures still stop.
5. Freeze a revised comparison protocol prospectively. Let the independent control
   run without depending on valid extraction; label rejection and source-only fallback
   separately and report coverage across all intended cases. Compare unchanged
   full-history prompts/model with the candidate system; disclose any extraction
   prompt/schema/policy changes. Separate ablations are needed to assign causality.

Offline acceptance cases must include the saved timestamp failure, impossible and
reversed dates, missing/forged citations, duplicate keys, unauthorized records,
future records, malformed output, empty legitimate memory, conditional permission,
and changed facts with future effective dates. Test interrupted execution, reservation
before dispatch, unknown usage, no automatic resend and truthful incomplete scoring.
Gold labels must remain outside extraction. Synthetic fixtures must be identified
as such, and tests must not equate parser success with faithful meaning.

Before another paid request: finish implementation and review, reproduce the saved
failure without a network, freeze exact requests/call cap/cost/stop rules, and obtain
approval for that changed scope. There is no need to reload or share the key now.
Do not repeat single probes indefinitely; prepare a bounded development comparison
that can reveal control success, augmentation harm and extraction rejection.

If the ordinary control already succeeds, use it. If memory helps, next validate on
fresh families and a matched-total-compute control before claiming an architectural
gain. Public reproducibility and relevant Sarvam-supported language coverage remain
required; English/Hindi/Telugu development renderings are only the starting slice.
