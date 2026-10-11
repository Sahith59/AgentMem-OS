# Sarvam comparison v6 — cited date witnesses

The first complete v5 comparison scored memory24/24 vs original-control23/24.
Its unchanged replication scored memory23/24 vs control24/24: pooled47/48each.
There is no replicated accuracy gain. The memory loss wrongly authorized a
conditional callback after extraction fabricated an approval interpretation.

Separately, both runs admitted effective dates copied from observation metadata
and dates with no cited-text support. V6 changes **only local admission**:
every non-null valid_from/valid_until must have a literal ISO-date occurrence in
the cited, authorized original text. If any date lacks that witness, the whole
augmentation is rejected; all eligible originals survive unchanged. No guessed
normalization, date repair, extra model call or partial fact repair.

Models, extraction prompts, answer prompts, request construction, fixtures,
labels, schedules and limits stay identical to v5. The SQLite request can differ
because admission changes which facts survive. All extraction cost is charged.
Historical code/results remain frozen. This fifth campaign slot is at most
72calls, zero retries, ₹297.90830592 reservation, fixed105B/null/4096.

This guard is intentionally conservative and **is not semantic entailment**.
It rejects derived endpoints (such as the day before an explicit date), spelled
dates, or incompletely cited facts even when their meaning could be valid. A
date appearing in the wrong semantic role still passes. In particular, the
observed false conditional-approval fact would pass; a regression test preserves
that counterexample. Do not call dates/conditions/identity fully validated.

Offline replay of runs003/004 newly rejects8/5 augmentation sets respectively,
in addition to2existing structural rejections each. This is an evidence-admission
result, not a counterfactual answer score. The paid experiment must report actual
answer gains/losses and rejection costs. No product/novelty/all-language claim.

CLI: `python -m benchmarks.sarvam_comparison_v6 prepare|check|run|score --help`.
Same development-only and independent-review limits as v3. Freeze before use;
template approval is false. The five-run campaign authorizes only the final slot,
with no sixth run or automatic retries.
