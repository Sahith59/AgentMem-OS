"""Conservative focus abstention. No source deletion, new model or intent regex."""

from .luna_evidence_plan_v3 import apply_plan, parse_plan


def apply_guarded(value, raw, *, char_cap=40_000, max_focus_chars=4_000):
    parsed = parse_plan(value, raw)
    candidate, report = apply_plan(value, raw, char_cap=char_cap, max_focus_chars=max_focus_chars)
    roles = {t.id: t.role for t in value.turns}
    suspect = [tid for tid in parsed["qualification_turn_ids"] if roles[tid] == "assistant"]
    # Do not strip a selected qualifier from a claim. Abstain from the whole
    # optional focus intervention, retaining ALL baseline sources and attribution.
    if suspect and parsed["sufficiency"] == "uncertain":
        from .evidence_focus import digest

        candidate = value.packet
        report.update(
            status="UNCHANGED_UNCERTAIN_ASSISTANT_QUALIFICATION",
            receipts=[],
            candidate_sha256=digest(candidate),
        )
    report.update(
        eligibility_policy="guarded-focus-v1",
        blocked_qualification_ids=suspect if parsed["sufficiency"] == "uncertain" else [],
        selection_correctness="NOT_CERTIFIED",
    )
    return candidate, report
