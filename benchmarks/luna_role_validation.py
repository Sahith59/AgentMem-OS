"""Label-side semantic checks. Never pass these labels to the inference provider."""

from .evidence_focus import semantic_review


def review(case, support, qualification, rejected, sufficiency):
    selected = set(support + qualification)
    if not case["id"].startswith("fresh_"):
        return dict(semantic_review(case, list(selected)), population="exposed_development")
    assignments = {t["id"]: "unclassified" for t in case["turns"]}
    for role, ids in [
        ("support", support),
        ("qualification", qualification),
        ("rejected", rejected),
    ]:
        assignments.update({tid: role for tid in ids})
    invalid_roles = [
        tid for tid, role in assignments.items() if role not in case["allowed_roles"][tid]
    ]
    capacity = case["capacity_exceeds_focus"]
    membership = (
        selected.issubset(assignments) and len(selected) <= 8
        if capacity
        else any(selected == set(option) for option in case["acceptable_evidence_sets"])
    )
    status_ok = sufficiency in case["allowed_sufficiency"]
    return dict(
        population="capacity_awareness" if capacity else "internal_validation",
        membership_pass=membership,
        sufficiency_pass=status_ok,
        invalid_role_ids=invalid_roles,
        **{"pass": membership and status_ok and not invalid_roles},
    )
