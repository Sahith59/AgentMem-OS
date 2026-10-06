"""Offline candidate: canonicalize source order before frozen v1 role checks.

Ordering is a renderer responsibility, not a model capability. Membership,
within-list uniqueness, and selected/rejected separation remain strict.
No paid runner imports this version until a separately frozen experiment.
"""

import json

from .evidence_focus import _unique_object, digest, validate_input
from .evidence_role_normalization import normalize as normalize_v1
from .guarded_focus import apply_guarded
from .luna_evidence_plan import canonical
from .luna_evidence_plan_v3 import ROLE_FIELDS


def normalize(value, raw):
    validate_input(value)
    data = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(data, dict):
        raise ValueError("Invalid plan object")
    known = {turn.id for turn in value.turns}
    reordered = []
    for field in ROLE_FIELDS:
        ids = data.get(field)
        if (
            not isinstance(ids, list)
            or any(not isinstance(i, str) for i in ids)
            or len(ids) != len(set(ids))
            or not set(ids) <= known
        ):
            raise ValueError("Invalid source membership or duplicate role ID")
        ordered = [turn.id for turn in value.turns if turn.id in ids]
        if ids != ordered:
            reordered.append(field)
        data[field] = ordered
    normalized, previous = normalize_v1(value, canonical(data))
    return normalized, dict(
        policy="source-order-qualification-wins-v2",
        changed=bool(reordered) or previous["changed"],
        reordered_fields=reordered,
        overlap_ids=previous["overlap_ids"],
        raw_sha256=digest(raw),
        normalized_sha256=digest(normalized),
        source_union_changed=False,
        semantic_correctness="NOT_CERTIFIED",
    )


def apply_normalized(value, raw, *, char_cap=40_000, max_focus_chars=4_000):
    normalized, audit = normalize(value, raw)
    candidate, report = apply_guarded(
        value, normalized, char_cap=char_cap, max_focus_chars=max_focus_chars
    )
    report["role_normalization"] = audit
    return candidate, report
