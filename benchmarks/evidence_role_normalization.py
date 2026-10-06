"""Versioned qualification-wins normalization; never rewrite a paid checkpoint.

Only cross-list support/qualification overlap is recoverable. V3 already directs
mixed evidence into qualification. Every ID and the source-ordered union stay
unchanged; all other schema/source violations still fail validation.
"""

import json

from .evidence_focus import _unique_object, digest
from .guarded_focus import apply_guarded
from .luna_evidence_plan import canonical
from .luna_evidence_plan_v3 import parse_plan


def normalize(value, raw):
    data = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(data, dict):
        raise ValueError("Invalid plan object")
    for key in ("support_turn_ids", "qualification_turn_ids"):
        ids = data.get(key)
        if (
            not isinstance(ids, list)
            or any(not isinstance(i, str) for i in ids)
            or len(ids) != len(set(ids))
            or ids != [t.id for t in value.turns if t.id in ids]
        ):
            raise ValueError("Invalid role list; only cross-list overlap is recoverable")
    overlap = set(data["support_turn_ids"]) & set(data["qualification_turn_ids"])
    data["support_turn_ids"] = [i for i in data["support_turn_ids"] if i not in overlap]
    normalized = canonical(data)
    parse_plan(value, normalized)  # unknown/rejected overlaps, ordering, cap, types remain strict
    return normalized, dict(
        policy="qualification-wins-v1",
        changed=bool(overlap),
        overlap_ids=[t.id for t in value.turns if t.id in overlap],
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
