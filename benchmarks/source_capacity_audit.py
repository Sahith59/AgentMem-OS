"""Evaluator-only whole-source capacity bounds; never a runtime selector.

``targets`` are evaluation annotations. The returned witnesses MUST NOT feed an
answerer or be described as predicted answer improvements. Full local windows
are a syntactic contract, not a certificate of semantic sufficiency.
"""

from agentmem_os.llm.evidence_packet import eligible_sources


def measure_capacity(snapshot, *, scope, as_of, present_ids, targets,
                     anchor_ids=None, neighbor_turns=1):
    """Find exact minimum union costs covering any/all annotated target IDs.

    Costs include all attribution, source bodies, framing and the separator
    appended to the baseline. Search only bundles that cover a target. Choosing
    a bundle covering the first uncovered target explores every minimal cover;
    shared source IDs are charged once. No relevance scores or answers enter.
    """
    if type(neighbor_turns) is not int or neighbor_turns < 0:
        raise ValueError("Invalid neighborhood")
    eligible = eligible_sources(snapshot, scope=scope, as_of=as_of)
    by_id = {t.id: t for t in snapshot.turns}
    allowed = {t.id for t in eligible}
    present, target = frozenset(present_ids), frozenset(targets)
    anchors = allowed if anchor_ids is None else frozenset(anchor_ids)
    if (not present <= allowed or not target <= by_id.keys()
            or not anchors <= by_id.keys() or len(target) > 12):
        raise ValueError("Unknown, ineligible or excessive evaluation input")
    if present & target:
        raise ValueError("Targets must be absent from certified baseline sources")
    sessions = {}
    for t in snapshot.turns:
        sessions.setdefault(t.session, {})[t.position] = t
    costs = {
        t.id: len(f"[{t.id} | {t.role} | observed {t.observed_at.isoformat()}]\n")
        + len(t.text) + 1 for t in eligible
    }
    prefix = "[ORIGINAL SOURCE EVIDENCE: relevance and completeness are unverified]\n"

    def cost(ids):
        return len(prefix) + 2 + sum(costs[i] for i in ids) if ids else 0

    bundles = []
    for anchor in sorted(anchors & allowed):
        t = by_id[anchor]
        session = sessions[t.session]
        positions = range(max(min(session), t.position - neighbor_turns),
                          min(max(session), t.position + neighbor_turns) + 1)
        if any(p not in session for p in positions):
            continue
        required = frozenset(session[p].id for p in positions)
        if not required <= allowed or not required & target:
            continue
        bundles.append((anchor, required - present))
    choices = {i: [b for b in bundles if i in b[1]] for i in target}

    def witness(ids, seeds):
        return dict(chars=cost(ids), source_ids=sorted(ids), anchor_ids=sorted(seeds),
                    targets_covered=sorted(target & ids))

    def key(ids, seeds):
        return cost(ids), tuple(sorted(ids)), tuple(sorted(seeds))

    any_best = min(bundles, key=lambda b: key(b[1], (b[0],)), default=None)
    any_cover = witness(any_best[1], (any_best[0],)) if any_best else None
    best, seen = None, set()

    def search(ids, seeds):
        nonlocal best
        state = ids, seeds
        if state in seen:
            return
        seen.add(state)
        if best is not None and cost(ids) > best[0][0]:
            return
        uncovered = target - ids
        if not uncovered:
            candidate = key(ids, seeds), witness(ids, seeds)
            if best is None or candidate[0] < best[0]:
                best = candidate
            return
        # Fewest candidate bundles first; stable tie-break, no source labels
        # beyond the explicitly evaluator-only target set.
        first = min(uncovered, key=lambda i: (len(choices[i]), i))
        for anchor, source_ids in choices[first]:
            search(ids | source_ids, seeds | {anchor})

    search(frozenset(), frozenset())
    return dict(
        status="EVALUATOR_ONLY_NOT_AN_ACCURACY_CEILING",
        targets=sorted(target), eligible_targets=sorted(target & allowed),
        unreachable_targets=sorted(i for i in target if not choices[i]),
        candidate_bundles=len(bundles), search_states=len(seen),
        any_cover=any_cover, all_cover=best[1] if best else None,
    )
