"""Opt-in source-presence selection with atomic original-position neighborhoods.

No model upgrades, generated summaries or answer labels. Syntactic neighborhood
completion is auditable; semantic sufficiency remains unknown.
"""

import math

from .evidence_packet import RetrievalHit, digest, eligible_sources
from .hybrid_source_retrieval import rank
from .source_presence import map_presence


def select(snapshot, hits, baseline, *, scope, as_of, use_presence=True,
           char_budget=40000, extra_budget=4000, max_anchors=8, neighbor_turns=1,
           allocation="greedy"):
    if (type(baseline) is not str or type(use_presence) is not bool
            or any(type(v) is not int or v < 0 for v in
                   (char_budget, extra_budget, max_anchors, neighbor_turns))
            or len(baseline) > char_budget or allocation not in ("greedy", "joint")
            or (allocation == "joint" and max_anchors > 8)):
        raise ValueError("Invalid baseline or policy")
    eligible = eligible_sources(snapshot, scope=scope, as_of=as_of)
    by_id = {t.id: t for t in snapshot.turns}
    allowed = {t.id for t in eligible}
    ledger = map_presence(snapshot, baseline, scope=scope, as_of=as_of)
    present = {r["id"] for r in ledger["receipts"] if not r["future"]} if use_presence else set()
    ranked, seen = [], set()
    for h in hits:
        if (type(h) is not RetrievalHit or h.source_id not in by_id
                or h.source_sha256 != digest(by_id[h.source_id].text)
                or not math.isfinite(h.score) or h.score < 0
                or type(h.tie_order) is not int or h.tie_order < 0):
            raise ValueError("Unbound retrieval hit")
        if h.source_id in allowed and h.source_id not in seen:
            ranked.append(h)
            seen.add(h.source_id)
    ranked.sort(key=lambda h: (-h.score, h.tie_order, h.source_id))
    anchors = [h for h in ranked if h.source_id not in present][:max_anchors]
    by_session = {}
    for t in snapshot.turns:
        by_session.setdefault(t.session, {})[t.position] = t
    chosen, admitted = set(), []
    allowance = max(0, min(extra_budget, char_budget - len(baseline)) - 2)
    block_prefix = "[ORIGINAL SOURCE EVIDENCE: relevance and completeness are unverified]\n"

    def source_header(t):
        return f"[{t.id} | {t.role} | observed {t.observed_at.isoformat()}]\n"

    def render(ids):
        if not ids:
            return "", []
        text = block_prefix
        receipts = []
        for t in sorted((by_id[i] for i in ids),
                        key=lambda t: (t.observed_at, t.session, t.position)):
            header = source_header(t)
            start = len(text) + len(header)
            text += header + t.text + "\n"
            receipts.append(dict(id=t.id, session=t.session, position=t.position,
                                 role=t.role, observed_at=t.observed_at.isoformat(),
                                 sha256=digest(t.text), start=start, end=start + len(t.text)))
        return text, receipts

    bundles, valid = [], []
    for h in anchors:
        t = by_id[h.source_id]
        session = by_session[t.session]
        positions = range(max(min(session), t.position - neighbor_turns),
                          min(max(session), t.position + neighbor_turns) + 1)
        missing_positions = [i for i in positions if i not in session]
        required = [session[i].id for i in positions if i in session]
        row = dict(anchor=t.id, required=required, baseline_reused=sorted(set(required) & present),
                   missing_positions=missing_positions)
        if missing_positions:
            row["status"] = "position_gap"
        elif set(required) - allowed:
            row["status"] = "future_neighbor"
        else:
            valid.append(len(bundles))
        bundles.append(row)

    optimization = None
    if allocation == "joint":
        # At most eight frozen anchors: exact subset enumeration, no model or
        # answer-dependent utility. Neighbors are costs, not extra rank votes.
        try:
            math.fsum(anchors[i].score for i in valid)
        except OverflowError as error:
            raise ValueError("Nonfinite allocation utility") from error
        costs = {i: len(source_header(by_id[i])) + len(by_id[i].text) + 1 for i in allowed}

        def cost(ids):
            return len(block_prefix) + sum(costs[i] for i in ids) if ids else 0

        best_key, best_seed, best_covered = None, (), ()
        packets = set()
        for mask in range(1 << len(valid)):
            seed = tuple(i for bit, i in enumerate(valid) if mask & (1 << bit))
            ids = frozenset(i for j in seed for i in bundles[j]["required"] if i not in present)
            packets.add(ids)
            size = cost(ids)
            if size > allowance:
                continue
            covered = tuple(i for i in valid if set(bundles[i]["required"]) <= present | ids)
            utility = math.fsum(anchors[i].score for i in covered)
            key = (-utility, size, covered, seed, tuple(sorted(ids)))
            if best_key is None or key < best_key:
                best_key, chosen, best_seed, best_covered = key, set(ids), seed, covered
        optimization = dict(
            policy="exact-complete-bundle-rrf-sum-v1", utility=-best_key[0],
            subset_count=1 << len(valid), distinct_enumerated_source_sets=len(packets),
            seed_anchor_ids=[anchors[i].source_id for i in best_seed],
            covered_anchor_ids=[anchors[i].source_id for i in best_covered],
            rendered_block_chars=best_key[1],
        )
        for i in valid:
            row = bundles[i]
            if i in best_covered:
                row["status"] = "admitted"
                admitted.append(row["anchor"])
            elif cost(set(row["required"]) - present) > allowance:
                row["status"] = "bundle_budget_nonfit"
            else:
                row["status"] = "allocation_not_selected"
    else:
        for i in valid:
            row = bundles[i]
            new_ids = set(row["required"]) - present
            if len(render(chosen | new_ids)[0]) > allowance:
                row["status"] = "bundle_budget_nonfit"
            else:
                chosen |= new_ids
                admitted.append(row["anchor"])
                row["status"] = "admitted"
    block, receipts = render(chosen)
    offset = len(baseline) + 2 if block else len(baseline)
    packet = baseline + ("\n\n" + block if block else "")
    report = dict(
        schema="source-aware-selection-v1", scope=scope, as_of=as_of.isoformat(),
        policy="presence-aware" if use_presence else "matched-no-presence-control",
        baseline_sha256=digest(baseline), candidate_sha256=digest(packet),
        baseline_chars=len(baseline), total_chars=len(packet), block_offset=offset,
        total_char_budget=char_budget, extra_budget=extra_budget,
        ranked_sources=len(ranked), candidate_hits=[h.__dict__ for h in anchors],
        skipped_present=[h.source_id for h in ranked if h.source_id in present],
        admitted_anchors=admitted, bundles=bundles, receipts=receipts, presence=ledger,
        semantic_completeness="NOT_CERTIFIED", answer_accuracy="NOT_MEASURED",
    )
    if optimization is not None:
        report["allocation"] = optimization
    return packet, report


def supplement(snapshot, question, baseline, *, scope, as_of, encoder=None,
               positive_lexical_only=False, **policy):
    hits = rank(snapshot, question, scope=scope, as_of=as_of, encoder=encoder,
                positive_lexical_only=positive_lexical_only)
    text, report = select(snapshot, hits, baseline, scope=scope, as_of=as_of, **policy)
    report["ranking"] = ("dense-positive-lexical-rrf-k60" if positive_lexical_only
                         else "existing-dense-lexical-rrf-k60")
    return text, report
