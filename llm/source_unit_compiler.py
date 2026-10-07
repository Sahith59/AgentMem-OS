"""Compile original-source quote/dependency units for OFFLINE PREVIEW ONLY.

This checks declared structure, not meaning. There is intentionally no semantic
approval input, model call, answer-packet adapter or automatic fallback here.
"""

import json
from dataclasses import asdict
from datetime import datetime

from .evidence_packet import SourceSnapshot, SourceTurn, digest, eligible_sources
from .source_unit_contract import (
    DEPENDENCY_KINDS,
    MAX_DEPENDENCIES,
    MAX_SPANS,
    Dependency,
    QuoteSpan,
    SourceUnit,
    UnresolvedDependency,
)


def _json(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # Keep one physical line even for Unicode line separators recognized by
    # common JSONL readers. json.loads restores the exact original codepoints.
    return text.replace("\x85", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def snapshot_digest(snapshot):
    """Bind order and all original source metadata/text, including future turns."""
    if (
        type(snapshot) is not SourceSnapshot
        or type(snapshot.turns) is not tuple
        or any(
            type(t) is not SourceTurn or type(t.observed_at) is not datetime for t in snapshot.turns
        )
    ):
        raise ValueError("Invalid snapshot shape")
    return digest(
        _json(
            dict(
                scope=snapshot.scope,
                turns=[
                    dict(
                        id=t.id,
                        session=t.session,
                        position=t.position,
                        role=t.role,
                        observed_at=t.observed_at.isoformat(),
                        text=t.text,
                    )
                    for t in snapshot.turns
                ],
            )
        )
    )


def _validate(snapshot, unit, question, scope, as_of, char_budget):
    eligible = eligible_sources(snapshot, scope=scope, as_of=as_of)
    if (
        type(question) is not str
        or not question.strip()
        or type(char_budget) is not int
        or char_budget < 0
        or type(unit) is not SourceUnit
        or type(unit.id) is not str
        or not unit.id.strip()
        or unit.snapshot_sha256 != snapshot_digest(snapshot)
        or unit.question_sha256 != digest(question)
        or type(unit.as_of) is not datetime
        or unit.as_of.isoformat() != as_of.isoformat()
    ):
        raise ValueError("Invalid unit binding or budget")
    if (
        any(
            type(v) is not tuple
            for v in (unit.spans, unit.root_span_ids, unit.dependencies, unit.unresolved)
        )
        or not 0 < len(unit.spans) <= MAX_SPANS
        or not 0 < len(unit.root_span_ids) <= len(unit.spans)
        or len(unit.dependencies) > MAX_DEPENDENCIES
        or len(unit.unresolved) > MAX_DEPENDENCIES
    ):
        raise ValueError("Invalid or unbounded unit shape")
    sources = {t.id: t for t in snapshot.turns}
    spans, by_source = {}, {}
    for span in unit.spans:
        if (
            type(span) is not QuoteSpan
            or type(span.id) is not str
            or not span.id.strip()
            or span.id in spans
            or type(span.source_id) is not str
            or span.source_id not in sources
        ):
            raise ValueError("Invalid or duplicate span/source ID")
        t = sources[span.source_id]
        if (
            span.source_sha256 != digest(t.text)
            or type(span.start) is not int
            or type(span.end) is not int
            or not 0 <= span.start < span.end <= len(t.text)
            or span.quote_sha256 != digest(t.text[span.start : span.end])
        ):
            raise ValueError("Invalid quote offsets or source/quote hash")
        spans[span.id] = span
        by_source.setdefault(t.id, []).append(span)
    for group in by_source.values():
        group.sort(key=lambda s: (s.start, s.end, s.id))
        if any(a.end > b.start for a, b in zip(group, group[1:])):
            raise ValueError("Overlapping or duplicate source ranges")
    if any(type(i) is not str or i not in spans for i in unit.root_span_ids) or len(
        set(unit.root_span_ids)
    ) != len(unit.root_span_ids):
        raise ValueError("Invalid or duplicate roots")
    edges, adjacency = set(), {i: [] for i in spans}
    for dep in unit.dependencies:
        if (
            type(dep) is not Dependency
            or type(dep.source_span_id) is not str
            or type(dep.required_span_id) is not str
            or type(dep.kind) is not str
            or dep.source_span_id not in spans
            or dep.required_span_id not in spans
            or dep.kind not in DEPENDENCY_KINDS
        ):
            raise ValueError("Invalid dependency")
        edge = dep.source_span_id, dep.required_span_id, dep.kind
        if edge in edges:
            raise ValueError("Duplicate dependency")
        edges.add(edge)
        adjacency[dep.source_span_id].append(dep.required_span_id)
    pending, visited = set(), set()

    def visit(node):
        if node in pending:
            raise ValueError("Cyclic dependency")
        if node in visited:
            return
        pending.add(node)
        for child in adjacency[node]:
            visit(child)
        pending.remove(node)
        visited.add(node)

    for node in unit.root_span_ids:
        visit(node)
    if visited != set(spans):
        raise ValueError("Orphaned quote span")
    unresolved = set()
    for issue in unit.unresolved:
        if (
            type(issue) is not UnresolvedDependency
            or type(issue.span_id) is not str
            or issue.span_id not in spans
            or type(issue.kind) is not str
            or issue.kind not in DEPENDENCY_KINDS
        ):
            raise ValueError("Invalid unresolved dependency")
        if (issue.span_id, issue.kind) in unresolved:
            raise ValueError("Duplicate unresolved dependency")
        unresolved.add((issue.span_id, issue.kind))
    return sources, spans, by_source, {t.id for t in eligible}


def compile_unit(snapshot, unit, *, question, scope, as_of, char_budget):
    """Return (preview, report), never an answer-ready or semantically approved unit.

    Malformed/unbound inputs raise ValueError. Declared unresolved dependencies,
    future sources or budget nonfit return an empty preview. Output quote tokens
    are JSON strings: decode before comparing receipt offsets to original text.
    Omission ranges cover only referenced sources, not the complete history.
    """
    sources, spans, by_source, allowed = _validate(
        snapshot, unit, question, scope, as_of, char_budget
    )
    unit_data = asdict(unit)
    unit_data["as_of"] = unit.as_of.isoformat()
    report = dict(
        schema="source-unit-compile-v1",
        unit_id=unit.id,
        scope=scope,
        snapshot_sha256=unit.snapshot_sha256,
        question_sha256=unit.question_sha256,
        as_of=as_of.isoformat(),
        unit_sha256=digest(_json(unit_data)),
        structural_integrity="PASS",
        semantic_completeness="NOT_CERTIFIED",
        answer_path_eligible=False,
        answer_accuracy="NOT_MEASURED",
        context_scope="DECLARED_SOURCE_IDS_ONLY",
        declared_roots=list(unit.root_span_ids),
        declared_dependencies=[asdict(d) for d in unit.dependencies],
        unresolved=[asdict(d) for d in unit.unresolved],
        char_budget=char_budget,
        required_chars=None,
        used_chars=0,
        receipts=[],
        emitted_ranges=[],
        omissions=[],
        preview_sha256=digest(""),
        reasons=[],
    )
    future = sorted(set(by_source) - allowed)
    if future:
        report["reasons"].append("future_source")
        report["future_source_ids"] = future
    if unit.unresolved:
        report["reasons"].append("unresolved_dependency")
    if report["reasons"]:
        report["status"] = "REFUSED"
        return "", report

    preview = (
        _json(
            dict(
                schema="source-unit-preview-v1",
                unit_id=unit.id,
                semantic_completeness="NOT_CERTIFIED",
                answer_path_eligible=False,
                context_scope="DECLARED_SOURCE_IDS_ONLY",
            )
        )
        + "\n"
    )
    receipts, emitted, omissions = [], [], []
    order = sorted(
        by_source,
        key=lambda i: (sources[i].observed_at, sources[i].session, sources[i].position, i),
    )
    for sid in order:
        t = sources[sid]
        metadata = dict(
            source_id=sid,
            session=t.session,
            position=t.position,
            role=t.role,
            observed_at=t.observed_at.isoformat(),
            source_sha256=digest(t.text),
        )
        preview += _json(dict(kind="source", **metadata)) + "\n"
        groups = []
        for span in by_source[sid]:
            if groups and groups[-1][-1].end == span.start:
                groups[-1].append(span)
            else:
                groups.append([span])
        cursor = 0

        def omit(start, end):
            if start == end:
                return ""
            row = dict(kind="omitted", source_id=sid, start=start, end=end)
            omissions.append(dict(row, sha256=digest(t.text[start:end])))
            return _json(row) + "\n"

        for group in groups:
            lo, hi = group[0].start, group[-1].end
            preview += omit(cursor, lo)
            text = t.text[lo:hi]
            prefix = _json(dict(kind="quote", source_id=sid, start=lo, end=hi))[:-1]
            prefix += ',"text":'
            payload = _json(text)
            start = len(preview) + len(prefix)
            preview += prefix + payload + "}\n"
            fragment = dict(
                source_id=sid,
                start=lo,
                end=hi,
                sha256=digest(text),
                payload_start=start,
                payload_end=start + len(payload),
                encoding="JSON_STRING",
            )
            emitted.append(fragment)
            for span in group:
                receipts.append(
                    dict(
                        metadata,
                        span_id=span.id,
                        start=span.start,
                        end=span.end,
                        quote_sha256=span.quote_sha256,
                        payload_start=start,
                        payload_end=start + len(payload),
                        decoded_start=span.start - lo,
                        decoded_end=span.end - lo,
                        encoding="JSON_STRING",
                    )
                )
            cursor = hi
        preview += omit(cursor, len(t.text))
    report["required_chars"] = len(preview)
    if len(preview) > char_budget:
        report.update(status="REFUSED", reasons=["budget_nonfit"])
        return "", report
    report.update(
        status="PREVIEW_ONLY",
        used_chars=len(preview),
        receipts=receipts,
        emitted_ranges=emitted,
        omissions=omissions,
        preview_sha256=digest(preview),
    )
    return preview, report
