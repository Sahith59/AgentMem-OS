"""Conservative whole-source mapping for the known historical packet renderers.

This reconstructs identity only when original scoped source records disambiguate
the complete rendered text. It is not semantic deduplication or fact entailment.
"""

import re
from collections import defaultdict

from .evidence_packet import digest, eligible_sources

_DATE_LINE = re.compile(r"(?m)^\[\d{4}/\d{2}/\d{2} \([A-Za-z]{3}\) \d{2}:\d{2}\]")
_ROLE_LINE = re.compile(r"(?m)^(?:USER|ASSISTANT): ")
_MARKERS = (
    "<[SEMANTIC MEMORY]>", "</[SEMANTIC MEMORY]>",
    "<[RECENT TURNS]>", "</[RECENT TURNS]>",
    "[ADDITIONAL SOURCE EVIDENCE]", "[source ",
)


def map_presence(snapshot, baseline, *, scope, as_of):
    """Return offset-bound, unique source matches; uncertainty never means present.

    Only time-eligible records participate in matching and ambiguity checks.
    Sources with embedded framing syntax disable reconstruction conservatively.
    The caller must supply its complete original scoped snapshot.
    """
    turns = eligible_sources(snapshot, scope=scope, as_of=as_of)
    if type(baseline) is not str:
        raise ValueError("Explicit baseline required")
    by_text, by_role_text = defaultdict(list), defaultdict(list)
    unsafe = False
    for t in turns:
        by_text[t.text].append(t)
        by_role_text[t.role, t.text].append(t)
        # Dated text is renderer metadata only at the start of the original turn.
        body = t.text.split("] ", 1)[-1]
        if (any(marker in t.text for marker in _MARKERS)
                or _ROLE_LINE.search(t.text) or _DATE_LINE.search(body)):
            unsafe = True
    report = dict(
        schema="source-presence-v1", scope=scope, baseline_sha256=digest(baseline),
        receipts=[], ambiguous=[], unmatched_ids=[], hazards=[],
        inference="EXACT_RENDERER_MATCH_ONLY", semantic_completeness="NOT_CERTIFIED",
    )
    if unsafe:
        report["hazards"].append("source_contains_reserved_framing")
        report["unmatched_ids"] = [t.id for t in turns]
        return report

    regions = []
    for kind in ("SEMANTIC MEMORY", "RECENT TURNS"):
        opening, closing = f"<[{kind}]>\n", f"</[{kind}]>"
        if baseline.count(opening) == baseline.count(closing) == 1:
            start = baseline.index(opening) + len(opening)
            end = baseline.index(closing)
            if start <= end:
                regions.append((kind, start, end))
        elif opening in baseline or closing in baseline:
            report["hazards"].append(f"ambiguous_{kind}_framing")
    header = "[ADDITIONAL SOURCE EVIDENCE]\n"
    if baseline.count(header) == 1:
        start = baseline.index(header) + len(header)
        regions.append(("ADDITIONAL SOURCE EVIDENCE", start, len(baseline)))
    elif header in baseline:
        report["hazards"].append("ambiguous_additional_source_framing")
    # Nested or intersecting regions cannot certify top-level source frames.
    regions.sort(key=lambda r: r[1])
    if any(a[2] > b[1] for a, b in zip(regions, regions[1:])):
        report["hazards"].append("overlapping_renderer_regions")
        report["unmatched_ids"] = [t.id for t in turns]
        return report

    candidates = []
    for kind, lo, hi in regions:
        for t in turns:
            stamp = t.observed_at.strftime("[%Y/%m/%d (%a) %H:%M] ")
            if not t.text.startswith(stamp):
                continue
            if kind == "SEMANTIC MEMORY":
                prefix, identities = "", by_text[t.text]
            elif kind == "RECENT TURNS":
                prefix, identities = f"{t.role.upper()}: ", by_role_text[t.role, t.text]
            else:
                prefix = f"[source {digest(t.text)[:16]} | {t.role}]\n"
                identities = by_role_text[t.role, t.text]
            pattern = prefix + t.text
            start = baseline.find(pattern, lo, hi)
            while start != -1:
                end = start + len(pattern)
                left_ok = start == lo or baseline[start - 1] == "\n"
                right_ok = end == hi or baseline[end] == "\n"
                if left_ok and right_ok:
                    # A complete smaller source can also prefix a larger source.
                    # Neither a truncated larger turn nor a delimiter in it may
                    # silently certify the smaller event's identity.
                    prefix_collision = any(
                        other != t.text and other.startswith(t.text + "\n")
                        for other in by_text
                    )
                    if len(identities) != 1 or prefix_collision:
                        report["ambiguous"].append(dict(
                            id=t.id, start=start + len(prefix), end=end,
                            reason="nonunique_source" if len(identities) != 1 else "source_prefix",
                        ))
                    else:
                        candidates.append(dict(
                            id=t.id, session=t.session, position=t.position, role=t.role,
                            observed_at=t.observed_at.isoformat(), sha256=digest(t.text),
                            start=start + len(prefix), end=end, renderer=kind,
                            future=t.observed_at > as_of,
                        ))
                start = baseline.find(pattern, start + 1, hi)
    # Exact source copies nested inside another reconstructed source are unsafe.
    good = []
    for row in candidates:
        overlap = any(
            other["id"] != row["id"]
            and max(row["start"], other["start"]) < min(row["end"], other["end"])
            for other in candidates
        )
        if overlap:
            report["ambiguous"].append(dict(
                id=row["id"], start=row["start"], end=row["end"], reason="source_overlap",
            ))
        else:
            good.append(row)
    # One certificate per source; a source may be represented in several tiers.
    seen = set()
    for row in sorted(good, key=lambda r: (r["start"], r["id"])):
        if row["id"] not in seen:
            seen.add(row["id"])
            report["receipts"].append(row)
    report["unmatched_ids"] = [t.id for t in turns if t.id not in seen]
    return report
