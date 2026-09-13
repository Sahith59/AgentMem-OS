"""Opt-in source supplement for a frozen benchmark packet.

Retains the entire original packet and appends complete, attributed turns using
only unused character capacity. This addresses vocabulary-pruned evidence
without replacing existing facts or raw context. It remains experimental:
lossless text retention does not guarantee that the answerer will not regress.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from agentmem_os.benchmarks.uncapped_lexical_retrieval import rank_turns

_HEADER = "\n\n[ADDITIONAL SOURCE EVIDENCE]\n"


def supplement_packet(
    packet: str,
    turns: Sequence[Mapping[str, str]],
    query: str,
    *,
    char_cap: int = 40_000,
    max_extra_chars: int = 4_000,
) -> tuple[str, list[dict]]:
    """Return a packet plus receipts, without gold answers or annotations.

    Budget includes headings and attribution. Oversized turns are skipped,
    never clipped. Duplicate bodies with conflicting roles are omitted rather
    than assigning an arbitrary speaker. The input packet must already fit.
    """
    if char_cap < 0 or max_extra_chars < 0:
        raise ValueError("Character budgets must be non-negative")
    if len(packet) > char_cap:
        raise ValueError("Existing packet exceeds the character cap")
    available = min(char_cap - len(packet), max_extra_chars)
    if available <= len(_HEADER):
        return packet, []

    by_text: dict[str, set[str]] = {}
    for turn in turns:
        text = turn.get("content", "")
        role = turn.get("role", "")
        if text and role in {"user", "assistant", "system", "tool"}:
            by_text.setdefault(text, set()).add(role)
    contents = list(by_text)
    # Rank the complete scoped source population, then omit already-delivered
    # turns. This retains their contribution to IDF instead of changing scores
    # according to what the baseline happened to retrieve.
    ranked = rank_turns(contents, query, top_k=len(contents))
    extras = ""
    receipts = []
    for text in ranked:
        if text in packet or len(by_text[text]) != 1:
            continue
        role = next(iter(by_text[text]))
        digest = hashlib.sha256(text.encode()).hexdigest()
        block = f"[source {digest[:16]} | {role}]\n{text}\n"
        if len(_HEADER) + len(extras) + len(block) > available:
            continue
        start = len(packet) + len(_HEADER) + len(extras) + len(block) - len(text) - 1
        extras += block
        receipts.append({"role": role, "source_sha256": digest,
                         "packet_start": start, "packet_end": start + len(text)})
    if not receipts:
        return packet, []
    return packet + _HEADER + extras, receipts
