"""Append a small number of high-confidence complete source turns.

The original packet remains byte-for-byte as a prefix. This candidate lowers
attention dilution; it does not solve reasoning, conflicting operands,
abstention policy, extraction lineage, or disputed benchmark references.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from agentmem_os.benchmarks.precise_lexical_retrieval import rank_turns

_HEADER = "\n\n[ADDITIONAL SOURCE EVIDENCE]\n"


def supplement_packet(
    packet: str,
    turns: Sequence[Mapping[str, str]],
    query: str,
    *,
    char_cap: int = 40_000,
    max_extra_chars: int = 4_000,
) -> tuple[str, list[dict]]:
    """Return the preserved packet plus independently attributable turns."""
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
    ranked = rank_turns(list(by_text), query)
    extras = ""
    receipts = []
    for text in ranked:
        if text in packet or len(by_text[text]) != 1:
            continue
        role = next(iter(by_text[text]))
        source_hash = hashlib.sha256(text.encode()).hexdigest()
        block = f"[source {source_hash[:16]} | {role}]\n{text}\n"
        if len(_HEADER) + len(extras) + len(block) > available:
            continue
        start = len(packet) + len(_HEADER) + len(extras) + len(block) - len(text) - 1
        extras += block
        receipts.append(
            {
                "role": role,
                "source_sha256": source_hash,
                "packet_start": start,
                "packet_end": start + len(text),
            }
        )
    if not receipts:
        return packet, []
    return packet + _HEADER + extras, receipts
