"""Opt-in rank admission before presentation, preserving complete input chunks.

This is not a semantic selector or an original-source certificate. A retriever
may already have shortened its input chunks. No database or model calls here.
"""

import re

from .evidence_packet import digest

_DATE = re.compile(r"^\[([^\]]+)\]")
_OPEN = "<[SEMANTIC MEMORY]>\n"
_CLOSE = "\n</[SEMANTIC MEMORY]>"
_SEPARATOR = "\n---\n"


def pack_ranked_chunks(chunks, *, token_budget, counter, chronological=True):
    """Greedily admit whole chunks by rank within both final-section limits.

    Every trial uses its FINAL presentation and includes all framing. Once a
    chunk is admitted, a later chunk cannot evict it. Oversize chunks are skipped
    with a receipt. Duplicate text remains distinct input occurrences.
    """
    if (
        type(chunks) not in (list, tuple)
        or any(type(c) is not str for c in chunks)
        or type(token_budget) is not int
        or token_budget < 0
        or type(chronological) is not bool
    ):
        raise ValueError("Invalid ranked chunks or budget")
    chunks = tuple(chunks)
    char_budget = token_budget * 4

    def render(indices):
        order = list(indices)
        if chronological:
            keys = [(_DATE.match(chunks[i]), i) for i in order]
            # Preserve the existing assembler's date-detection convention.
            if sum(bool(m) for m, _ in keys) >= max(2, len(keys) // 2):
                order = [i for _, i in sorted((m[1] if m else "~undated", i) for m, i in keys)]
        if not order:
            return "", order
        return _OPEN + _SEPARATOR.join(chunks[i] for i in order) + _CLOSE, order

    def count(text):
        value = counter.count(text)
        if type(value) is not int or value < 0:
            raise ValueError("Invalid token count")
        return value

    admitted, decisions = [], []
    section, presentation = "", []
    used_tokens = 0
    for index, chunk in enumerate(chunks):
        if not chunk.strip():
            decisions.append(dict(index=index, status="blank_input"))
            continue
        trial, order = render([*admitted, index])
        if len(trial) > char_budget:
            decisions.append(dict(index=index, status="character_nonfit", trial_chars=len(trial)))
            continue
        tokens = count(trial)
        if tokens > token_budget:
            decisions.append(
                dict(
                    index=index, status="token_nonfit", trial_chars=len(trial), trial_tokens=tokens
                )
            )
            continue
        admitted.append(index)
        section, presentation, used_tokens = trial, order, tokens
        decisions.append(dict(index=index, status="admitted"))

    receipts, cursor = [], len(_OPEN)
    for index in presentation:
        chunk = chunks[index]
        receipts.append(
            dict(index=index, start=cursor, end=cursor + len(chunk), sha256=digest(chunk))
        )
        cursor += len(chunk) + len(_SEPARATOR)
    return section, dict(
        schema="whole-ranked-chunk-packing-v1",
        policy="whole_rank_v1",
        input_sha256=[digest(c) for c in chunks],
        section_sha256=digest(section),
        token_budget=token_budget,
        char_budget=char_budget,
        used_chars=len(section),
        used_tokens=used_tokens,
        chronological=chronological,
        admitted_indices=admitted,
        presentation_indices=presentation,
        decisions=decisions,
        receipts=receipts,
        preservation="COMPLETE_INPUT_CHUNKS_ONLY",
        original_source_identity="NOT_CERTIFIED",
        semantic_completeness="NOT_CERTIFIED",
        answer_accuracy="NOT_MEASURED",
    )
