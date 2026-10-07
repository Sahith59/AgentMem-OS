"""Canonical typed boundary for an audit-only original-source unit compiler.

No proposer, model calls, semantic approval or answer-path admission lives here.
Offsets are Python Unicode codepoint offsets into the exact original turn text.
An edge declares a dependency; it does not prove that all dependencies are known.
"""

from dataclasses import dataclass
from datetime import datetime

DEPENDENCY_KINDS = frozenset(
    {
        "entity",
        "event",
        "time",
        "negation",
        "modality",
        "role",
        "membership",
        "correction",
        "reference",
        "condition",
    }
)
MAX_SPANS = 128
MAX_DEPENDENCIES = 512


@dataclass(frozen=True)
class QuoteSpan:
    id: str
    source_id: str
    source_sha256: str
    start: int
    end: int
    quote_sha256: str


@dataclass(frozen=True)
class Dependency:
    source_span_id: str
    required_span_id: str
    kind: str


@dataclass(frozen=True)
class UnresolvedDependency:
    span_id: str
    kind: str


@dataclass(frozen=True)
class SourceUnit:
    """One question-bound rooted DAG, admitted as a whole or refused as a whole.

    Multiple roots and shared dependencies are allowed. All spans must be
    reachable. Overlapping ranges within an original source are rejected; share
    one span node instead. No generated claim text or completeness flag exists.
    """

    id: str
    snapshot_sha256: str
    question_sha256: str
    as_of: datetime
    root_span_ids: tuple[str, ...]
    spans: tuple[QuoteSpan, ...]
    dependencies: tuple[Dependency, ...] = ()
    unresolved: tuple[UnresolvedDependency, ...] = ()
