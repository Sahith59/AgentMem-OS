"""Bounded raw-turn reserve for explicit relative-time questions.

This module is benchmark-only.  It does not alter ContextAssembler or the
production retrieval path.  The caller supplies already-scoped turn rows and
an explicit question date.  At most ``limit`` completed first-person event
turns inside the requested time window are moved ahead of the ordinary raw
retrieval ranking.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Sequence


_STAMP_RE = re.compile(r"^\[(\d{4}/\d{2}/\d{2})(?:\s[^\]]*)?\]")
_RELATIVE_RE = re.compile(
    r"\b(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?P<unit>day|week|month|year)s?\s+ago\b",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    r"\b(?:past|last|previous)\s+"
    r"(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?P<unit>day|week|month|year)s?\b",
    re.IGNORECASE,
)
_COUNT = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# These cues require a first-person subject and completed-event wording.  They
# intentionally omit broad words such as "was" and "had", which admitted many
# descriptions and plans in the development audit.
_COMPLETED_EVENT_RE = re.compile(
    r"\b(?:i|we)\b[^.!?\n]{0,180}\b(?:"
    r"attended|participated|visited|went|returned|completed|finished|"
    r"hiked|joined|got\s+back|took\s+part"
    r")\b",
    re.IGNORECASE,
)
_PLAN_RE = re.compile(
    r"\b(?:plan(?:ning|ned)?|consider(?:ing|ed)?|might|may|want(?:ed)?|"
    r"thinking\s+of|going\s+to)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TemporalWindow:
    start: date
    end: date
    target: date | None
    kind: str


def parse_date(value: str | date | datetime) -> date:
    """Parse benchmark timestamps without consulting the wall clock."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    match = re.search(r"\d{4}/\d{2}/\d{2}", str(value))
    if not match:
        raise ValueError("reference date must contain YYYY/MM/DD")
    return datetime.strptime(match.group(0), "%Y/%m/%d").date()


def _count(value: str) -> int:
    return int(value) if value.isdigit() else _COUNT[value.lower()]


def _subtract_months(value: date, months: int) -> date:
    total = value.year * 12 + value.month - 1 - months
    year, month0 = divmod(total, 12)
    month = month0 + 1
    return date(year, month, min(value.day, calendar.monthrange(year, month)[1]))


def _subtract(value: date, count: int, unit: str) -> date:
    if unit == "day":
        return value - timedelta(days=count)
    if unit == "week":
        return value - timedelta(weeks=count)
    if unit == "month":
        return _subtract_months(value, count)
    if unit == "year":
        try:
            return value.replace(year=value.year - count)
        except ValueError:  # February 29
            return value.replace(year=value.year - count, day=28)
    raise ValueError(f"unsupported temporal unit: {unit}")


def temporal_window(query: str, reference_date: str | date | datetime,
                    point_tolerance_days: int = 7) -> TemporalWindow | None:
    """Return an explicit relative-time window, or ``None`` when absent."""
    reference = parse_date(reference_date)
    match = _RELATIVE_RE.search(query)
    if match:
        target = _subtract(reference, _count(match["count"]),
                           match["unit"].lower())
        tolerance = timedelta(days=point_tolerance_days)
        return TemporalWindow(target - tolerance, target + tolerance,
                              target, "point")
    match = _RANGE_RE.search(query)
    if match:
        start = _subtract(reference, _count(match["count"]),
                          match["unit"].lower())
        return TemporalWindow(start, reference, None, "range")
    return None


def _field(turn, name: str):
    if isinstance(turn, dict):
        return turn.get(name)
    return getattr(turn, name, None)


def completed_user_event(turn) -> bool:
    """Conservative admission check for an explicitly completed user event."""
    if str(_field(turn, "role") or "").lower() != "user":
        return False
    content = str(_field(turn, "content") or "")
    match = _COMPLETED_EVENT_RE.search(content)
    if not match:
        return False
    prefix = content[max(0, match.start() - 80):match.end()]
    return not _PLAN_RE.search(prefix)


def select_dated_event_turns(query: str, reference_date, turns: Iterable,
                             limit: int = 3) -> list[str]:
    """Select a bounded ranked reserve without changing ordinary retrieval."""
    if limit <= 0:
        return []
    window = temporal_window(query, reference_date)
    if window is None:
        return []

    candidates: list[tuple[str, date, int]] = []
    for ordinal, turn in enumerate(turns):
        if not completed_user_event(turn):
            continue
        content = str(_field(turn, "content") or "")
        stamp = _STAMP_RE.match(content)
        if not stamp:
            continue
        occurred = datetime.strptime(stamp.group(1), "%Y/%m/%d").date()
        if window.start <= occurred <= window.end:
            candidates.append((content, occurred, ordinal))
    if not candidates:
        return []

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [item[0] for item in candidates]
    word_vectorizer = TfidfVectorizer(
        max_features=2048, sublinear_tf=True, min_df=1)
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=4096,
        sublinear_tf=True, min_df=1)
    try:
        word_matrix = word_vectorizer.fit_transform(texts)
        word_scores = cosine_similarity(
            word_vectorizer.transform([query]), word_matrix)[0]
        char_matrix = char_vectorizer.fit_transform(texts)
        char_scores = cosine_similarity(
            char_vectorizer.transform([query]), char_matrix)[0]
    except ValueError:
        return []

    # Word matching is the primary signal. Character-within-word matching is
    # a bounded fallback for morphology such as trip/trips and hike/hiked.
    similarities = [max(word, char * 0.75)
                    for word, char in zip(word_scores, char_scores)]

    def key(index: int):
        _, occurred, ordinal = candidates[index]
        distance = (abs((occurred - window.target).days)
                    if window.target is not None else 0)
        return (-similarities[index], distance, ordinal)

    ranked = sorted(range(len(candidates)), key=key)
    return [candidates[index][0] for index in ranked[:limit]
            if similarities[index] > 0.01]


def prepend_reserve(base_chunks: Sequence[str], reserve: Sequence[str]) -> list[str]:
    """Prepend reserve chunks once, preserving order in both inputs."""
    result = []
    seen = set()
    for chunk in [*reserve, *base_chunks]:
        if chunk not in seen:
            result.append(chunk)
            seen.add(chunk)
    return result
