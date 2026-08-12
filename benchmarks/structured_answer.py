"""
STRUCTURED ANSWER STAGE — "LLM enumerates, code computes" (TReMu/PAL-style).

WHY (DECISION_AND_FAILURE_LOG §3.1ae + the 29-question autopsy)
The systematic failures are set-construction and date arithmetic done in
the model's head: off-by-one counts (rollercoasters 9 vs 10 with all
instances IN CONTEXT), wrong-instance-in-window ("sports event two weeks
ago" answered with a volleyball game from the wrong week), botched date
math ("0 days" for a 24-day gap), and failure to abstain when evidence is
genuinely insufficient. Three retrieval-side probes proved no packing or
budget change fixes these.

EVIDENCE BASE (researched, sourced in the log):
  TReMu (arXiv 2502.01630): LLM+code over timeline, +16pp over CoT on
    LoCoMo temporal with GPT-4o, inference-only.
  PAL (arXiv 2211.10435): code-computed counting 96.7% vs CoT 73.0%.
  Test of Time (arXiv 2406.09170): frontier models are 13-16% on duration
    arithmetic regardless of prompting — no prompt fixes date math.
  Fidelity Before Structure (arXiv 2601.00821): enumerate over VERBATIM
    evidence, never over lossy extracted artifacts — which is what our
    episodes-verbatim context already provides.

DIVISION OF LABOR (the supersession DNA, applied to answering):
  the LLM does the one thing only it can do — read prose and emit
  candidate instances as structured lines; THIS module does everything
  code does better — resolve relative windows against the question date,
  dedup near-identical instances, filter, sort, count, subtract dates —
  and refuses honestly when the computed evidence is insufficient.

Every deterministic function here is unit-tested at $0. The LLM stage
lives in qa_accuracy_eval.py (answerer="structured") and falls back to
the plain reasoning path on any parse failure.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

_WORDNUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8}

_DATE_RE = re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})")


def parse_date(s):
    """YYYY/MM/DD or YYYY-MM-DD anywhere in the string, else None."""
    if not s:
        return None
    m = _DATE_RE.search(str(s))
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def resolve_window(question: str, question_date: str):
    """The date window a relative phrase names, computed DETERMINISTICALLY
    from the question's own date — never guessed by the model.
    Deliberately WIDE (±6 days on point references): users are imprecise,
    and the window's job is to exclude the WRONG-WEEK instance, not to
    split hairs. Returns (lo, hi) or None."""
    now = parse_date(question_date)
    if now is None:
        return None
    q = question.lower()
    m = re.search(r"(\d+|a|an|one|two|three|four|five|six|seven|eight)\s+"
                  r"(day|week|month)s?\s+ago", q)
    if m:
        n = _WORDNUM.get(m.group(1))
        if n is None:
            try:
                n = int(m.group(1))
            except ValueError:
                return None
        days = n * {"day": 1, "week": 7, "month": 30}[m.group(2)]
        c = now - timedelta(days=days)
        return c - timedelta(days=6), c + timedelta(days=6)
    if re.search(r"\ba couple of days ago\b", q):
        return now - timedelta(days=7), now
    if re.search(r"\blast\s+(saturday|sunday|monday|tuesday|wednesday"
                 r"|thursday|friday)\b", q):
        return now - timedelta(days=10), now
    if re.search(r"\blast week\b", q):
        return now - timedelta(days=14), now - timedelta(days=2)
    if re.search(r"\b(past|last) month\b", q):
        return now - timedelta(days=35), now
    if re.search(r"\byesterday\b", q):
        return now - timedelta(days=2), now
    return None


def _unit_for(question: str) -> str:
    """The unit the question itself names — '5' judged wrong where
    '5 hours' is gold (v1's measured verbalization failure)."""
    q = question.lower()
    for u in ("times", "days", "hours", "weeks", "months", "years"):
        if re.search(rf"\b{u}\b", q):
            return u
    return ""


_STOP = frozenset("the a an of in on at to for with and or my i we".split())


def _key(desc: str) -> frozenset:
    return frozenset(w for w in re.findall(r"[a-z0-9]+", desc.lower())
                     if w not in _STOP)


def dedup(items: list) -> list:
    """Two candidate instances are THE SAME EVENT when their content
    words overlap >=60% and their dates (when both present) are within 2
    days. Conservative on purpose: with distinct dates they stay
    distinct even if worded identically ('rode rollercoasters' on two
    different days is two instances — collapsing them is exactly the
    undercount we are fixing)."""
    out = []
    for it in items:
        k, d = _key(it.get("desc", "")), parse_date(it.get("date"))
        dup = False
        for o in out:
            ok, od = _key(o.get("desc", "")), parse_date(o.get("date"))
            if d and od and abs((d - od).days) > 2:
                continue
            if not k or not ok:
                continue
            inter = len(k & ok) / min(len(k), len(ok))
            if inter >= 0.6 and (d is None or od is None
                                 or abs((d - od).days) <= 2):
                dup = True
                break
        if not dup:
            out.append(it)
    return out


def compute(payload: dict, question: str, question_date: str):
    """The deterministic stage. payload is the LLM's enumeration:
      {"operation": "count"|"date_diff"|"order"|"window_recall"|"direct",
       "items": [{"desc": str, "date": "YYYY/MM/DD"|null,
                  "count": int|null}, ...],
       "start": str|null, "end": str|null}   (endpoints for date_diff)
    Returns (answer_string | None, note). None => caller falls back to
    the plain reasoning answer — this stage REFUSES to guess."""
    op = (payload.get("operation") or "").strip()
    # v2 (2026-08-12): the v1 probe FAILED (72 computed answers, 47%
    # correct, 20 stable answers broken). Root cause: v1 gave CODE the
    # semantic work — set membership via dedup + window filtering — which
    # this project had ALREADY proven code cannot do (the relevance-
    # threshold refutation). PAL/TReMu assign code the ARITHMETIC only;
    # the model keeps selection. v2 restores that: the model submits its
    # FINAL, self-filtered set; code trusts membership and only computes.
    items = [i for i in (payload.get("items") or [])
             if isinstance(i, dict) and i.get("desc")]
    win = resolve_window(question, question_date)
    if win and op == "window_recall":
        dated = [i for i in items if parse_date(i.get("date"))]
        inside = [i for i in dated
                  if win[0] <= parse_date(i["date"]) <= win[1]]
        # safety only: if the model's set contains dated items and NONE
        # is inside the computable window, the selection contradicts the
        # deterministic anchor — refuse to the fallback rather than
        # answer from the wrong week (the volleyball failure).
        if dated and not inside:
            return None, "selection outside computed window"
        if inside:
            items = inside

    if op == "count":
        if not items:
            return None, "no items enumerated"
        total = sum(max(1, int(i.get("count") or 1)) for i in items)
        unit = _unit_for(question)
        ans = f"{total} {unit}".strip()
        return ans, f"{len(items)} items, total {total}"

    if op == "date_diff":
        a, b = parse_date(payload.get("start")), parse_date(payload.get("end"))
        if not a or not b:
            return None, "missing endpoint"
        days = abs((b - a).days)
        if re.search(r"\bmonths?\b", question.lower()):
            return f"{round(days / 30.44)} months", f"{days} days"
        if re.search(r"\bweeks?\b", question.lower()):
            return f"{round(days / 7)} weeks", f"{days} days"
        if re.search(r"\byears?\b|\bhow old\b", question.lower()):
            return f"{days // 365} years old", f"{days} days"
        return f"{days} days", f"{days} days"

    if op == "order":
        dated = [i for i in items if parse_date(i.get("date"))]
        if len(dated) < 2:
            return None, "fewer than 2 dated items"
        dated.sort(key=lambda i: parse_date(i["date"]))
        return ", ".join(i["desc"] for i in dated), f"{len(dated)} ordered"

    if op == "window_recall":
        if not items:
            return ("The available memories do not contain an event in "
                    "that time window."), "empty window"
        best = max(items, key=lambda i: len(i.get("desc", "")))
        return best["desc"], f"{len(items)} in window"

    return None, f"unhandled operation {op!r}"


def route(question: str) -> bool:
    """Does this question get the structured stage? Count, date-diff,
    ordering, and relative-window recall shapes only — everything else
    keeps the plain reasoning path untouched."""
    q = question.lower()
    return bool(
        re.search(r"\bhow (many|much|often|old|long)\b", q)
        or re.search(r"\b(days|weeks|months|years)\s+(had\s+)?passed\b", q)
        or re.search(r"\bwhich .{0,40}\b(first|last|most recently)\b", q)
        or re.search(r"\border of\b", q)
        or re.search(r"\b(ago|last (saturday|sunday|monday|tuesday"
                     r"|wednesday|thursday|friday|week|month))\b", q))
