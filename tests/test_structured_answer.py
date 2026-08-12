"""G1 for the structured answer stage — every deterministic function
pinned at $0. The LLM half is NOT tested here (it costs money); these
pins guarantee that when the LLM enumerates correctly, code computes
correctly — the entire point of the division of labor."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from structured_answer import (compute, dedup, parse_date, resolve_window,
                               route)


def test_window_two_weeks_ago():
    lo, hi = resolve_window("I mentioned a sports event two weeks ago. "
                            "What was it?", "2023/07/01 (Sat) 13:38")
    assert lo <= parse_date("2023/06/17") <= hi
    assert not (lo <= parse_date("2023/06/29") <= hi), \
        "a 2-day-old event must NOT be 'two weeks ago' — that is the " \
        "volleyball-instead-of-soccer failure"


def test_window_recall_rejects_out_of_window_instance():
    ans, _ = compute({"operation": "window_recall", "items": [
        {"desc": "volleyball league game", "date": "2023/06/29"},
        {"desc": "company charity soccer tournament", "date": "2023/06/17"},
    ]}, "I mentioned participating in a sports event two weeks ago. "
       "What was the event?", "2023/07/01")
    assert "soccer" in ans and "volleyball" not in ans


def test_count_keeps_distinct_dates_and_multiplicity():
    # The rollercoaster shape: same wording, different days, stated counts.
    ans, _ = compute({"operation": "count", "items": [
        {"desc": "rode rollercoasters at the theme park",
         "date": "2023/07/04", "count": 3},
        {"desc": "rode rollercoasters at the theme park",
         "date": "2023/08/12", "count": 3},
        {"desc": "rode a rollercoaster at the state fair",
         "date": "2023/09/01", "count": 1},
        {"desc": "rollercoaster rides at the amusement park",
         "date": "2023/10/20", "count": 3},
    ]}, "How many times did I ride rollercoasters across all the events?",
       "2023/11/01")
    assert ans.startswith("10"), "3+3+1+3 must equal 10"


def test_dedup_collapses_same_event_same_day_only():
    items = dedup([
        {"desc": "attended the jazz festival downtown", "date": "2023/05/01"},
        {"desc": "the jazz festival downtown I attended", "date": "2023/05/01"},
        {"desc": "attended the jazz festival downtown", "date": "2023/06/10"},
    ])
    assert len(items) == 2, "same wording on a DIFFERENT day is a " \
        "distinct instance — collapsing it is the undercount bug"


def test_date_diff_days_and_months():
    ans, _ = compute({"operation": "date_diff", "start": "2023/04/26",
                      "end": "2023/05/20", "items": []},
                     "How many days had passed since I started lessons?",
                     "2023/05/20")
    assert ans == "24 days", "unit must be included — v1 lost '5 hours' to a bare '5'"
    ans, _ = compute({"operation": "date_diff", "start": "2022/11/15",
                      "end": "2023/05/15", "items": []},
                     "How many months passed between the degrees?",
                     "2023/07/01")
    assert ans == "6 months"


def test_order_sorts_by_date():
    ans, _ = compute({"operation": "order", "items": [
        {"desc": "Game of Thrones", "date": "2023/04/01"},
        {"desc": "The Crown", "date": "2023/05/01"},
    ]}, "Which show did I start watching first?", "2023/06/01")
    assert ans.startswith("Game of Thrones")


def test_refuses_instead_of_guessing():
    ans, note = compute({"operation": "count", "items": []},
                        "How many tanks do I have?", "2023/06/01")
    assert ans is None
    ans, _ = compute({"operation": "date_diff", "start": None,
                      "end": "2023/05/15", "items": []},
                     "How many months passed?", "2023/07/01")
    assert ans is None, "a missing endpoint must refuse, never guess — " \
        "the bus-vs-taxi lesson: gold is 'not enough information'"


def test_route_targets_only_the_failing_shapes():
    assert route("How many movie festivals did I attend?")
    assert route("Which show did I start watching first?")
    assert route("Who did I go with to the music event last Saturday?")
    assert not route("What company does Rachel work at?")
    assert not route("Can you suggest a hotel for my trip to Miami?")
