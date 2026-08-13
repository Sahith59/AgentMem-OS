"""Pins for read-time update resolution + advice-intent routing.

Every positive pin is a REAL failure shape from the n=500
knowledge-update/preference autopsy (2026-08-13); every negative pin is
a REAL adversarial case found in the same corpus during design. A
mutation that loosens a gate must turn one of these red.
"""
from types import SimpleNamespace

from agentmem_os.llm.fact_update_resolver import (
    annotation_for, find_update_pairs, is_update_pair, qualifier_classes,
    value_tokens)


def _f(fid, text, date, session, ftype="state"):
    return SimpleNamespace(id=fid, fact_text=text, t_mentioned=date,
                           source_session_id=session, fact_type=ftype)


# ── the wake-time case (autopsy KU-1): cross-session, cross-TYPE ──────
def test_wake_time_update_pairs_across_sessions_and_types():
    old = _f(1, "The user wakes up around 8:30 am on Saturdays.",
             "2023/05/23", "s1", "event")
    new = _f(2, "The user likes to wake up at 7:30 am on Saturdays.",
             "2023/05/27", "s2", "state")
    assert is_update_pair(old, new)
    pairs = find_update_pairs([old, new])
    assert pairs == {1: new}
    assert "7:30" in annotation_for(new)
    assert "2023/05/27" in annotation_for(new)


# ── the gym-time case (autopsy KU-5): months apart, same type ─────────
def test_gym_time_update_pairs():
    old = _f(1, "The user has gym sessions on Mondays, Wednesdays, and "
                "Fridays at 7:00 pm.", "2023/02/11", "s1")
    new = _f(2, "The user's gym sessions on Mondays, Wednesdays, and "
                "Fridays are at 6:00 pm.", "2023/05/30", "s2")
    assert is_update_pair(old, new)


# ── ADVERSARIAL (real corpus): weekday fact must NOT annotate a
#    Saturday fact — different attribute, high word overlap ───────────
def test_weekday_and_saturday_wake_times_do_not_pair():
    weekday = _f(1, "The user wants to wake up at 7:15 am on weekdays.",
                 "2023/05/23", "s1")
    saturday = _f(2, "The user likes to wake up at 7:30 am on Saturdays.",
                  "2023/05/27", "s2")
    assert qualifier_classes(weekday.fact_text) != qualifier_classes(
        saturday.fact_text)
    assert not is_update_pair(weekday, saturday)


# ── ADVERSARIAL (the followers case): same-day facts are ambiguity,
#    not an update ────────────────────────────────────────────────────
def test_same_day_facts_do_not_pair():
    a = _f(1, "The user has 1250 followers on Instagram.",
           "2023/05/25", "s1")
    b = _f(2, "The user is close to having 1300 followers on Instagram.",
           "2023/05/25", "s2")
    assert not is_update_pair(a, b)


# ── same-session pairs belong to the write-time judge ─────────────────
def test_same_session_facts_do_not_pair():
    a = _f(1, "The user runs 5 km every morning.", "2023/03/01", "s1")
    b = _f(2, "The user runs 8 km every morning.", "2023/03/05", "s1")
    assert not is_update_pair(a, b)


# ── value-less or unrelated facts never pair ─────────────────────────
def test_no_values_or_unrelated_facts_do_not_pair():
    a = _f(1, "The user enjoys hiking in the mountains.",
           "2023/03/01", "s1")
    b = _f(2, "The user enjoys hiking with friends.", "2023/04/01", "s2")
    assert not is_update_pair(a, b)          # no value tokens
    c = _f(3, "The user has 3 dogs.", "2023/03/01", "s3")
    d = _f(4, "The user visited 3 countries last year.",
           "2023/04/02", "s4")
    assert not is_update_pair(c, d)          # values equal + unrelated


def test_value_tokens_extraction():
    assert "8:30 am" in value_tokens("wakes at 8:30 am on Saturdays")
    assert value_tokens("no numbers here") == frozenset()


# ── newest update wins when several exist ─────────────────────────────
def test_multiple_updates_annotate_with_newest():
    v1 = _f(1, "The user's rent is 1500 dollars a month.",
            "2023/01/01", "s1")
    v2 = _f(2, "The user's rent is 1600 dollars a month.",
            "2023/03/01", "s2")
    v3 = _f(3, "The user's rent is 1700 dollars a month.",
            "2023/06/01", "s3")
    pairs = find_update_pairs([v1, v2, v3])
    assert pairs[1].id == 3
    assert pairs[2].id == 3


# ── advice-intent routing pins (context_assembler) ────────────────────
def test_advice_intent_fires_on_the_autopsy_preference_questions():
    from agentmem_os.llm.context_assembler import _ADVICE_INTENT_RE as R
    for q in [
        "Any advice on getting better results?",
        "Can you recommend a show or movie for me to watch tonight?",
        "Do you have any helpful tips?",
        "I'm not sure which one to choose. Any suggestions?",
        "Do you think it would be a good idea to attend my reunion?",
    ]:
        assert R.search(q), q


def test_advice_intent_loses_to_recall_on_assistant_questions():
    # "You recommended five bottles..." is a RECALL of assistant
    # content; suppression must beat boosting (same precedence rule as
    # aggregation routing).
    from agentmem_os.llm.context_assembler import (
        _ADVICE_INTENT_RE, _CONVERSATION_RECALL_RE)
    q = ("I'm looking back at our previous conversation about building "
         "a cocktail bar. You recommended five bottles to make the "
         "widest variety of gin-based cocktails. Can you remind me?")
    assert _ADVICE_INTENT_RE.search(q)
    assert _CONVERSATION_RECALL_RE.search(q)  # recall wins in assemble()


def test_advice_intent_does_not_fire_on_plain_fact_questions():
    from agentmem_os.llm.context_assembler import _ADVICE_INTENT_RE as R
    for q in [
        "What breed is my dog?",
        "Where did I go on my most recent family trip?",
        "How many followers do I have on Instagram now?",
    ]:
        assert not R.search(q), q
