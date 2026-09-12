from datetime import date

from agentmem_os.benchmarks.dated_event_reserve import (
    completed_user_event,
    prepend_reserve,
    select_dated_event_turns,
    temporal_window,
)


def _turn(content, role="user"):
    return {"role": role, "content": content}


def test_relative_point_window_is_anchored_to_supplied_date():
    window = temporal_window("Where was the event two weeks ago?",
                             "2023/02/01 (Wed) 08:41")
    assert window.kind == "point"
    assert window.target == date(2023, 1, 18)
    assert window.start == date(2023, 1, 11)
    assert window.end == date(2023, 1, 25)


def test_calendar_month_range_does_not_use_wall_clock():
    window = temporal_window("Which trips were in the past three months?",
                             "2023/06/01")
    assert window.kind == "range"
    assert window.start == date(2023, 3, 1)
    assert window.end == date(2023, 6, 1)


def test_no_explicit_relative_window_means_no_reserve():
    assert temporal_window("Which museum did I visit?", "2023/02/01") is None


def test_admission_requires_user_role_and_completed_first_person_event():
    assert completed_user_event(_turn("[2023/01/15] I attended the exhibit."))
    assert not completed_user_event(_turn(
        "[2023/01/15] I attended the exhibit.", role="assistant"))
    assert not completed_user_event(_turn(
        "[2023/01/15] I am planning to attend the exhibit."))
    assert not completed_user_event(_turn(
        "[2023/01/15] Have you ever visited the exhibit?"))
    assert not completed_user_event(_turn(
        '[2023/01/15] "The last time I saw you, I was unfair." Why?'))


def test_point_query_selects_completed_relevant_event():
    turns = [
        _turn("[2023/01/14 (Sat)] I am planning to attend an art event."),
        _turn("[2023/01/15 (Sun)] I attended the Ancient Civilizations "
              "exhibit at the Metropolitan Museum of Art today."),
        _turn("[2023/01/15 (Sun)] I finished assembling a model car."),
        _turn("[2023/01/08 (Sun)] I attended a tour at MoMA."),
    ]
    selected = select_dated_event_turns(
        "I participated in an art-related event two weeks ago. Where was it?",
        "2023/02/01", turns, limit=1)
    assert selected == [turns[1]["content"]]


def test_point_query_prefers_closer_date_before_lexical_score():
    turns = [
        _turn("[2023/01/14] I attended a very detailed art event."),
        _turn("[2023/01/18] I attended the art exhibit."),
    ]
    assert select_dated_event_turns(
        "Where was the art event two weeks ago?", "2023/02/01",
        turns, limit=1) == [turns[1]["content"]]


def test_range_query_selects_completed_trip_without_admitting_plan():
    turns = [
        _turn("[2023/03/10 (Fri)] I just got back from a day hike to Muir "
              "Woods with my family and am preparing for future trips."),
        _turn("[2023/04/20 (Thu)] I returned from a road trip to Big Sur."),
        _turn("[2023/05/20 (Sat)] I am planning a trip to Tahoe."),
        _turn("[2023/02/20 (Mon)] I completed a winter trip to Tahoe."),
    ]
    selected = select_dated_event_turns(
        "What is the order of the three trips in the past three months?",
        "2023/06/01", turns, limit=3)
    assert turns[0]["content"] in selected
    assert turns[1]["content"] in selected
    assert turns[2]["content"] not in selected
    assert turns[3]["content"] not in selected


def test_ordered_range_reserves_earliest_eligible_event_first():
    turns = [
        _turn("[2023/05/15] I completed a detailed solo camping trip."),
        _turn("[2023/03/10] I got back from a day hike during my trips."),
    ]
    assert select_dated_event_turns(
        "What is the order of the trips in the past three months, from "
        "earliest to latest?", "2023/06/01", turns, limit=1) == [
            turns[1]["content"]]


def test_irrelevant_earlier_event_cannot_block_ordered_range_candidate():
    turns = [
        _turn("[2023/03/01] I attended a workshop about accounting."),
        _turn("[2023/03/10] I got back from a day hike during my trips."),
    ]
    assert select_dated_event_turns(
        "What is the order of the trips in the past three months, from "
        "earliest to latest?", "2023/06/01", turns, limit=1) == [
            turns[1]["content"]]


def test_prepend_reserve_deduplicates_without_reordering():
    assert prepend_reserve(["base-a", "shared", "base-b"],
                           ["reserved", "shared"]) == [
                               "reserved", "shared", "base-a", "base-b"]


def test_relative_time_words_alone_do_not_admit_unrelated_events():
    turns = [
        _turn("[2023/04/19] I returned from a trip to Bali."),
        _turn("[2023/04/17] I attended a filmmaking panel."),
    ]
    assert select_dated_event_turns(
        "What gardening-related activity did I do two weeks ago?",
        "2023/05/05", turns, limit=1) == []


def test_count_scaffold_time_does_not_match_unrelated_personal_best():
    turns = [
        _turn("[2023/05/20] I finished a marathon with a personal best time."),
    ]
    assert select_dated_event_turns(
        "How many times did I bake egg tarts in the past two weeks?",
        "2023/05/30", turns, limit=1) == []
