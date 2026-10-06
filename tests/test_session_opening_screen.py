import copy
import importlib.metadata
import json

import pytest

from benchmarks import session_opening_screen as screen
from benchmarks.session_opening_retrieval import Session, Turn, expand


def package(tmp_path):
    date = "2026/01/02 (Fri) 12:00"
    sessions = (
        Session(
            "2026/01/01 (Thu) 12:00",
            (Turn("user", "I changed an appliance."), Turn("user", "My kitchen is clean.")),
        ),
    )
    baseline = "My kitchen is clean."
    candidate, report = expand("Which kitchen items changed?", baseline, sessions)
    c = dict(
        id="test",
        question="Which kitchen items changed?",
        date=date,
        baseline=baseline,
        candidate=candidate,
        retrieval=report,
        type="multi-session",
        gold="appliance",
        abstention=False,
        sessions=[
            dict(observed_at=s.observed_at, turns=[dict(role=t.role, text=t.text) for t in s.turns])
            for s in sessions
        ],
    )
    prompt = "{context}\n{question}{today_line}"
    old = dict(
        answer_prompt=prompt,
        cases=[
            dict(
                id="test",
                runtime=dict(question=c["question"], question_date=date, packet=baseline),
                type=c["type"],
                gold=c["gold"],
                abstention=False,
            )
        ],
    )
    source = [
        dict(
            question_id="test",
            question=c["question"],
            question_date=date,
            haystack_dates=[sessions[0].observed_at, "2026/01/03 (Sat) 12:00"],
            haystack_sessions=[
                [dict(role=t.role, content=t.text, has_answer=True) for t in sessions[0].turns],
                [dict(role="user", content="Future kitchen appliance.", has_answer=False)],
            ],
        )
    ]
    records = {}
    for name, data in (("source_package", old), ("source_dataset", source)):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(data))
        records[name] = dict(path=str(path), sha256=screen.file_hash(path))
    p = dict(
        cases=[c],
        answer_prompt=prompt,
        answer_settings=screen.ANSWER_SETTINGS,
        judge_settings=screen.judge.SETTINGS,
        judge_rates=screen.judge.RATES,
        sklearn_version=importlib.metadata.version("scikit-learn"),
        code_sha256=screen.code_hashes(),
        maximum_attempts=4,
        **records,
    )
    p["request_bounds_nusd"] = {
        "test": {
            st: screen.reservation(screen.request(p, c, st, upper_bound=True))
            for st in screen.stages(c)
        }
    }
    p["maximum_reservation_nusd"] = sum(p["request_bounds_nusd"]["test"].values())
    return p


def approval(p, dest):
    return dict(
        approved=True,
        mode="offline-test",
        authorization_text="fake test",
        package_sha256=screen.validate(p),
        maximum_attempts=4,
        budget_nusd=p["maximum_reservation_nusd"],
        output_directory=str(dest.resolve()),
    )


def fake_provider():
    calls = []

    def fake(req):
        calls.append(req)
        return dict(
            id=str(len(calls)),
            request_id="req-" + str(len(calls)),
            model=req["model"],
            text="yes" if req["model"] == screen.judge.SETTINGS["model"] else "appliance",
            finish_reason="stop",
            usage=dict(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )

    return fake, calls


def test_four_calls_complete_and_rerun_makes_no_new_calls(tmp_path):
    p = package(tmp_path)
    dest = tmp_path / "run"
    a = approval(p, dest)
    fake, calls = fake_provider()
    result = screen.run(p, dest, a, mode="offline-test", provider=fake)
    assert result["correct"] == dict(baseline=1, focus=1) and len(calls) == 4
    assert result["gains"] == result["losses"] == 0
    screen.run(p, dest, a, mode="offline-test", provider=fake)
    assert len(calls) == 4


def test_error_is_preserved_and_never_retried(tmp_path):
    p = package(tmp_path)
    dest = tmp_path / "run"
    a = approval(p, dest)
    calls = []

    def fail(req):
        calls.append(req)
        raise RuntimeError("test failure")

    with pytest.raises(RuntimeError):
        screen.run(p, dest, a, mode="offline-test", provider=fail)
    with pytest.raises(ValueError, match="Unresolved"):
        screen.run(p, dest, a, mode="offline-test", provider=fail)
    state = json.loads((dest / "checkpoint.json").read_text())
    assert len(calls) == 1 and state["reserved_nusd"] > 0
    assert len(screen.verify(p, state)["unreconciled_usage_attempts"]) == 1


def test_future_source_tampering_and_changed_population_rejected(tmp_path):
    p = package(tmp_path)
    screen.validate(p)
    p["cases"][0]["sessions"].append(
        dict(
            observed_at="2026/01/03 (Sat) 12:00",
            turns=[dict(role="user", text="Future kitchen appliance.")],
        )
    )
    with pytest.raises(ValueError, match="date cutoff"):
        screen.validate(p)


def test_reference_never_changes_answer_requests(tmp_path):
    p = package(tmp_path)
    c = p["cases"][0]
    before = screen.request(p, c, "focus")
    c = copy.deepcopy(c)
    c["gold"] = "INJECTED GOLD"
    assert screen.request(p, c, "focus") == before
    assert "INJECTED GOLD" not in json.dumps(before)


def test_candidate_and_approval_tampering_rejected_before_dispatch(tmp_path):
    p = package(tmp_path)
    dest = tmp_path / "run"
    a = approval(p, dest)
    fake, calls = fake_provider()
    p["cases"][0]["candidate"] += " fabricated"
    with pytest.raises(ValueError, match="expansion"):
        screen.run(p, dest, a, mode="offline-test", provider=fake)
    assert calls == []
