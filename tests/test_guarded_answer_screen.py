import json
from dataclasses import asdict

import pytest

from benchmarks import guarded_answer_screen as screen
from benchmarks.evaluator_v1 import contract as judge
from benchmarks.evidence_focus import FocusInput, SourceTurn
from benchmarks.guarded_focus import apply_guarded
from benchmarks.luna_evidence_plan import SETTINGS


def runtime():
    turns = (
        SourceTurn("a", "assistant", "", "You could book a train."),
        SourceTurn("b", "user", "", "I have not booked a train."),
    )
    return FocusInput("Which train did I book?", "", "\n".join(t.text for t in turns), turns)


def plan():
    return dict(
        requirements=dict(
            operation="direct_recall", target="", time_window="", output_unit="", needed_facts=[]
        ),
        support_turn_ids=[],
        qualification_turn_ids=["a", "b"],
        rejected_turn_ids=[],
        sufficiency="uncertain",
    )


def package():
    case = dict(
        id="test",
        runtime=asdict(runtime()),
        gold="GOLD_ONLY_JUDGE",
        type="single-session-user",
        abstention=True,
    )
    p = dict(
        cases=[case],
        answer_prompt="{context}\n{question}{today_line}",
        answer_settings=screen.ANSWER_SETTINGS,
        planner_settings=SETTINGS,
        judge_settings=judge.SETTINGS,
        judge_rates=judge.RATES,
        code_sha256=screen.code_hashes(),
        gate=dict(
            minimum_applied_focus=1,
            minimum_net_gain_vs_each_control=1,
            maximum_losses_vs_each_control=0,
        ),
    )
    p["request_bounds_nusd"] = {
        "test": {
            s: screen.reservation(screen.request(p, case, s, upper_bound=True))
            for s in screen.STAGES
        }
    }
    p["maximum_reservation_nusd"] = sum(p["request_bounds_nusd"]["test"].values())
    p["maximum_attempts"] = 8
    return p


def auth(p, path):
    return dict(
        approved=True,
        mode="offline-test",
        authorization_text="fake test",
        package_sha256=screen.validate(p),
        maximum_attempts=8,
        budget_nusd=p["maximum_reservation_nusd"],
        output_directory=str(path),
    )


def provider():
    calls = []

    def invoke(req):
        calls.append(req)
        text = (
            json.dumps(plan())
            if req.get("response_format")
            else (
                "yes" if req["model"] == judge.SETTINGS["model"] else "No booking is established."
            )
        )
        return dict(
            text=text,
            model=req["model"],
            finish_reason="stop",
            id=f"fake{len(calls)}",
            request_id=f"req{len(calls)}",
            usage=dict(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )

    return invoke, calls


def test_guard_abstains_without_dropping_any_baseline_evidence():
    v = runtime()
    candidate, report = apply_guarded(v, json.dumps(plan()))
    assert candidate == v.packet and report["receipts"] == []
    assert report["status"] == "UNCHANGED_UNCERTAIN_ASSISTANT_QUALIFICATION"
    assert report["blocked_qualification_ids"] == ["a"]


@pytest.mark.parametrize("sufficiency", ["complete", "uncertain"])
def test_advice_as_direct_support_and_user_qualification_remain_possible(sufficiency):
    v = runtime()
    p = plan()
    p.update(support_turn_ids=["a"], qualification_turn_ids=["b"], sufficiency=sufficiency)
    candidate, report = apply_guarded(v, json.dumps(p))
    assert report["status"] == "APPLIED" and candidate.startswith(v.packet)
    assert len(report["receipts"]) == 2


def test_projection_and_compute_allowance():
    p = package()
    c = p["cases"][0]
    outputs = dict(plan=json.dumps(plan()), draft="A draft", baseline="b", revision="r", focus="f")
    for stage in screen.STAGES:
        req = screen.request(p, c, stage, outputs)
        if not stage.startswith("judge_"):
            assert "GOLD_ONLY_JUDGE" not in json.dumps(req)
        assert screen.reservation(req) <= p["request_bounds_nusd"]["test"][stage]
    assert (
        screen.request(p, c, "plan")["max_completion_tokens"]
        == screen.request(p, c, "draft")["max_completion_tokens"]
    )
    assert (
        screen.request(p, c, "revision", outputs)["max_completion_tokens"]
        == screen.request(p, c, "focus", outputs)["max_completion_tokens"]
    )
    assert screen.request(p, c, "focus", outputs) == screen.request(p, c, "baseline", outputs)


def test_fake_full_pipeline_and_resume_do_not_repeat_calls(tmp_path):
    p = package()
    fake, calls = provider()
    a = auth(p, tmp_path)
    result = screen.run(p, tmp_path, a, mode="offline-test", provider=fake)
    assert result["completed"] == 8 and result["complete_pairs"] == 1
    assert result["correct"] == dict(baseline=1, revision=1, focus=1)
    assert result["status"] == "FAIL_OR_INCOMPLETE"  # all-right mocks are not gains
    assert screen.run(p, tmp_path, a, mode="offline-test", provider=fake) == result
    assert len(calls) == 8


def test_failed_attempt_keeps_receipt_and_never_retries(tmp_path):
    p = package()
    good, calls = provider()
    a = auth(p, tmp_path)

    def bad(req):
        response = good(req)
        response.update(text="{", finish_reason="length")
        return response

    with pytest.raises(RuntimeError, match="no retry"):
        screen.run(p, tmp_path, a, mode="offline-test", provider=bad)
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    result = screen.verify(p, state)
    assert result["attempted"] == 1 and result["completed"] == 0 and result["usage_upper_nusd"] > 0
    with pytest.raises(ValueError, match="Unresolved"):
        screen.run(p, tmp_path, a, mode="offline-test", provider=good)
    assert len(calls) == 1


def test_bad_budget_and_paid_injection_fail_before_dispatch(tmp_path):
    p = package()
    fake, calls = provider()
    a = auth(p, tmp_path)
    a["budget_nusd"] -= 1
    with pytest.raises(ValueError):
        screen.run(p, tmp_path, a, mode="offline-test", provider=fake)
    a = auth(p, tmp_path)
    a["mode"] = "paid"
    with pytest.raises(ValueError, match="Real provider"):
        screen.run(p, tmp_path, a, mode="paid", provider=fake)
    assert not calls


def test_unknown_provider_usage_preserved_without_retry(tmp_path):
    p = package()
    good, calls = provider()
    a = auth(p, tmp_path)

    def unknown(req):
        response = good(req)
        response.pop("usage")
        return response

    with pytest.raises(RuntimeError, match="no retry"):
        screen.run(p, tmp_path, a, mode="offline-test", provider=unknown)
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    result = screen.verify(p, state)
    assert result["unreconciled_usage_attempts"] == ["test:plan"]
    assert result["usage_upper_nusd"] == 0  # known subtotal, not zero billing
    assert result["reserved_nusd"] > 0
    with pytest.raises(ValueError, match="Unresolved"):
        screen.run(p, tmp_path, a, mode="offline-test", provider=good)
    assert len(calls) == 1
