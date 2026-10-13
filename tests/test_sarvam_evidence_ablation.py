"""Offline tests for the bounded reporter-only evidence screen."""

import copy
import hashlib
import json
import shutil
import sqlite3

import pytest

from benchmarks import sarvam_evidence_ablation as ablation
from benchmarks.sarvam_native_controls.engine import FORMAT
from benchmarks.sarvam_native_study.transport import canonical, digest, save
from benchmarks.sarvam_ordinary_v2 import FINAL_INSTRUCTION
from benchmarks.sarvam_reporter_wording import CANDIDATE_FIRST


def sample(with_receipt=True):
    user = {
        "question": "Which requested date applies if the morning slot is unknown?",
        "intent": "read",
        "appointment_id": "BOOK-X",
        "requested_change": None,
        "source_messages": [
            {
                "id": "source-1",
                "role": "customer",
                "content": "If morning is free, choose 2028-04-01; otherwise 2028-05-01.",
            },
            {
                "id": "source-2",
                "role": "customer",
                "content": "Availability has not been checked. Do not change the booking.",
            },
        ],
        "application_evidence": {
            "current": {"date": "2028-03-01", "version": 3, "closed": False},
            "approvals": [],
            "trusted_issuers": ["service"],
            "prior_execution_receipts": [],
            "server_time": 123456,
        },
    }
    assistant = {
        "role": "assistant",
        "content": "I already confirmed April; morning is free.",
        "reasoning_content": "unsupported rationale",
        "refusal": None,
        "tool_calls": None,
    }
    history = [
        {"role": "system", "content": "Original actor instructions"},
        {"role": "user", "content": canonical(user)},
        assistant,
    ]
    tools, trace = [], []
    if with_receipt:
        call = {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "read_current",
                "arguments": canonical({"appointment_id": "BOOK-X"}),
            },
        }
        assistant["tool_calls"] = [call]
        reply = {"date": "2028-03-01", "version": 3}
        history.append({"role": "tool", "tool_call_id": "call-1", "content": canonical(reply)})
        tools = [{"type": "function", "function": {"name": "read_current"}}]
        trace = [
            {
                "call_id": "call-1",
                "name": "read_current",
                "arguments": {"appointment_id": "BOOK-X"},
                "response": reply,
            }
        ]
    evidence = {"original_messages": history, "original_tool_definitions": tools}
    request = {
        "model": "sarvam-105b",
        "temperature": 0,
        "reasoning_effort": "low",
        "max_tokens": 8192,
        "stream": False,
        "n": 1,
        "response_format": FORMAT,
        "messages": [
            {
                "role": "system",
                "content": CANDIDATE_FIRST + "\n" + FINAL_INSTRUCTION.split("\n", 1)[1],
            },
            {"role": "user", "content": canonical(evidence)},
        ],
    }
    return request, trace, evidence


def test_three_arms_fix_profile_and_change_only_evidence_layout():
    original, trace, evidence = sample()
    before = copy.deepcopy(original)
    rendered = [ablation.render(original, trace, arm) for arm in ablation.ARMS]
    assert original == before
    systems = {r[0]["messages"][0]["content"] for r in rendered}
    assert systems == {ablation.SYSTEM}
    assert all(
        {k: v for k, v in r[0].items() if k != "messages"}
        == {k: v for k, v in original.items() if k != "messages"}
        for r in rendered
    )
    assert len({digest(r[0]) for r in rendered}) == 3
    for arm, (request, sidecar, ledger) in zip(ablation.ARMS, rendered):
        ablation.verify_omissions(request, sidecar, ledger, arm)
        assert sidecar["original"] == evidence
        assert sidecar["actual_tool_trace"] == trace
        assert "audit" not in json.loads(request["messages"][1]["content"])


def test_withheld_actor_prose_preserves_exact_proposal_and_actual_receipt():
    original, trace, evidence = sample()
    request, sidecar, ledger = ablation.render(original, trace, "grouped_actor_prose_withheld")
    grouped = json.loads(request["messages"][1]["content"])
    claim = grouped["actor_claims"][0]
    assert claim["message"] == {
        "role": "assistant",
        "tool_calls": evidence["original_messages"][2]["tool_calls"],
        "refusal": None,
    }
    assert grouped["execution_view"]["actual_tool_receipts"] == [
        {"message_index": 3, "message": evidence["original_messages"][3]}
    ]
    assert (
        grouped["source_view"]["source_messages"]
        == json.loads(evidence["original_messages"][1]["content"])["source_messages"]
    )
    assert [(e["message_index"], e["field"]) for e in ledger] == [
        (2, "content"),
        (2, "reasoning_content"),
    ]
    assert all(
        e["value_sha256"] == digest(evidence["original_messages"][2][e["field"]]) for e in ledger
    )
    ablation.verify_omissions(request, sidecar, ledger, "grouped_actor_prose_withheld")


def test_unknown_actor_field_refused_before_any_request():
    original, trace, evidence = sample()
    evidence["original_messages"][2]["unknown_private_text"] = "would leak"
    original["messages"][1]["content"] = canonical(evidence)
    with pytest.raises(ValueError, match="Unknown actor field"):
        ablation.render(original, trace, "grouped_actor_prose_withheld")


def test_no_actor_prose_makes_two_grouped_arms_identical():
    original, trace, evidence = sample()
    message = evidence["original_messages"][2]
    message["content"] = ""
    message["reasoning_content"] = None
    message["refusal"] = None
    original["messages"][1]["content"] = canonical(evidence)
    lossless, _, _ = ablation.render(original, trace, "grouped_lossless")
    withheld, sidecar, ledger = ablation.render(original, trace, "grouped_actor_prose_withheld")
    assert lossless == withheld
    assert ledger == []
    ablation.verify_omissions(withheld, sidecar, ledger, "grouped_actor_prose_withheld")


def test_nontext_actor_prose_refused():
    original, trace, evidence = sample()
    evidence["original_messages"][2]["content"] = ["unexpected"]
    original["messages"][1]["content"] = canonical(evidence)
    with pytest.raises(ValueError, match="nontext"):
        ablation.render(original, trace, "grouped_actor_prose_withheld")


@pytest.mark.parametrize("tamper", ["receipt", "trace", "source", "ledger", "sidecar"])
def test_tampering_cannot_pass_reconstruction_or_ledger(tamper):
    original, trace, _ = sample()
    request, sidecar, ledger = ablation.render(original, trace, "grouped_actor_prose_withheld")
    request, sidecar, ledger = copy.deepcopy((request, sidecar, ledger))
    if tamper in ("receipt", "source"):
        evidence = json.loads(request["messages"][1]["content"])
        if tamper == "receipt":
            evidence["execution_view"]["actual_tool_receipts"][0]["message"]["content"] = canonical(
                {"date": "forged"}
            )
        else:
            evidence["source_view"]["source_messages"][0]["content"] = "fabricated"
        request["messages"][1]["content"] = canonical(evidence)
    elif tamper == "trace":
        sidecar["actual_tool_trace"][0]["response"]["date"] = "forged"
    elif tamper == "ledger":
        ledger[0]["value_sha256"] = "0" * 64
    else:
        sidecar["original"]["original_messages"][2]["content"] = "fabricated"
    with pytest.raises(ValueError):
        ablation.verify_omissions(request, sidecar, ledger, "grouped_actor_prose_withheld")


def test_paid_operation_requires_explicit_gate_before_key_or_network(monkeypatch, tmp_path):
    monkeypatch.setattr(ablation, "read_key", lambda: pytest.fail("key read"))
    monkeypatch.setattr(ablation, "call", lambda *a: pytest.fail("network call"))
    with pytest.raises(ValueError, match="explicit --allow-paid"):
        ablation.run(tmp_path)
    assert not (tmp_path / "started.json").exists()


def test_existing_start_refuses_retry_before_key(monkeypatch, tmp_path):
    monkeypatch.setattr(ablation, "check", lambda root: ({"jobs": []}, {}))
    monkeypatch.setattr(ablation, "read_key", lambda: pytest.fail("key read"))
    (tmp_path / "started.json").write_text("{}")
    with pytest.raises(FileExistsError, match="no retry"):
        ablation.run(tmp_path, allow_paid=True)


@pytest.mark.parametrize("tamper", ["receipt_request", "usage_total"])
def test_offline_audit_rejects_saved_receipt_tampering(monkeypatch, tmp_path, tamper):
    original, trace, _ = sample()
    job = {
        "id": "confirm-fresh-09-en-m0-chronological",
        "history_id": "confirm-fresh-09-en-m0",
        "arm": "chronological",
    }
    history = {"id": job["history_id"], "request": original, "tool_trace": trace}
    package = {"jobs": [job], "parent_root": "parent"}
    monkeypatch.setattr(ablation, "check", lambda root: (package, {history["id"]: history}))
    request, _, _ = ablation.render(original, trace, job["arm"])
    save(tmp_path / "started.json", {"manifest_sha256": digest(package)})
    save(
        tmp_path / "execution.json",
        {
            "outcomes": [{"id": job["id"], "terminal": "answer"}],
            "meter": {},
        },
    )
    calls = tmp_path / "calls"
    calls.mkdir()
    with sqlite3.connect(calls / "meter.sqlite") as db:
        db.execute(
            "CREATE TABLE calls (id TEXT PRIMARY KEY, stage TEXT, reservation INTEGER, "
            "charged INTEGER, usage_cost INTEGER, status TEXT)"
        )
        db.execute(
            "INSERT INTO calls VALUES (?,?,?,?,?,?)",
            (
                job["id"],
                ablation.STAGE,
                ablation.reservation(request),
                375,
                300,
                "known",
            ),
        )
    save(
        calls / (job["id"] + ".request.json"),
        {
            "endpoint": ablation.ENDPOINT,
            "request": request,
            "sha256": digest(request),
        },
    )
    save(
        calls / (job["id"] + ".dispatch.json"),
        {
            "dispatch_time": 1.0,
            "pacing_control": {"rpm": 90},
        },
    )
    body = {
        "model": request["model"],
        "id": "response-1",
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 0,
            "total_tokens": 21 if tamper == "usage_total" else 20,
        },
    }
    save(
        calls / (job["id"] + ".receipt.json"),
        {
            "job": job["id"],
            "request_sha256": "forged" if tamper == "receipt_request" else digest(request),
            "status": 200,
            "body_text": json.dumps(body),
            "elapsed_seconds": 0.1,
        },
    )
    save(tmp_path / (job["id"] + ".result.json"), {"terminal": "answer"})
    parent = tmp_path / "parent"
    parent.mkdir()
    save(parent / "runtime.json", [])
    save(parent / "gold.json", {})
    with pytest.raises(ValueError, match="Receipt/request binding|Usage/charge"):
        ablation.audit(tmp_path)


def _synthetic_parent_and_config(base):
    parent = base / "native-parent"
    root = base / "prepared"
    parent.mkdir()
    root.mkdir()
    source_messages = json.loads(sample(with_receipt=False)[2]["original_messages"][1]["content"])[
        "source_messages"
    ]
    runtime = [
        {
            "id": f"fresh-{workflow:02d}-{language}",
            "workflow": f"fresh-{workflow:02d}",
            "language": language,
            "source_messages": source_messages,
            "intent": "read",
            "appointment_id": "BOOK-X",
            "requested_change": None,
        }
        for workflow in range(40)
        for language in ("en", "hi", "te")
    ]
    save(parent / "runtime.json", runtime)
    state = {"appointments": []}
    expected = {"answer": None, "needs_clarification": True}
    save(
        parent / "gold.json",
        {
            c["id"]: {
                "runtime_sha256": digest(c),
                "question_target": "source_request_or_condition",
                "expected_answer": expected,
                "expected_state": state,
                "forbid_new_write": False,
            }
            for c in runtime
        },
    )
    parent_jobs = [
        {"id": f"confirm-{c['id']}-m{i}", "case_id": c["id"]} for c in runtime for i in range(2)
    ]
    parent_manifest = {"jobs": parent_jobs, "source_sha256": {}}
    save(parent / "manifest.json", parent_manifest)
    save(parent / "started.json", {"manifest_sha256": digest(parent_manifest)})
    save(
        parent / "execution.json",
        {
            "outcomes": [{"id": j["id"]} for j in parent_jobs],
        },
    )
    save(
        parent / "verification.json",
        {
            "verified": True,
            "complete": True,
            "native_episodes": 240,
            "unknown_calls": 0,
            "research_calls": 6431,
            "research_known_inr": 180.0,
            "research_retained_inr": 258.0,
        },
    )
    review_files = {}
    for language in ("en", "hi", "te"):
        name = f"semantic-review-{language}-complete.json"
        save(
            parent / name,
            {
                "reviewer": "assistant language judgment, not native-human review",
                "reviewed_episode_ids": [
                    j["id"] for j in parent_jobs if j["case_id"].endswith("-" + language)
                ],
                "reporter_reviews": [
                    {
                        "episode_id": j["id"],
                        "variant": variant,
                        "factual_verdict": "correct",
                        "status_verdict": "correct",
                    }
                    for j in parent_jobs
                    if j["case_id"].endswith("-" + language)
                    for variant in ("control", "candidate")
                ],
            },
        )
        review_files[language] = {
            "file": name,
            "sha256": hashlib.sha256((parent / name).read_bytes()).hexdigest(),
        }
    save(
        parent / "semantic-review.json",
        {
            "status": "ASSISTANT_LANGUAGE_REVIEW_COMPLETE",
            "parent_verification_sha256": hashlib.sha256(
                (parent / "verification.json").read_bytes()
            ).hexdigest(),
            "episodes": [
                {
                    "id": j["id"],
                    "reporters": ["control", "candidate"],
                    "source_gold_ambiguous": False,
                }
                for j in parent_jobs
            ],
            "review_files": review_files,
        },
    )
    save(parent / "review.json", {})
    save(parent / "budget.json", {})
    save(parent / "pacing-control.json", {"rpm": 90})
    (parent / "PROTOCOL.md").write_text("Synthetic parent fixture only.\n")
    (parent / "source").mkdir()
    selected_ids = [
        f"{workflow}-{language}"
        for workflows in ablation.SELECTED_WORKFLOWS.values()
        for workflow in workflows
        for language in ("en", "hi", "te")
    ]
    selection = base / "selection.json"
    save(
        selection,
        {
            "status": "PROSPECTIVE_SOURCE_ONLY_SELECTION_NO_PAID_RUN",
            "gold_or_saved_answers_read": False,
            "planned_reporter_calls_if_qualified": 108,
            "selected_workflows": {k: list(v) for k, v in ablation.SELECTED_WORKFLOWS.items()},
            "case_ids": selected_ids,
            "runtime_file_sha256": hashlib.sha256(
                (parent / "runtime.json").read_bytes()
            ).hexdigest(),
        },
    )
    for case_id in selected_ids:
        for i, model in enumerate(ablation.PROFILES):
            episode_id = f"confirm-{case_id}-m{i}"
            request, trace, evidence = sample(with_receipt=False)
            request["model"] = model
            request["reasoning_effort"] = ablation.PROFILES[model]
            paired = {
                "original": {
                    "id": episode_id,
                    "case_id": case_id,
                    "model": model,
                    "history": evidence["original_messages"],
                    "tool_trace": trace,
                    "seed_receipts": [],
                    "initial_backend": state,
                    "final_backend": state,
                },
                "reports": {},
                "actual_calls": [],
            }
            folder = parent / "episodes" / episode_id
            folder.mkdir(parents=True)
            save(folder / "paired.json", paired)
            calls = parent / "calls"
            calls.mkdir(exist_ok=True)
            save(
                calls / (episode_id + "-final-candidate.request.json"),
                {
                    "endpoint": ablation.ENDPOINT,
                    "request": request,
                    "sha256": digest(request),
                },
            )
    save(
        root / "budget.json",
        {
            "phase_cap_microinr": 240_000_000,
            "campaign_cap_microinr": 6_000_000_000,
            "prior_calls": 6431,
            "prior_known_microinr": 180_000_000,
            "prior_retained_microinr": 258_000_000,
            "stages": {ablation.STAGE: {"calls": 108, "cap_microinr": 240_000_000}},
        },
    )
    save(root / "pacing-control.json", {"rpm": 90})
    (root / "PROTOCOL.md").write_text("Synthetic, offline-only reporter screen.\n")
    return parent, root, selection


def _mock_108_receipts(root, package, inputs):
    calls = root / "calls"
    calls.mkdir()
    unit_cost = 100 * ablation.INPUT + 10 * ablation.OUTPUT
    unit_charge = (unit_cost * 5 + 3) // 4
    with sqlite3.connect(calls / "meter.sqlite") as db:
        db.execute(
            "CREATE TABLE calls (id TEXT PRIMARY KEY, stage TEXT, reservation INTEGER, "
            "charged INTEGER, usage_cost INTEGER, status TEXT)"
        )
        for index, job in enumerate(package["jobs"]):
            history = inputs[job["history_id"]]
            request, _, _ = ablation.render(history["request"], history["tool_trace"], job["arm"])
            body = {
                "id": f"mock-{index}",
                "model": request["model"],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": canonical(
                                {
                                    "answer": None,
                                    "needs_clarification": True,
                                    "action_status": "not_requested",
                                }
                            )
                        },
                    }
                ],
            }
            save(
                calls / (job["id"] + ".request.json"),
                {
                    "endpoint": ablation.ENDPOINT,
                    "request": request,
                    "sha256": digest(request),
                },
            )
            save(
                calls / (job["id"] + ".receipt.json"),
                {
                    "job": job["id"],
                    "request_sha256": digest(request),
                    "status": 200,
                    "body_text": json.dumps(body),
                    "elapsed_seconds": 0.01,
                },
            )
            save(
                calls / (job["id"] + ".dispatch.json"),
                {
                    "dispatch_time": float(index),
                    "pacing_control": {"rpm": 90},
                },
            )
            result = ablation.classify(body)
            save(root / (job["id"] + ".result.json"), result)
            db.execute(
                "INSERT INTO calls VALUES (?,?,?,?,?,?)",
                (
                    job["id"],
                    ablation.STAGE,
                    ablation.reservation(request),
                    unit_charge,
                    unit_cost,
                    "known",
                ),
            )
    save(root / "started.json", {"manifest_sha256": digest(package)})
    save(
        root / "execution.json",
        {
            "outcomes": [{"id": j["id"], "terminal": "answer"} for j in package["jobs"]],
            "meter": {
                ablation.STAGE: {
                    "calls": 108,
                    "charged_with_headroom_inr": 108 * unit_charge / 1e6,
                    "known_usage_estimate_inr": 108 * unit_cost / 1e6,
                    "unknown_or_pending": 0,
                }
            },
        },
    )


def test_full_mock_campaign_relocates_and_audits_without_parent_or_network(tmp_path, monkeypatch):
    parent, prepared, selection = _synthetic_parent_and_config(tmp_path)
    package = ablation.prepare(prepared, parent, selection, parent / "verification.json")
    assert package["maximum_calls"] == 108
    assert package["worst_case_retained_microinr"] == 232_243_200
    assert package["parent_root"] == "provenance/parent"
    assert all(
        not value.startswith(str(tmp_path))
        for value in (package["parent_root"], package["selection_file"])
    )
    order = {}
    for job in package["jobs"]:
        order.setdefault(job["history_id"], []).append(job["arm"])
    assert len(order) == 36
    assert all(set(arms) == set(ablation.ARMS) for arms in order.values())
    relocated = tmp_path / "relocated"
    shutil.copytree(prepared, relocated)
    shutil.rmtree(prepared)
    shutil.rmtree(parent)
    selection.unlink()
    frozen, inputs = ablation.check(relocated)
    assert frozen == package
    _mock_108_receipts(relocated, frozen, inputs)
    monkeypatch.setattr(ablation, "read_key", lambda: pytest.fail("key read"))
    monkeypatch.setattr(ablation, "call", lambda *args: pytest.fail("network call"))
    report = ablation.audit(relocated)
    assert report["verified"] and report["complete"] and report["calls"] == 108
    assert all(group["n"] == 18 for group in report["groups"])
    assert all(group["factual_answer_correct"] == 18 for group in report["groups"])
    empty_id = frozen["jobs"][1]["id"]
    empty_path = relocated / "calls" / (empty_id + ".receipt.json")
    empty_receipt = json.loads(empty_path.read_text())
    empty_body = json.loads(empty_receipt["body_text"])
    empty_body["choices"][0]["message"]["content"] = ""
    empty_receipt["body_text"] = json.dumps(empty_body)
    empty_path.write_text(json.dumps(empty_receipt))
    (relocated / (empty_id + ".result.json")).write_text(json.dumps(ablation.classify(empty_body)))
    empty_report = ablation.audit(relocated)
    assert empty_report["complete"]
    assert (
        next(row for row in empty_report["rows"] if row["id"] == empty_id)["valid_final"] is False
    )
    first = frozen["jobs"][0]["id"]
    path = relocated / "calls" / (first + ".receipt.json")
    broken = json.loads(path.read_text())
    broken_body = json.loads(broken["body_text"])
    del broken_body["choices"]
    broken["body_text"] = json.dumps(broken_body)
    path.write_text(json.dumps(broken))
    (relocated / (first + ".result.json")).write_text(
        json.dumps({"terminal": "infrastructure_error", "error_class": "KeyError"})
    )
    execution = json.loads((relocated / "execution.json").read_text())
    execution["outcomes"][0]["terminal"] = "infrastructure_error"
    (relocated / "execution.json").write_text(json.dumps(execution))
    infrastructure_report = ablation.audit(relocated)
    assert not infrastructure_report["complete"]
    forged = json.loads(path.read_text())
    forged["request_sha256"] = "forged"
    path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="Receipt/request binding"):
        ablation.audit(relocated)


def test_incomplete_semantic_review_blocks_prepare_before_freeze(tmp_path):
    parent, root, selection = _synthetic_parent_and_config(tmp_path)
    path = parent / "semantic-review.json"
    review = json.loads(path.read_text())
    review["episodes"][-1]["reporters"] = ["control"]
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="240/480 reporter coverage"):
        ablation.prepare(root, parent, selection, parent / "verification.json")
    assert not (root / "inputs.json").exists()
    assert not (root / "manifest.json").exists()


def test_language_review_missing_reporter_blocks_even_with_matching_hash(tmp_path):
    parent, root, selection = _synthetic_parent_and_config(tmp_path)
    language_path = parent / "semantic-review-hi-complete.json"
    language_review = json.loads(language_path.read_text())
    language_review["reporter_reviews"].pop()
    language_path.write_text(json.dumps(language_review))
    aggregate_path = parent / "semantic-review.json"
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["review_files"]["hi"]["sha256"] = hashlib.sha256(
        language_path.read_bytes()
    ).hexdigest()
    aggregate_path.write_text(json.dumps(aggregate))
    with pytest.raises(ValueError, match="160 reporter judgments"):
        ablation.prepare(root, parent, selection, parent / "verification.json")
    assert not (root / "inputs.json").exists()


def test_selected_source_gold_ambiguity_blocks_prepare(tmp_path):
    parent, root, selection = _synthetic_parent_and_config(tmp_path)
    path = parent / "semantic-review.json"
    review = json.loads(path.read_text())
    selected = next(row for row in review["episodes"] if row["id"] == "confirm-fresh-09-en-m0")
    selected["source_gold_ambiguous"] = True
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="source/gold ambiguity"):
        ablation.prepare(root, parent, selection, parent / "verification.json")
    assert not (root / "inputs.json").exists()
