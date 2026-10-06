import copy
import json

import pytest

from benchmarks import guarded_answer_screen as old
from benchmarks import guarded_answer_screen_normalized as new
from benchmarks.answer_screen_continuation import bootstrap, inherit, load_sources
from tests.test_guarded_answer_screen import auth, package, provider


def prepare(tmp_path):
    p = package()
    source = tmp_path / "source"
    a = auth(p, source)
    fake, calls = provider()

    def overlap(req):
        result = fake(req)
        body = json.loads(result["text"])
        body["support_turn_ids"] = body["qualification_turn_ids"][:]
        result["text"] = json.dumps(body)
        return result

    with pytest.raises(RuntimeError):
        old.run(p, source, a, mode="offline-test", provider=overlap)
    records = {}
    for name, data in [("package", p), ("approval", a)]:
        path = source / (name + ".json")
        path.write_text(json.dumps(data))
        records[name] = dict(path=str(path), sha256=old.file_hash(path))
    cp = source / "checkpoint.json"
    records["checkpoint"] = dict(path=str(cp), sha256=old.file_hash(cp))
    state = json.loads(cp.read_text())
    target = json.loads(json.dumps(p))
    target["code_sha256"] = new.code_hashes()
    cap = p["maximum_reservation_nusd"] - state["reserved_nusd"] + 1000
    target["continuation"] = dict(
        sources=records,
        source_checkpoint_sha256=records["checkpoint"]["sha256"],
        source_reserved_nusd=state["reserved_nusd"],
        inherited_jobs=list(state["jobs"]),
        additional_cap_nusd=cap,
    )
    dest = tmp_path / "continued"
    approval = dict(
        approved=True,
        mode="offline-test",
        authorization_text="Fake continuation",
        package_sha256=new.validate(target),
        maximum_attempts=8,
        maximum_new_attempts=7,
        additional_budget_nusd=cap,
        budget_nusd=state["reserved_nusd"] + cap,
        output_directory=str(dest),
    )
    return target, dest, approval, records, calls


def test_carry_response_without_another_planner_request(tmp_path):
    target, dest, a, records, old_calls = prepare(tmp_path)
    state = bootstrap(target, dest, a, mode="offline-test")
    assert state["jobs"]["test:plan"]["inherited_original_status"] == "error"
    fake, calls = provider()

    # Give new receipts distinct identities from the inherited paid receipt.
    def unique(req):
        result = fake(req)
        result["id"] = "continued-" + result["id"]
        return result

    result = new.run(target, dest, a, mode="offline-test", provider=unique)
    assert result["completed"] == 8 and len(calls) == 7 and len(old_calls) == 1
    assert all("response_format" not in req for req in calls)
    assert (
        json.loads(open(records["checkpoint"]["path"]).read())["jobs"]["test:plan"]["status"]
        == "error"
    )
    new.run(target, dest, a, mode="offline-test", provider=unique)
    assert len(calls) == 7


def test_cannot_restart_all_calls_without_bootstrap(tmp_path):
    target, dest, a, _, _ = prepare(tmp_path)
    fake, calls = provider()
    with pytest.raises(ValueError, match="bootstrap"):
        new.run(target, dest, a, mode="offline-test", provider=fake)
    assert calls == []


def test_changed_population_and_approval_are_rejected(tmp_path):
    target, dest, a, records, _ = prepare(tmp_path)
    broken = copy.deepcopy(target)
    broken["cases"][0]["gold"] = "changed"
    with pytest.raises(ValueError, match="population"):
        inherit(broken, load_sources(records))
    a["maximum_new_attempts"] = 8
    with pytest.raises(ValueError, match="approval"):
        bootstrap(target, dest, a, mode="offline-test")


def test_changed_inherited_artifact_and_existing_directory_rejected(tmp_path):
    target, dest, a, records, _ = prepare(tmp_path)
    bootstrap(target, dest, a, mode="offline-test")
    with pytest.raises(FileExistsError):
        bootstrap(target, dest, a, mode="offline-test")
    from pathlib import Path

    path = Path(records["checkpoint"]["path"])
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="artifact"):
        load_sources(records)


def test_provenance_tag_and_insufficient_remaining_cap_rejected(tmp_path):
    target, dest, a, records, _ = prepare(tmp_path)
    bad = copy.deepcopy(target)
    bad["continuation"]["source_checkpoint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="provenance"):
        inherit(bad, load_sources(records))
    bad = copy.deepcopy(target)
    bad["continuation"]["additional_cap_nusd"] = 0
    with pytest.raises(ValueError, match="remaining"):
        new.validate(bad)
