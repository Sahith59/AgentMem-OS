import json
import sqlite3

import pytest

from benchmarks.sarvam_native_study.transport import Meter, payload, save


def test_unknown_retains_reservation_and_no_duplicate_dispatch(tmp_path):
    meter = Meter(tmp_path)
    req = payload("sarvam-105b", [])
    reservation = meter.reserve("one", "setup", req)
    assert not meter.settle("one", {}, req["model"], req["max_tokens"])
    assert meter.summary()["setup"]["charged_with_headroom_inr"] == reservation / 1e6
    with pytest.raises(sqlite3.IntegrityError):
        meter.reserve("one", "setup", req)


def test_known_usage_settles_with_headroom(tmp_path):
    meter = Meter(tmp_path)
    req = payload("sarvam-105b", [])
    meter.reserve("one", "setup", req)
    assert meter.settle(
        "one",
        {"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}},
        req["model"],
        4096,
    )
    s = meter.summary()["setup"]
    assert s["known_usage_estimate_inr"] == 0.0027
    assert s["charged_with_headroom_inr"] == 0.003375
    with pytest.raises(ValueError):
        meter.settle("one", {}, req["model"], 4096)


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": True, "completion_tokens": 0, "total_tokens": 1},
        {"prompt_tokens": 1, "completion_tokens": 5000, "total_tokens": 5001},
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 8},
    ],
)
def test_invalid_usage_keeps_reservation(tmp_path, usage):
    meter = Meter(tmp_path)
    meter.reserve("a", "setup", payload("sarvam-105b", []))
    assert not meter.settle("a", {"usage": usage}, "sarvam-105b", 4096)
    assert meter.summary()["setup"]["unknown_or_pending"] == 1


def test_budget_ceiling_and_immutable_artifact(tmp_path):
    meter = Meter(tmp_path)
    with meter.connect() as db:
        db.execute("INSERT INTO calls VALUES ('used','setup',300000000,300000000,NULL,'unknown')")
    with pytest.raises(ValueError, match="Stage ceiling"):
        meter.reserve("a", "setup", payload("sarvam-105b", []))
    target = tmp_path / "receipt.json"
    save(target, {"old": 1})
    with pytest.raises(FileExistsError):
        save(target, {"new": 2})
    assert json.loads(target.read_text()) == {"old": 1}
