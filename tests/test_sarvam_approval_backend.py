import copy
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from benchmarks.sarvam_native_controls import approval_backend as backend

CALLER = {"tenant": "shop", "customer": "customer-1"}
ARGS = {
    "appointment_id": "BOOK-1",
    "date": "2026-12-23",
    "expected_version": 3,
    "idempotency_key": "CHANGE-1",
    "approval_id": "APPROVAL-1",
}


def fixture():
    return {
        "issuers": ["supervisor-1"],
        "appointments": [
            {**CALLER, "id": "BOOK-1", "date": "2026-11-23", "version": 3, "closed": False}
        ],
        "approvals": [
            {
                **CALLER,
                "id": "APPROVAL-1",
                "appointment_id": "BOOK-1",
                "operation": "update_appointment",
                "date": "2026-12-23",
                "expected_version": 3,
                "issuer": "supervisor-1",
                "issued_at": 50,
                "not_before": 80,
                "expires_at": 200,
                "granted": True,
                "revoked": False,
            }
        ],
    }


def make(tmp_path, data=None):
    p = tmp_path / "backend.sqlite"
    backend.create(p, data or fixture())
    return p


def execute(p, args=None, **kw):
    return backend.execute(p, args or ARGS, caller=CALLER, now=kw.pop("now", 100), **kw)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"granted": False}, "approval_pending"),
        ({"revoked": True}, "approval_revoked"),
        ({"expires_at": 100}, "approval_expired"),
        ({"not_before": 101}, "approval_not_effective"),
        ({"issued_at": 101}, "approval_not_effective"),
        ({"issuer": "model-invented-supervisor"}, "untrusted_issuer"),
        ({"date": "2026-12-24"}, "approval_scope_mismatch"),
        ({"expected_version": 2}, "approval_scope_mismatch"),
        ({"appointment_id": "OTHER"}, "approval_scope_mismatch"),
        ({"operation": "cancel_appointment"}, "approval_scope_mismatch"),
        ({"tenant": "other-tenant"}, "approval_missing"),
        ({"customer": "other-customer"}, "approval_missing"),
        ({"id": "other-approval"}, "approval_missing"),
    ],
)
def test_guard_blocks_bad_approval_that_unenforced_control_accepts(tmp_path, change, reason):
    data = fixture()
    data["approvals"][0].update(change)
    candidate = make(tmp_path, data)
    control = tmp_path / "control.sqlite"
    backend.create(control, data)
    before = backend.inspect_state(candidate)
    assert execute(candidate) == {
        "status": "rejected",
        "reason": reason,
        "date": "2026-11-23",
        "version": 3,
    }
    assert backend.inspect_state(candidate) == before
    assert execute(control, enforce=False)["status"] == "confirmed"
    assert backend.current(control, "BOOK-1", caller=CALLER)["date"] == "2026-12-23"


def test_valid_update_and_duplicate_survive_later_expiry_revocation(tmp_path):
    p = make(tmp_path)
    first = execute(p)
    assert first == {"status": "confirmed", "date": "2026-12-23", "version": 4}
    before = backend.inspect_state(p)
    with backend.connect(p) as db:
        db.execute("UPDATE approvals SET revoked=1")
    duplicate = execute(p, now=201)
    assert duplicate == {
        "status": "already_applied",
        "date": "2026-12-23",
        "version": 4,
        "original_receipt": first,
    }
    assert backend.inspect_state(p) == before
    assert execute(p, {**ARGS, "date": "2026-12-24"})["reason"] == "idempotency_conflict"
    assert execute(p, {**ARGS, "approval_id": "DIFFERENT"})["reason"] == "idempotency_conflict"
    assert backend.inspect_state(p) == before


def test_duplicate_returns_current_state_separately_from_original_receipt(tmp_path):
    p = make(tmp_path)
    first = execute(p)
    with backend.connect(p) as db:
        db.execute("UPDATE appointments SET date='2026-12-28',version=5")
    result = execute(p)
    assert result["date"] == "2026-12-28" and result["version"] == 5
    assert result["original_receipt"] == first


@pytest.mark.parametrize(
    "caller",
    [{"tenant": "other", "customer": "customer-1"}, {"tenant": "shop", "customer": "other"}],
)
def test_caller_cannot_read_or_write_other_identity(tmp_path, caller):
    p = make(tmp_path)
    before = backend.inspect_state(p)
    refusal = {"status": "rejected", "reason": "not_found_or_forbidden"}
    for enforce in (False, True):
        assert backend.execute(p, ARGS, caller=caller, now=100, enforce=enforce) == refusal
    assert backend.current(p, "BOOK-1", caller=caller) == refusal
    assert backend.inspect_state(p) == before


@pytest.mark.parametrize(
    ("update", "reason"),
    [
        ({"closed": True}, "closed"),
        ({"version": 4}, "version_mismatch"),
    ],
)
def test_closed_and_stale_versions_never_change(tmp_path, update, reason):
    data = fixture()
    data["appointments"][0].update(update)
    p = make(tmp_path, data)
    before = backend.inspect_state(p)
    for enforce in (False, True):
        assert execute(p, enforce=enforce)["reason"] == reason
    assert backend.inspect_state(p) == before


@pytest.mark.parametrize(
    "args",
    [
        {**ARGS, "caller": CALLER},
        {**ARGS, "enforce": False},
        {**ARGS, "now": 90},
        {**ARGS, "expected_version": True},
        {**ARGS, "expected_version": -1},
        {**ARGS, "date": "2026-02-30"},
        {**ARGS, "date": "20261223"},
        {**ARGS, "approval_id": ""},
        {},
        None,
    ],
)
def test_malformed_or_injected_arguments_never_write(tmp_path, args):
    p = make(tmp_path)
    before = backend.inspect_state(p)
    assert backend.execute(p, args, caller=CALLER, now=100)["status"] == "rejected"
    assert backend.inspect_state(p) == before


def test_missing_server_approval_cannot_be_replaced_by_model_claim(tmp_path):
    data = fixture()
    data["approvals"] = []
    p = make(tmp_path, data)
    assert execute(p, {**ARGS, "approval_id": "customer said yes"})["reason"] == "approval_missing"
    assert backend.current(p, "BOOK-1", caller=CALLER)["version"] == 3


@pytest.mark.parametrize("same_key", [True, False])
def test_concurrent_requests_commit_at_most_once(tmp_path, same_key):
    p = make(tmp_path)
    requests = [copy.deepcopy(ARGS) for _ in range(8)]
    if not same_key:
        for index, args in enumerate(requests):
            args["idempotency_key"] += str(index)
    with ThreadPoolExecutor(max_workers=8) as pool:
        replies = list(pool.map(lambda args: execute(p, args), requests))
    assert sum(r["status"] == "confirmed" for r in replies) == 1
    assert sum(r["status"] == "already_applied" for r in replies) == (7 if same_key else 0)
    assert sum(r.get("reason") == "version_mismatch" for r in replies) == (0 if same_key else 7)
    state = backend.inspect_state(p)
    assert len(state["operations"]) == 1 and state["appointments"][0]["version"] == 4


def test_failed_receipt_commit_rolls_back_date_update(tmp_path):
    p = make(tmp_path)
    before = backend.inspect_state(p)
    with backend.connect(p) as db:
        db.execute(
            "CREATE TRIGGER fail_receipt BEFORE INSERT ON operations BEGIN "
            "SELECT RAISE(ABORT,'injected write failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        execute(p)
    assert backend.inspect_state(p) == before


def test_read_only_and_invalid_server_context(tmp_path):
    p = make(tmp_path)
    before = backend.inspect_state(p)
    assert backend.current(p, "BOOK-1", caller=CALLER) == {
        "date": "2026-11-23",
        "version": 3,
        "closed": False,
    }
    with pytest.raises(ValueError):
        backend.execute(p, ARGS, caller={}, now=100)
    with pytest.raises(ValueError):
        backend.execute(p, ARGS, caller=CALLER, now=True)
    with pytest.raises(ValueError):
        execute(p, enforce="false")
    assert backend.inspect_state(p) == before
