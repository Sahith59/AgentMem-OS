import hashlib

import pytest

from agentmem_os.benchmarks.verify_source_supplement import verify_packet


def sample():
    baseline = "old context"
    body = "[2024/05/01] Yoga is three times weekly."
    digest = hashlib.sha256(body.encode()).hexdigest()
    prefix = baseline + f"\n\n[ADDITIONAL SOURCE EVIDENCE]\n[source {digest[:16]} | user]\n"
    candidate = prefix + body + "\n"
    receipt = {"role": "user", "source_sha256": digest,
               "packet_start": len(prefix), "packet_end": len(prefix) + len(body)}
    return baseline, candidate, [{"role": "user", "content": body}], [receipt]


def test_valid_independently_constructed_receipt():
    assert verify_packet(*sample()) == 1


@pytest.mark.parametrize("change", ["prefix", "role", "scope", "hash", "offset", "unreceipted", "cap"])
def test_rejects_corrupted_evidence(change):
    baseline, candidate, turns, receipts = sample()
    kwargs = {}
    if change == "prefix":
        candidate = "X" + candidate[1:]
    elif change == "role":
        receipts[0]["role"] = "assistant"
    elif change == "scope":
        turns = [{"role": "user", "content": "another source"}]
    elif change == "hash":
        receipts[0]["source_sha256"] = "0" * 64
    elif change == "offset":
        receipts[0]["packet_end"] -= 1
    elif change == "unreceipted":
        candidate += "UNSUPPORTED CLAIM"
    else:
        kwargs["char_cap"] = len(candidate) - 1
    with pytest.raises(ValueError):
        verify_packet(baseline, candidate, turns, receipts, **kwargs)
