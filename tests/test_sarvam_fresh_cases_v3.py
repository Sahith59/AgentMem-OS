from collections import Counter

from benchmarks.sarvam_fresh_cases_v2 import build as old_build
from benchmarks.sarvam_fresh_cases_v3 import build
from benchmarks.sarvam_native_study.transport import digest


def test_preference_questions_preserve_all_evidence_and_gold():
    before, old_gold = old_build()
    after, gold = build()
    assert len(after) == 120
    changes = []
    for a, b in zip(before, after):
        assert a["id"] == b["id"]
        assert {k: v for k, v in a.items() if k != "question"} == {
            k: v for k, v in b.items() if k != "question"
        }
        assert {k: v for k, v in old_gold[a["id"]].items() if k != "runtime_sha256"} == {
            k: v for k, v in gold[b["id"]].items() if k != "runtime_sha256"
        }
        assert gold[b["id"]]["runtime_sha256"] == digest(b)
        if a["question"] != b["question"]:
            changes.append(b)
    assert len(changes) == 12
    assert {c["family"] for c in changes} == {"separate_booking"}
    assert Counter(c["language"] for c in changes) == {"en": 4, "hi": 4, "te": 4}
    for c in changes:
        marker = {"en": "preferred date", "hi": "पसंद", "te": "ఇష్టమైన తేదీ"}[c["language"]]
        assert marker in c["question"]
        assert gold[c["id"]]["expected_answer"]["answer"] is not None
