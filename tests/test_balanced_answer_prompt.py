import hashlib

from benchmarks.balanced_answer_prompt import BALANCED_REASONING_PROMPT


def test_balanced_prompt_is_the_measured_candidate():
    assert hashlib.sha256(BALANCED_REASONING_PROMPT.encode()).hexdigest() == (
        "a029463451e4a5041923ed1da981146726366ec0d879225e0d61348a495c229e"
    )


def test_balanced_prompt_formats_and_requires_matching_evidence():
    rendered = BALANCED_REASONING_PROMPT.format(
        context="stored memories", question="What changed?", today_line=""
    )
    assert "stored memories" in rendered
    assert "What changed?" in rendered
    assert "exact entity, requested attribute and relevant time" in rendered
    assert "inventing an estimate" in rendered
    assert "PARTIAL EVIDENCE IS STILL EVIDENCE" not in rendered
