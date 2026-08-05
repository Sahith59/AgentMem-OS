"""
LongMemEval's OFFICIAL judge, ported verbatim from the benchmark authors'
`src/evaluation/evaluate_qa.py` (github.com/xiaowu0162/LongMemEval).

Why this file exists — a real defect found 2026-08-05:

This repo was judging every LongMemEval question with ONE generic
"does the prediction convey the same facts as the gold answer" prompt. The
benchmark's authors specify **five different judge prompts**, chosen by
question type, and the differences are not cosmetic:

  • temporal-reasoning — the official prompt says "do NOT penalize off-by-one
    errors for the number of days." Ours penalized them. LongMemEval's own
    gold answers acknowledge this ambiguity in their text ("6 days. 7 days
    (including the last day) is also acceptable").

  • knowledge-update — official: "If the response contains some previous
    information along with an updated answer, the response should be
    considered as correct as long as the updated answer is the required
    answer." Ours marked those wrong. This is exactly the category where a
    memory system that correctly surfaces BOTH the old and superseding fact
    gets punished for being thorough.

  • single-session-preference — the gold field is a **RUBRIC describing what a
    personalized response should do**, not an answer. Official: "The model
    does not need to reflect all the points in the rubric. The response is
    correct as long as it recalls and utilizes the user's personal
    information correctly." Ours compared a helpful answer against a rubric
    as if the rubric were a fact, so the category scored near zero *even with
    perfect retrieval* (measured: 2/10 at the oracle ceiling).

  • abstention (question_id ending in "_abs", 30 of 500) — has its own prompt
    entirely: the model is correct if it *recognizes the question cannot be
    answered*. Ours had no concept of this and scored them as ordinary
    factual misses.

Consequence: our numbers were not comparable to any published LongMemEval
figure, and were biased DOWNWARD — most against the categories a good memory
system should win. Every vendor number we compare against uses this judge.

The official harness also answers "yes"/"no" (not CORRECT/INCORRECT) and pins
the judge to gpt-4o-2024-08-06.
"""
from __future__ import annotations

_STANDARD = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no."
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. In addition, "
    "do not penalize off-by-one errors for the number of days. If the question asks for the "
    "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
    "predicting 19 days when the answer is 18), the model's response is still correct."
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, the "
    "response should be considered as correct as long as the updated answer is the required "
    "answer."
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response "
    "from a model. Please answer yes if the response satisfies the desired response. "
    "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
    "The response is correct as long as it recalls and utilizes the user's personal "
    "information correctly."
    "\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. The "
    "model could say that the information is incomplete, or some other information is given "
    "but the asked information is not."
    "\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer yes or no only."
)


def build_judge_prompt(question_type: str, question: str, gold: str, response: str,
                        abstention: bool = False) -> str:
    """
    The official per-question-type judge prompt. `abstention` is True when the
    question_id ends in "_abs" — those get their own template regardless of type.
    Unknown/missing types fall back to the standard template (which is what the
    non-LongMemEval datasets get).
    """
    if abstention:
        return _ABSTENTION.format(question, gold, response)
    if question_type == "temporal-reasoning":
        return _TEMPORAL.format(question, gold, response)
    if question_type == "knowledge-update":
        return _KNOWLEDGE_UPDATE.format(question, gold, response)
    if question_type == "single-session-preference":
        return _PREFERENCE.format(question, gold, response)
    # single-session-user / single-session-assistant / multi-session / unknown
    return _STANDARD.format(question, gold, response)


def parse_judge_verdict(raw: str) -> bool:
    """
    The official harness asks for "yes"/"no". Check the negative first — "no"
    is a substring of nothing dangerous here, but a bare "yes" can appear
    inside a longer refusal, so anchor on the leading token when possible.
    """
    out = (raw or "").strip().lower()
    if out.startswith("no"):
        return False
    if out.startswith("yes"):
        return True
    return "yes" in out and "no" not in out


def is_abstention(question_id: str) -> bool:
    return str(question_id or "").endswith("_abs")
