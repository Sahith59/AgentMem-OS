"""Offline semantic-review material, never an answer packet or an automatic judge.

Reviewers need the full eligible source pool, including distant qualifications,
and the legacy packet to distinguish a new source from new information. This
module adds no ranking, model, database, semantic approval, or runtime adapter.
"""

import json
import math
from dataclasses import asdict
from numbers import Real

from agentmem_os.llm.evidence_packet import RetrievalHit, digest, eligible_sources
from agentmem_os.llm.source_unit_compiler import compile_unit, snapshot_digest
from agentmem_os.llm.source_unit_contract import DEPENDENCY_KINDS, SourceUnit

REVIEW_QUESTIONS = {
    "entity": "Whose fact is this? Are similarly named people or objects distinguished?",
    "event": "Which event does this describe? Are repeated mentions distinct events?",
    "time": "Which observation, event, relative date and question window apply?",
    "negation": "Does any eligible source deny or limit the proposed claim?",
    "modality": "Is this completed, planned, hypothetical, considered or uncertain?",
    "role": "Is this a user assertion, an assistant suggestion, or requested past advice?",
    "membership": "Does each quantity or item belong to the question's requested set?",
    "correction": "Does an earlier or later eligible source correct this information?",
    "reference": "What do pronouns and references mean, including outside nearby turns?",
    "condition": "Which conditions and exceptions must accompany the quoted material?",
    "question_relevance": "Does this information help answer the actual question?",
    "new_information": "Is its meaning already present elsewhere in the legacy packet?",
    "answer_sufficiency": "What other evidence is needed, and what remains unknown?",
}


def serialize_dossier(value):
    """Canonical JSON preserving text on decode and physical JSONL boundaries."""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text.replace("\x85", "\\u0085").replace("\u2028", "\\u2028").replace(
        "\u2029", "\\u2029"
    )


def build_review_dossier(
    snapshot, hits, units, *, question, baseline, scope, as_of, unit_char_budget=4000
):
    """Export complete review material; every semantic obligation is UNREVIEWED.

    SourceUnit proposals come from the caller, not a hidden selector. Compiler
    refusals remain refusals. The full source pool is deliberately not squeezed
    into an answer budget; unit_char_budget applies only to compiler previews.
    Benchmark labels/grades must be kept outside these typed inputs. Label-free
    input does not make an exposed development dataset an unseen evaluation.
    """
    eligible = eligible_sources(snapshot, scope=scope, as_of=as_of)
    if (
        type(question) is not str
        or not question.strip()
        or type(baseline) is not str
        or type(hits) is not tuple
        or type(units) is not tuple
        or type(unit_char_budget) is not int
        or unit_char_budget < 0
    ):
        raise ValueError("Invalid review inputs")
    by_id = {t.id: t for t in eligible}
    ranked, seen = [], set()
    for hit in hits:
        if (
            type(hit) is not RetrievalHit
            or type(hit.source_id) is not str
            or hit.source_id not in by_id
            or hit.source_id in seen
            or hit.source_sha256 != digest(by_id[hit.source_id].text)
            or isinstance(hit.score, bool)
            or not isinstance(hit.score, Real)
            or not math.isfinite(hit.score)
            or hit.score < 0
            or type(hit.tie_order) is not int
            or hit.tie_order < 0
        ):
            raise ValueError("Invalid, duplicate or ineligible review hit")
        seen.add(hit.source_id)
        ranked.append(
            dict(
                source_id=hit.source_id,
                source_sha256=hit.source_sha256,
                score=float(hit.score),
                tie_order=hit.tie_order,
                whole_body_literal_in_baseline=by_id[hit.source_id].text in baseline,
            )
        )

    proposals, seen_units = [], set()
    for unit in units:
        if (
            type(unit) is not SourceUnit
            or type(unit.id) is not str
            or unit.id in seen_units
        ):
            raise ValueError("Invalid or duplicate review unit")
        seen_units.add(unit.id)
        preview, report = compile_unit(
            snapshot, unit, question=question, scope=scope, as_of=as_of,
            char_budget=unit_char_budget,
        )
        proposals.append(
            dict(
                unit_id=unit.id,
                spans=[asdict(span) for span in unit.spans],
                preview=preview,
                report=report,
                review={key: "UNREVIEWED" for key in REVIEW_QUESTIONS},
            )
        )

    assert DEPENDENCY_KINDS <= REVIEW_QUESTIONS.keys()
    body = dict(
        schema="source-unit-review-v1",
        question=question,
        question_sha256=digest(question),
        scope=scope,
        as_of=as_of.isoformat(),
        snapshot_sha256=snapshot_digest(snapshot),
        baseline=dict(
            text=baseline, sha256=digest(baseline),
            authority="LEGACY_MIXED_EVIDENCE_NOT_TRUTH", source_identity="NOT_CERTIFIED",
        ),
        sources=[
            dict(
                id=t.id, session=t.session, position=t.position, role=t.role,
                observed_at=t.observed_at.isoformat(), text=t.text, sha256=digest(t.text),
            )
            for t in eligible
        ],
        future_source_ids=[t.id for t in snapshot.turns if t.id not in by_id],
        ranked_hits=ranked,
        proposals=proposals,
        review_questions=dict(REVIEW_QUESTIONS),
        pool_scope="ALL_ELIGIBLE_ORIGINAL_TURNS_IN_SUPPLIED_SNAPSHOT",
        supplied_scope_is_authorization=False,
        baseline_temporal_cleanliness="NOT_CERTIFIED",
        original_baseline_source_identity="NOT_CERTIFIED",
        semantic_completeness="NOT_CERTIFIED",
        answer_path_eligible=False,
        paid_run_ready=False,
        answer_accuracy="NOT_MEASURED",
        limits=[
            "Supplied rankings and proposals are candidates, not semantic evidence judgments.",
            "Literal body presence is not novelty, identity, entailment or a corrected answer.",
            "Full eligible pool is for offline review; its size is not an answer-budget claim.",
            "Reviewing an exposed dataset is not independent held-out validation.",
        ],
    )
    return dict(body, dossier_sha256=digest(serialize_dossier(body)))
