"""Require explicit cited-text witnesses for dates; do not claim entailment."""

import re

from benchmarks.sarvam_comparison_v5.contract import (  # noqa: F401
    CASE_IDS,
    FIXTURES,
    MODEL,
    RATES,
    ROOT,
    SETTINGS,
    canonical,
    eligible,
    file_sha,
    loads,
    parse_answer,
    request,
    runtime_cases,
    sha,
    source_packet,
)
from benchmarks.sarvam_comparison_v5.contract import admit_facts as structural_admission


def admit_facts(text, case):
    """Reject the complete augmentation when a validity date has no cited witness.

    Only original record text counts: observation metadata, query dates, generated
    facts, unauthorized/future records and uncited records cannot supply a witness.
    Literal ISO-date occurrence is necessary here, never sufficient for entailment.
    Spelled/relative dates and derived endpoints are deliberately not normalized.
    """
    result = structural_admission(text, case)
    if result["status"] == "rejected":
        return result
    originals = {row["id"]: row["text"] for row in eligible(case)}
    missing = []
    for index, fact in enumerate(result["facts"]):
        for field in ("valid_from", "valid_until"):
            value = fact[field]
            if value is None:
                continue
            pattern = re.compile(r"(?<![\w-])" + re.escape(value) + r"(?![\w-])")
            if not any(pattern.search(originals[s]) for s in fact["source_ids"]):
                missing.append({"fact_index": index, "field": field, "value": value})
    if missing:
        return {
            "status": "rejected",
            "facts": [],
            "reason": "VALIDITY_DATE_WITHOUT_CITED_LITERAL_WITNESS",
            "missing_date_witnesses": missing,
            "semantic_review": "UNREVIEWED",
        }
    return {**result, "date_witness_check": "LITERAL_ONLY_NOT_ENTAILMENT"}
