"""Minimal schema compatibility change; all admission and answer semantics retained."""

from benchmarks.sarvam_comparison_v3.contract import (  # noqa: F401
    CASE_IDS,
    FIXTURES,
    MODEL,
    RATES,
    ROOT,
    SETTINGS,
    admit_facts,
    canonical,
    eligible,
    file_sha,
    loads,
    parse_answer,
    runtime_cases,
    sha,
    source_packet,
)
from benchmarks.sarvam_comparison_v3.contract import request as parent_request


def request(stage, case, facts=None, sources=None):
    req = parent_request(stage, case, facts, sources)
    if stage == "extract":
        # Live v3 HTTP500 disclosed upstream400 unsupported uniqueItems. Remove only
        # that provider grammar keyword. Shared local parse_facts still rejects
        # duplicate citations before any augmentation enters the reopened store.
        ids = req["response_format"]["json_schema"]["schema"]["properties"]["facts"]["items"][
            "properties"
        ]["source_ids"]
        del ids["uniqueItems"]
    return req
