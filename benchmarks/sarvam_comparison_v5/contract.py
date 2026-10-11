"""Move the fact contract from provider grammar to prompt; keep local validation."""

from benchmarks.sarvam_comparison_v4.contract import (  # noqa: F401
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
from benchmarks.sarvam_comparison_v4.contract import request as parent_request
from benchmarks.sarvam_discovery.contract import REQUEST_BYTE_LIMIT


def request(stage, case, facts=None, sources=None):
    req = parent_request(stage, case, facts, sources)
    if stage == "extract":
        schema = req["response_format"]["json_schema"]["schema"]
        req["response_format"] = {"type": "json_object"}
        req["messages"][0]["content"] += (
            "\nOutput must satisfy this JSON Schema, which the application validates locally:\n"
            + canonical(schema)
        )
    if len(canonical(req).encode()) > REQUEST_BYTE_LIMIT:
        raise ValueError("Request exceeds frozen byte ceiling; no truncation")
    return req
