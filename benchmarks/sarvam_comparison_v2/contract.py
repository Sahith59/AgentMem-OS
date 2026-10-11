"""Same prompts and parsers; fixed 12-case slice and explicit null reasoning."""

from benchmarks.sarvam_discovery import contract as original
from benchmarks.sarvam_discovery.contract import (  # noqa: F401
    ENDPOINT,
    FIXTURES,
    INPUT_TOKEN_RESERVATION,
    MODEL,
    RATES,
    ROOT,
    canonical,
    eligible,
    file_sha,
    loads,
    parse_answer,
    parse_facts,
    sha,
)

CASE_IDS = tuple(
    f"{template}-{language}"
    for template in ("D01", "D06", "D07", "D09")
    for language in ("en", "hi", "te")
)
SETTINGS = {**original.SETTINGS, "reasoning_effort": None}


def runtime_cases():
    indexed = {case["id"]: case for case in original.runtime_cases()}
    return [indexed[case_id] for case_id in CASE_IDS]


def request(stage, case, facts=None, sources=None):
    payload = original.request(stage, case, facts, sources)
    payload["reasoning_effort"] = None
    return payload
