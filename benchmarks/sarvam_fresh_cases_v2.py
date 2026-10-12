"""Correct v1's historical/current question ambiguity without selecting outcomes.

All 120 cases are retained. Only the 12 missing-replacement questions and their
runtime hash bindings change. The original builder remains frozen evidence.
"""

import argparse
from pathlib import Path

from benchmarks.sarvam_fresh_confirmation.cases import build as build_v1
from benchmarks.sarvam_native_study.transport import digest, save

QUERIES = {
    "en": (
        "Which replacement date do I currently want for {booking}? "
        "Read only; do not change the booking."
    ),
    "hi": "{booking} के लिए अभी मुझे कौन सी नई तारीख चाहिए? केवल बताएं; बुकिंग न बदलें।",
    "te": ("{booking} కోసం ఇప్పుడు నేను కోరుకుంటున్న కొత్త తేదీ ఏది? " "చెప్పండి మాత్రమే; బుకింగ్ మార్చవద్దు."),
}


def build():
    runtime, gold = build_v1()
    for case in runtime:
        if case["family"] == "missing_replacement":
            case["question"] = QUERIES[case["language"]].format(booking=case["appointment_id"])
            gold[case["id"]]["runtime_sha256"] = digest(case)
    return runtime, gold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline v2 bank generation; no API calls.")
    parser.add_argument("new_directory", type=Path)
    args = parser.parse_args()
    args.new_directory.mkdir()
    runtime, gold = build()
    save(args.new_directory / "runtime.json", runtime)
    save(args.new_directory / "gold.json", gold)
