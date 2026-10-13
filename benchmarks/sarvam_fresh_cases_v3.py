"""Ask for the stated preference instead of implying an executed/requested change.

V2 remains preserved. Only the twelve separate-booking questions and their hash
bindings change; sources, models, expected answers and all state stay fixed.
"""

import argparse
from pathlib import Path

from benchmarks.sarvam_fresh_cases_v2 import build as build_v2
from benchmarks.sarvam_native_study.transport import digest, save

QUERIES = {
    "en": (
        "What is my current preferred date for {booking}? " "Read only; do not change the booking."
    ),
    "hi": "{booking} के लिए अभी मेरी पसंद की तारीख कौन सी है? केवल बताएं; बुकिंग न बदलें।",
    "te": "{booking} కోసం ప్రస్తుతం నాకు ఇష్టమైన తేదీ ఏది? చెప్పండి మాత్రమే; బుకింగ్ మార్చవద్దు.",
}


def build():
    runtime, gold = build_v2()
    for case in runtime:
        if case["family"] == "separate_booking":
            case["question"] = QUERIES[case["language"]].format(booking=case["appointment_id"])
            gold[case["id"]]["runtime_sha256"] = digest(case)
    return runtime, gold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline v3 bank generation; no API calls.")
    parser.add_argument("new_directory", type=Path)
    args = parser.parse_args()
    args.new_directory.mkdir()
    runtime, gold = build()
    save(args.new_directory / "runtime.json", runtime)
    save(args.new_directory / "gold.json", gold)
