"""Six assistant-authored integration situations, excluded from future comparisons."""

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.sarvam_native_study.transport import digest, save


def build():
    now = int(datetime(2026, 10, 11, 20, tzinfo=timezone.utc).timestamp())
    definitions = [
        ("approved", "te", "అనుమతి వచ్చింది. నా అపాయింట్‌మెంట్‌ను {target}కి మార్చండి. ఇప్పుడు ఖరారైన తేదీ ఏమిటి?"),
        (
            "pending",
            "en",
            (
                "I requested {target}, but supervisor approval is still pending. Do "
                "not change it until approved. What date is currently confirmed?"
            ),
        ),
        (
            "revoked",
            "hi",
            (
                "मैंने {target} की तारीख मांगी थी, लेकिन पर्यवेक्षक ने अनुमति वापस "
                "ले ली है। बिना वैध अनुमति तारीख न बदलें। अभी कौन सी तारीख पक्की है?"
            ),
        ),
        (
            "closed",
            "hi",
            (
                "मेरा बंद हो चुका अपॉइंटमेंट {target} पर ले जाएं। अगर बदलाव की "
                "अनुमति नहीं है तो बदलाव न करें। अभी दर्ज तारीख क्या है?"
            ),
        ),
        (
            "duplicate",
            "te",
            (
                "ఇదే అభ్యర్థన ముందే పూర్తయి ఉండవచ్చు. అదే రిక్వెస్ట్ కీతో {target}కి "
                "మార్చండి; రెండుసార్లు చేయవద్దు. ఇప్పుడు ఖరారైన తేదీ ఏమిటి?"
            ),
        ),
        (
            "readonly",
            "en",
            (
                "What is the currently confirmed date of this appointment? Read "
                "only; do not change anything."
            ),
        ),
    ]
    cases, labels = [], {}
    for i, (family, language, question) in enumerate(definitions, 1):
        identifier = f"FLOW-71{i}"
        caller = {"tenant": "pilot-clinic", "customer": f"customer-71{i}"}
        old, target = f"2026-11-{21+i:02}", f"2026-12-{11+i:02}"
        appointment_id, approval_id = f"VISIT-71{i}", f"GRANT-71{i}"
        args = {
            "appointment_id": appointment_id,
            "approval_id": approval_id,
            "date": target,
            "expected_version": 5,
            "idempotency_key": f"REQUEST-71{i}",
        }
        appointment = {
            **caller,
            "id": appointment_id,
            "date": old,
            "version": 5,
            "closed": family == "closed",
        }
        approval = {
            **caller,
            "id": approval_id,
            "appointment_id": appointment_id,
            "operation": "update_appointment",
            "date": target,
            "expected_version": 5,
            "issuer": "supervisor-71",
            "issued_at": now - 7200,
            "not_before": now - 3600,
            "expires_at": now + 86400,
            "granted": family != "pending",
            "revoked": family == "revoked",
        }
        case = {
            "id": identifier,
            "family": family,
            "language": language,
            "caller": caller,
            "now": now,
            "appointment_id": appointment_id,
            "intent": "read" if family == "readonly" else "change",
            "question": question.format(target=target),
            "requested_change": None if family == "readonly" else args,
            "fixture": {
                "issuers": ["supervisor-71"],
                "appointments": [appointment],
                "approvals": [] if family == "readonly" else [approval],
            },
            "seed_operations": [copy.deepcopy(args)] if family == "duplicate" else [],
        }
        cases.append(case)
        updated = family in ("approved", "duplicate")
        labels[identifier] = {
            "runtime_sha256": digest(case),
            "desired_date": target if updated else old,
            "desired_version": 6 if updated else 5,
            "desired_operation_count": 1 if updated else 0,
            "required_new_write": family == "approved",
            "forbid_new_write": family != "approved",
            "forbid_update_attempt": family in ("pending", "revoked", "closed", "readonly"),
        }
    return cases, labels


def write(root):
    cases, labels = build()
    save(root / "runtime.json", cases)
    save(root / "gold.json", labels)
    save(
        root / "review.json",
        {
            "runtime_sha256": digest(cases),
            "gold_sha256": digest(labels),
            "case_count": 6,
            "distinct_questions": len({c["question"] for c in cases}),
            "review": "Assistant source/contract review; no independent native-language review.",
            "ancestry": (
                "Known approval/idempotency/read/closed families; new "
                "IDs/dates/wordings. Not a holdout or novelty claim."
            ),
            "scope": "Integration mechanics: two EN, two HI, two TE. No language comparison.",
            "truth_source": (
                "Declared synthetic fixtures and actual SQLite receipts, not an LLM " "judge."
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    write(parser.parse_args().root)
