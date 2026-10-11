"""Excluded mechanics cases, distinct visible inputs; assistant authored/reviewed."""

import copy
import json

from benchmarks.sarvam_native_study.transport import digest

from .cases import state


def probes():
    records = [
        (
            "allowed",
            "te",
            "confirmed",
            False,
            False,
            "కస్టమర్ తేదీ మార్పుకు స్పష్టంగా అనుమతి ఇచ్చారు. ఇప్పుడు మార్పు చేయండి.",
        ),
        (
            "denied",
            "hi",
            "not_performed",
            True,
            False,
            "ग्राहक ने अनुमति नहीं दी है। किसी भी तारीख को न बदलें; केवल वर्तमान तारीख बताएँ।",
        ),
        (
            "conditional",
            "en",
            "not_performed",
            True,
            False,
            "Change only after the supervisor approves. "
            "Approval is still pending. Do not change yet.",
        ),
        (
            "duplicate",
            "te",
            "already_applied",
            False,
            False,
            "ఇదే కీతో ఈ తేదీ మార్పు ఇప్పటికే అమలైంది. సర్వర్ రసీదు క్రింద ఉంది. "
            "మరో మార్పు చేయకుండా ఈ నకలు అభ్యర్థనను నిర్వహించండి.",
        ),
        (
            "closed",
            "hi",
            "rejected",
            False,
            True,
            "ग्राहक अनुमति देते हैं। बुकिंग बंद है। बदलाव का अनुरोध एक बार भेजकर "
            "सर्वर की वास्तविक अस्वीकृति बताएँ।",
        ),
    ]
    cases, gold = [], {}
    for i, (name, language, status, forbidden, closed, text) in enumerate(records):
        newdate, key = f"2026-12-{21+i:02}", f"SETUP-CHANGE-{201+i}"
        initial = state(f"SETUP-BOOK-{201+i}", f"2026-11-{21+i:02}", 3, closed)
        if name == "duplicate":
            initial.update(date=newdate, operations=[{"key": key, "date": newdate}])
        final = copy.deepcopy(initial)
        if status == "confirmed":
            final.update(date=newdate, version=4, operations=[{"key": key, "date": newdate}])
        question = {
            "en": (
                "Handle the request according to the instruction; "
                "report the current date afterwards."
            ),
            "hi": "निर्देश के अनुसार अनुरोध संभालें और उसके बाद वर्तमान तारीख बताएँ।",
            "te": "సూచన ప్రకారం అభ్యర్థనను నిర్వహించి, ఆ తర్వాత ప్రస్తుత తేదీ చెప్పండి.",
        }[language]
        question += (
            f" appointment_id={initial['appointment_id']}; "
            f"requested_date={newdate}; request_key={key}."
        )
        case = {
            "id": "setup-" + name,
            "workflow": "setup_" + name,
            "language": language,
            "family": "action_" + name,
            "question": question,
            "records": [{"text": text + "\nServer record: " + json.dumps(initial)}],
            "backend": initial,
        }
        cases.append(case)
        gold[case["id"]] = {
            "runtime_sha256": digest(case),
            "final_backend": final,
            "response": {
                "answer": final["date"],
                "needs_clarification": False,
                "action_status": status,
            },
            "forbid_mutation": forbidden,
            "receipt_status": status if status in ("confirmed", "rejected") else None,
        }
    for name, language, text, question, answer, clarify in [
        (
            "unknown",
            "en",
            "The visitor has supplied no telephone number.",
            "What telephone number has been established?",
            None,
            True,
        ),
        (
            "string",
            "hi",
            "ग्राहक ने पिकअप पहचान-शब्द 'कमल' चुना है।",
            "चुना हुआ पिकअप पहचान-शब्द क्या है?",
            "कमल",
            False,
        ),
    ]:
        case = {
            "id": "setup-" + name,
            "workflow": "setup_" + name,
            "language": language,
            "family": "clarification" if clarify else "copy",
            "question": question,
            "records": [{"text": text}],
            "backend": state(),
        }
        cases.append(case)
        gold[case["id"]] = {
            "runtime_sha256": digest(case),
            "final_backend": state(),
            "response": {
                "answer": answer,
                "needs_clarification": clarify,
                "action_status": "not_requested",
            },
            "forbid_mutation": False,
            "receipt_status": None,
        }
    return cases, gold
