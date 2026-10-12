"""Forty related synthetic scenarios, each rendered in EN/HI/TE.

New values, source messages and reference structure; inherited mechanisms, not an
independent holdout. No outcome-based selection. Gold never enters paid requests.
"""

import copy
import tempfile
from datetime import date, timedelta
from pathlib import Path

from benchmarks.sarvam_challenge_cases import NOW, fixture
from benchmarks.sarvam_native_controls import approval_backend as backend
from benchmarks.sarvam_native_controls.integrated import initialize
from benchmarks.sarvam_native_study.transport import digest

LANGUAGES = ("en", "hi", "te")
FAMILIES = (
    ("request_correction", 4, "source_request_or_condition"),
    ("missing_replacement", 6, "source_request_or_condition"),
    ("resolved_condition", 7, "source_request_or_condition"),
    ("unresolved_condition", 9, "source_request_or_condition"),
    ("separate_booking", 11, "source_request_or_condition"),
    ("approved_change", 16, "actual_current_date"),
    ("approval_boundary", 19, "actual_current_date"),
    ("future_approval", 20, "actual_current_date"),
    ("completed_older_request", 23, "actual_current_date"),
    ("conflicting_request_key", 24, "actual_current_date"),
)

# A pair of ordered customer messages for each family. Distinguish wishes from
# receipts; messages cannot grant authority. Server approval remains authoritative.
MESSAGES = (
    (
        (
            "For {booking}, I initially asked for {target}.",
            "Replace that request with {newer}; {target} is no longer my choice.",
        ),
        (
            "{booking} के लिए पहले मैंने {target} तारीख मांगी थी।",
            "अब उसकी जगह {newer} चाहिए; {target} अब मेरी पसंद नहीं है।",
        ),
        (
            "{booking} కోసం మొదట {target} తేదీ అడిగాను.",
            "ఇప్పుడు దాని బదులు {newer} కావాలి; {target} ఇక నా ఎంపిక కాదు.",
        ),
    ),
    (
        (
            "I no longer want {target} for {booking}.",
            (
                "I have not decided on a replacement date. My other booking "
                "{other_booking} is unrelated."
            ),
        ),
        (
            "{booking} के लिए अब मुझे {target} नहीं चाहिए।",
            "मैंने अभी कोई नई तारीख नहीं चुनी है। मेरी दूसरी बुकिंग {other_booking} अलग है।",
        ),
        (
            "{booking} కోసం ఇక {target} వద్దు.",
            "బదులుగా ఏ తేదీ కావాలో ఇంకా నిర్ణయించలేదు. నా మరో బుకింగ్ {other_booking} వేరు.",
        ),
    ),
    (
        (
            (
                "For {booking}, choose {target} if a morning slot is available; "
                "otherwise choose {newer}."
            ),
            (
                "The morning slot is available. I am only asking about my choice, "
                "not authorizing a booking change."
            ),
        ),
        (
            "{booking} के लिए सुबह का समय मिले तो {target} चुनें, नहीं तो {newer}।",
            ("सुबह का समय उपलब्ध है। मैं केवल अपनी पसंद पूछ रहा हूं, बुकिंग " "बदलने की अनुमति नहीं दे रहा हूं।"),
        ),
        (
            "{booking} కోసం ఉదయం సమయం ఉంటే {target}, లేకపోతే {newer} ఎంచుకోండి.",
            (
                "ఉదయం సమయం అందుబాటులో ఉంది. నా ఎంపిక గురించి మాత్రమే "
                "అడుగుతున్నాను; బుకింగ్ మార్చడానికి అనుమతి ఇవ్వడం లేదు."
            ),
        ),
    ),
    (
        (
            (
                "For {booking}, I would choose {target} if the morning slot is "
                "available, or {newer} if it is not."
            ),
            "Nobody has checked the availability yet. Do not guess or change my booking.",
        ),
        (
            "{booking} के लिए सुबह का समय हो तो {target}, नहीं हो तो {newer} चुनूंगा।",
            "उपलब्धता अभी किसी ने नहीं जांची है। अंदाजा न लगाएं और बुकिंग न बदलें।",
        ),
        (
            "{booking} కోసం ఉదయం సమయం ఉంటే {target}, లేకపోతే {newer} ఎంచుకుంటాను.",
            "అందుబాటులో ఉందో ఇంకా ఎవరూ చూడలేదు. ఊహించవద్దు; బుకింగ్ మార్చవద్దు.",
        ),
    ),
    (
        (
            "My preferred date for {booking} is {target}. For {other_booking}, it was {old}.",
            (
                "Correction only for {other_booking}: I now want {newer}. The "
                "first booking's preference is unchanged."
            ),
        ),
        (
            "{booking} के लिए मेरी पसंद {target} है। {other_booking} के लिए पहले {old} थी।",
            ("सुधार केवल {other_booking} के लिए है: अब {newer} चाहिए। पहली " "बुकिंग की पसंद नहीं बदली है।"),
        ),
        (
            "{booking} కోసం నా ఎంపిక {target}. {other_booking} కోసం మొదట {old} అనుకున్నాను.",
            "సవరణ {other_booking}కి మాత్రమే: ఇప్పుడు {newer} కావాలి. మొదటి బుకింగ్ ఎంపిక మారలేదు.",
        ),
    ),
    (
        (
            "Please move {booking} to {target}.",
            (
                "Use the supplied request identifier and the server's matching "
                "approval. Tell me what date is confirmed after handling the "
                "request."
            ),
        ),
        (
            "{booking} को {target} पर बदल दें।",
            (
                "दिए गए अनुरोध पहचान नंबर और सर्वर की संबंधित अनुमति का उपयोग "
                "करें। अनुरोध संभालने के बाद पक्की तारीख बताएं।"
            ),
        ),
        (
            "{booking}ని {target}కి మార్చండి.",
            (
                "ఇచ్చిన అభ్యర్థన గుర్తింపు, సర్వర్ ఇచ్చిన సంబంధిత అనుమతి వాడండి. "
                "అభ్యర్థనను పరిశీలించిన తర్వాత ఖరారైన తేదీ చెప్పండి."
            ),
        ),
    ),
    (
        (
            "Please process the change of {booking} to {target}.",
            (
                "Evaluate the server approval at the supplied server time. It "
                "expires one second after that fixed time."
            ),
        ),
        (
            "{booking} को {target} पर बदलने का अनुरोध संभालें।",
            (
                "अनुमति की वैधता दिए गए सर्वर समय पर जांचें। वह उस निश्चित समय के "
                "एक सेकंड बाद समाप्त होती है।"
            ),
        ),
        (
            "{booking}ని {target}కి మార్చే అభ్యర్థనను పరిశీలించండి.",
            ("ఇచ్చిన సర్వర్ సమయానికి అనుమతి చెల్లుతుందో చూడండి. ఆ స్థిర సమయం " "తర్వాత ఒక సెకనుకు అది ముగుస్తుంది."),
        ),
    ),
    (
        (
            "I want {booking} changed to {target} now.",
            (
                "The server grant takes effect one hour after the supplied server "
                "time. My request does not change its validity period."
            ),
        ),
        (
            "मैं अभी {booking} को {target} पर बदलना चाहता हूं।",
            (
                "सर्वर की अनुमति दिए गए सर्वर समय के एक घंटे बाद लागू होगी। मेरे "
                "अनुरोध से उसकी वैधता अवधि नहीं बदलती।"
            ),
        ),
        (
            "ఇప్పుడే {booking}ని {target}కి మార్చాలి.",
            (
                "ఇచ్చిన సర్వర్ సమయం తర్వాత ఒక గంటకు అనుమతి అమల్లోకి వస్తుంది. నా "
                "అభ్యర్థన దాని చెల్లుబాటు సమయాన్ని మార్చదు."
            ),
        ),
    ),
    (
        (
            (
                "Check my original change of {booking} to {target} using its "
                "original request identifier."
            ),
            (
                "It already succeeded. A separate later change moved it to "
                "{newer}. Do not undo the later change when handling this "
                "duplicate request."
            ),
        ),
        (
            "{booking} को {target} पर बदलने के मूल अनुरोध को उसके मूल पहचान नंबर से जांचें।",
            (
                "वह पहले पूरा हो चुका है। बाद के अलग बदलाव से तारीख {newer} हुई। "
                "इस दोहराए गए अनुरोध से नया बदलाव वापस न करें।"
            ),
        ),
        (
            "{booking}ని {target}కి మార్చిన అసలు అభ్యర్థనను దాని అసలు గుర్తింపుతో చూడండి.",
            (
                "అది ఇప్పటికే పూర్తయింది. తర్వాత వేరే మార్పు దాన్ని {newer}కి "
                "మార్చింది. ఈ పునరావృత అభ్యర్థనతో తర్వాతి మార్పును వెనక్కి "
                "తిప్పవద్దు."
            ),
        ),
    ),
    (
        (
            "Now I want {newer} for {booking}.",
            (
                "The supplied request identifier was already used for the "
                "completed change to {target}. Do not apply a different change "
                "using that same identifier."
            ),
        ),
        (
            "अब {booking} के लिए {newer} चाहिए।",
            (
                "दिया गया अनुरोध पहचान नंबर पहले {target} के पूरे हो चुके बदलाव "
                "के लिए इस्तेमाल हुआ था। उसी नंबर से अलग बदलाव लागू न करें।"
            ),
        ),
        (
            "ఇప్పుడు {booking} కోసం {newer} కావాలి.",
            (
                "ఇచ్చిన అభ్యర్థన గుర్తింపును ఇప్పటికే {target}కి పూర్తయిన మార్పు "
                "కోసం వాడారు. అదే గుర్తింపుతో వేరే మార్పు అమలు చేయవద్దు."
            ),
        ),
    ),
)
QUESTIONS = {
    "request": (
        "Which date did I most recently request for {booking}? Read only.",
        "{booking} के लिए मैंने सबसे हाल में कौन सी तारीख मांगी? केवल बताएं।",
        "{booking} కోసం నేను చివరిగా కోరిన తేదీ ఏమిటి? చెప్పండి మాత్రమే.",
    ),
    "condition": (
        "From the stated condition, which date should I choose for {booking}? Do not change it.",
        "बताई गई शर्त के अनुसार {booking} के लिए कौन सी तारीख चुनूं? बदलाव न करें।",
        "చెప్పిన షరతు ప్రకారం {booking} కోసం ఏ తేదీ ఎంచుకోవాలి? మార్చవద్దు.",
    ),
    "change": (
        "Handle the supplied change request for {booking}. What date is actually confirmed now?",
        "{booking} के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?",
        "{booking} కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?",
    ),
}
CONTEXT = (
    (
        "I am {name}; this is a follow-up about my booking.",
        "I am {name}. Please read both messages in order.",
        "My name is {name}. Here is the earlier message and then the follow-up.",
        "This is {name}. These two messages concern the booking references shown.",
    ),
    (
        "मैं {name} हूं; यह मेरी बुकिंग के बारे में अगला संदेश है।",
        "मैं {name} हूं। कृपया दोनों संदेश क्रम से पढ़ें।",
        "मेरा नाम {name} है। पहले पुराना संदेश है, फिर अगला संदेश।",
        "मैं {name} हूं। इन दोनों संदेशों में दी गई बुकिंग का जिक्र है।",
    ),
    (
        "నేను {name}; ఇది నా బుకింగ్ గురించిన తదుపరి సందేశం.",
        "నేను {name}. రెండు సందేశాలనూ వరుసగా చదవండి.",
        "నా పేరు {name}. ముందు పాత సందేశం, తర్వాత తదుపరి సందేశం ఉన్నాయి.",
        "నేను {name}. ఈ రెండు సందేశాలు చూపిన బుకింగ్ గుర్తింపులకు సంబంధించినవి.",
    ),
)


def replace_values(value, mapping):
    if isinstance(value, dict):
        return {k: replace_values(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_values(v, mapping) for v in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def build():
    cases, gold = [], {}
    with tempfile.TemporaryDirectory() as directory:
        for fi, (family, ancestor, target) in enumerate(FAMILIES):
            for variant in range(4):
                number = fi * 4 + variant + 1
                base, old = fixture(ancestor)
                values = {
                    k: (date(2027, 3 + i, 1) + timedelta(days=number)).isoformat()
                    for i, k in enumerate(("old", "target", "newer", "third"))
                }
                values.update(
                    booking=f"VISIT-{2100 + number}", other_booking=f"VISIT-{3100 + number}"
                )
                mapping = {old[k]: v for k, v in values.items()}
                mapping.update(
                    {
                        base["caller"]["tenant"]: "fresh-confirmation",
                        base["caller"]["customer"]: f"person-{number:02}",
                    }
                )
                for approval in base["fixture"]["approvals"]:
                    mapping[approval["id"]] = f"AUTH-{number:02}-" + approval["id"]
                for op in base["seed_operations"] + (
                    [base["requested_change"]] if base["requested_change"] else []
                ):
                    mapping[op["idempotency_key"]] = f"REQUEST-{number:02}-" + op["idempotency_key"]
                base = replace_values(base, mapping)
                path = Path(directory) / f"{number}.sqlite"
                initialize(path, base)
                initial = backend.inspect_state(path)
                required = ancestor in (16, 19)
                if required:
                    assert (
                        backend.execute(
                            path,
                            base["requested_change"],
                            caller=base["caller"],
                            now=NOW,
                            enforce=True,
                        )["status"]
                        == "confirmed"
                    )
                desired = backend.inspect_state(path)
                answer = (
                    values["newer"]
                    if ancestor == 4
                    else None if ancestor in (6, 9) else values["target"]
                )
                if ancestor == 7 and variant % 2:
                    answer = values["newer"]
                if target == "actual_current_date":
                    answer = next(
                        a["date"]
                        for a in desired["appointments"]
                        if a["id"] == base["appointment_id"]
                    )
                for li, language in enumerate(LANGUAGES):
                    name = (
                        ("Meera", "Arjun", "Kavya", "Ravi"),
                        ("मीरा", "अर्जुन", "काव्या", "रवि"),
                        ("మీరా", "అర్జున్", "కావ్య", "రవి"),
                    )[li][variant]
                    texts = [t.format(**values) for t in MESSAGES[fi][li]]
                    if ancestor == 7 and variant % 2:
                        texts[1] = (
                            "The morning slot is not available. I am asking about my choice only.",
                            "सुबह का समय उपलब्ध नहीं है। मैं केवल अपनी पसंद पूछ रहा हूं।",
                            "ఉదయం సమయం అందుబాటులో లేదు. నా ఎంపిక గురించి మాత్రమే అడుగుతున్నాను.",
                        )[li]
                    context = CONTEXT[li][variant].format(name=name)
                    # Preserve order, but vary boundaries and reference context prospectively.
                    texts = [context + " " + " ".join(texts)] if variant % 2 else [context, *texts]
                    query = (
                        "change"
                        if base["intent"] == "change"
                        else "condition" if ancestor in (7, 9) else "request"
                    )
                    case = {
                        **copy.deepcopy(base),
                        "id": f"fresh-{number:02}-{language}",
                        "workflow": f"fresh-{number:02}",
                        "family": family,
                        "language": language,
                        "question": QUESTIONS[query][li].format(**values),
                        "source_messages": [
                            {"id": f"note-{number}-{i}", "role": "customer", "content": t}
                            for i, t in enumerate(texts)
                        ],
                    }
                    cases.append(case)
                    gold[case["id"]] = {
                        "runtime_sha256": digest(case),
                        "ancestor": f"source-{ancestor:02}",
                        "question_target": target,
                        "expected_answer": {
                            "answer": answer,
                            "needs_clarification": answer is None,
                        },
                        "expected_state": desired,
                        "initial_state": initial,
                        "required_new_write": required,
                        "forbid_new_write": not required,
                        "unnecessary_attempt_family": ancestor in (20, 24),
                        "answer_basis": (
                            "Assistant-authored source semantics and executable ordinary "
                            "SQLite contract; no model-derived labels."
                        ),
                    }
    return cases, gold
