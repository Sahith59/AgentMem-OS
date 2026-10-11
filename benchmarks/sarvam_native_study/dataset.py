"""Synthetic diagnostic cases. Labels are generated offline, never by the runner.

200 parameterized situations, 100 grouping IDs, 10 capability families.
Grouping IDs do not imply 100 independently authored linguistic templates.
Translations and mixed renderings share the parent split. No independent native
validation is implied. This is a diagnostic bank, not representative user traffic.
"""

# Long multilingual source sentences stay intact for language review.
# ruff: noqa: E501

import argparse
import random
from pathlib import Path

from .transport import digest, save

LANGS = ("en", "hi", "te")
FAMILIES = (
    "correction",
    "effective_date",
    "conditional",
    "backend_outcome",
    "identity",
    "arithmetic",
    "distractors",
    "ambiguity",
    "action_update",
    "action_retry",
)


def tri(en, hi, te):
    return dict(zip(LANGS, (en, hi, te), strict=True))


def scenario(family, i):
    """Construct parallel source meaning and separately retained evaluator values."""
    records = []
    order = f"O-{1000+i}"
    a, b, c = f"K{31+i}", f"B{61+i}", f"Z{91+i}"
    day = 10 + i
    current, desired = f"2026-11-{day:02}", f"2026-12-{i+1:02}"
    yes = i % 2 == 0
    clarification = False
    backend = {
        "appointment_id": order,
        "date": current,
        "version": 4,
        "closed": False,
        "operations": [],
    }
    action_status, writes = "not_requested", 0

    def add(text, role="user"):
        records.append({"id": f"s{len(records)+1:03}", "role": role, "text": text})

    if family == "correction":
        add(
            tri(
                f"For order {order}, my pickup point is {a}.",
                f"ऑर्डर {order} के लिए मेरा पिकअप स्थान {a} है।",
                f"ఆర్డర్ {order} కోసం నా పికప్ స్థలం {a}.",
            )
        )
        add(
            tri(
                f"Correction for {order}: use {b} instead of {a}.",
                f"{order} के लिए सुधार: {a} की जगह {b} इस्तेमाल करें।",
                f"{order} కోసం సవరణ: {a} బదులుగా {b} ఉపయోగించండి.",
            )
        )
        # Alternating explicit reversal and a historical mention distinguish recency from meaning.
        if yes:
            add(
                tri(
                    f"Cancel that correction for {order}; keep {a}.",
                    f"{order} के लिए वह सुधार रद्द करें; {a} ही रखें।",
                    f"{order} కోసం ఆ సవరణను రద్దు చేయండి; {a} నే ఉంచండి.",
                )
            )
            answer = a
        else:
            add(
                tri(
                    f"The old label still says {a}; it has not been replaced yet.",
                    f"पुराने लेबल पर अब भी {a} लिखा है; उसे अभी बदला नहीं गया है।",
                    f"పాత లేబుల్‌పై ఇంకా {a} ఉంది; దాన్ని ఇంకా మార్చలేదు.",
                )
            )
            answer = b
        question = tri(
            f"Which pickup point should now be used for {order}?",
            f"अब {order} के लिए कौन सा पिकअप स्थान इस्तेमाल करना चाहिए?",
            f"ఇప్పుడు {order} కోసం ఏ పికప్ స్థలం ఉపయోగించాలి?",
        )
    elif family == "effective_date":
        start, end = f"2026-11-{day:02}", f"2026-12-{i+1:02}"
        add(
            tri(
                f"My service desk is {a}. Use {b} from {start} inclusive until {end} exclusive; then return to {a}.",
                f"मेरा सेवा डेस्क {a} है। {start} से, उस दिन सहित, {end} से पहले तक {b} इस्तेमाल करें; फिर {a} पर लौटें।",
                f"నా సేవా డెస్క్ {a}. {start} రోజుతో సహా, {end} రోజుకు ముందు వరకు {b} ఉపయోగించండి; ఆ తర్వాత {a} కు తిరిగి వెళ్లండి.",
            )
        )
        asof = ["2026-11-01", start, end, "2026-12-25"][i % 4]
        answer = b if start <= asof < end else a
        add(
            tri(
                f"This schedule was recorded on 2026-10-05. A draft for {c} was never approved.",
                f"यह समय-सारणी 2026-10-05 को दर्ज हुई। {c} का मसौदा कभी मंज़ूर नहीं हुआ।",
                f"ఈ షెడ్యూల్ 2026-10-05 న నమోదైంది. {c} కోసం ముసాయిదా ఎప్పుడూ ఆమోదించబడలేదు.",
            )
        )
        question = tri(
            f"Which desk applies on {asof}?",
            f"{asof} को कौन सा डेस्क लागू है?",
            f"{asof} న ఏ డెస్క్ వర్తిస్తుంది?",
        )
    elif family == "conditional":
        approval, arrived = bool(i & 1), bool(i & 2)
        mode = (i // 4) % 3
        rule = [
            tri(
                "Call only if approval AND the replacement part have arrived.",
                "कॉल तभी करें जब मंज़ूरी और नया पुर्जा, दोनों आ चुके हों।",
                "ఆమోదం మరియు కొత్త విడిభాగం రెండూ వచ్చినప్పుడే కాల్ చేయండి.",
            ),
            tri(
                "Call if either approval OR the replacement part has arrived.",
                "मंज़ूरी या नया पुर्जा, इनमें से कोई एक आ गया हो तो कॉल करें।",
                "ఆమోదం లేదా కొత్త విడిభాగం ఏదో ఒకటి వచ్చినా కాల్ చేయండి.",
            ),
            tri(
                "Call if approval has arrived UNLESS the replacement part has also arrived.",
                "मंज़ूरी आ गई हो तो कॉल करें, लेकिन नया पुर्जा भी आ गया हो तो कॉल न करें।",
                "ఆమోదం వస్తే కాల్ చేయండి, కానీ కొత్త విడిభాగం కూడా వచ్చి ఉంటే కాల్ చేయవద్దు.",
            ),
        ][mode]
        add(rule)
        add(
            tri(
                "Approval has arrived." if approval else "Approval has NOT arrived.",
                "मंज़ूरी आ गई है।" if approval else "मंज़ूरी नहीं आई है।",
                "ఆమోదం వచ్చింది." if approval else "ఆమోదం రాలేదు.",
            ),
            "operator",
        )
        add(
            tri(
                "The replacement part has arrived."
                if arrived
                else "The replacement part has NOT arrived.",
                "नया पुर्जा आ गया है।" if arrived else "नया पुर्जा नहीं आया है।",
                "కొత్త విడిభాగం వచ్చింది." if arrived else "కొత్త విడిభాగం రాలేదు.",
            ),
            "operator",
        )
        allowed = [approval and arrived, approval or arrived, approval and not arrived][mode]
        answer = "yes" if allowed else "no"
        question = tri(
            "Is a call allowed now? Answer yes or no; do not make a call.",
            "क्या अब कॉल की अनुमति है? yes या no दें; कॉल न करें।",
            "ఇప్పుడు కాల్ చేయడానికి అనుమతి ఉందా? yes లేదా no ఇవ్వండి; కాల్ చేయవద్దు.",
        )
    elif family == "backend_outcome":
        add(
            tri(
                f"For {order}, confirmed delivery is {current}, version 4.",
                f"{order} की पक्की डिलीवरी {current} है, संस्करण 4।",
                f"{order} కోసం నిర్ధారించిన డెలివరీ {current}, వెర్షన్ 4.",
            ),
            "backend",
        )
        add(
            tri(
                f"Please change {order} delivery to {desired}.",
                f"कृपया {order} की डिलीवरी {desired} कर दें।",
                f"దయచేసి {order} డెలివరీని {desired} కు మార్చండి.",
            )
        )
        status = ["accepted", "rejected", "pending", "accepted"][i % 4]
        add(
            tri(
                f"Change result for {order}: {status}; requested date {desired}; version 5 only if accepted.",
                f"{order} के बदलाव का परिणाम: {status}; अनुरोधित तारीख {desired}; accepted होने पर ही संस्करण 5।",
                f"{order} మార్పు ఫలితం: {status}; కోరిన తేదీ {desired}; accepted అయితేనే వెర్షన్ 5.",
            ),
            "backend",
        )
        add(
            tri(
                f"Delayed duplicate receipt: {order}, version 4, confirmed {current}.",
                f"देर से आई डुप्लिकेट रसीद: {order}, संस्करण 4, पक्की तारीख {current}।",
                f"ఆలస్యంగా వచ్చిన నకిలీ రసీదు: {order}, వెర్షన్ 4, నిర్ధారించిన తేదీ {current}.",
            ),
            "backend",
        )
        answer = desired if status == "accepted" else current
        question = tri(
            f"What is the latest confirmed delivery date for {order}?",
            f"{order} की नवीनतम पक्की डिलीवरी तारीख क्या है?",
            f"{order} కోసం తాజా నిర్ధారించిన డెలివరీ తేదీ ఏమిటి?",
        )
    elif family == "identity":
        x, y = f"C-{i+100}", f"C-{i+200}"
        add(
            tri(
                f"Customer {x}, display name Ravi: pickup code {a}.",
                f"ग्राहक {x}, नाम रवि: पिकअप कोड {a}।",
                f"కస్టమర్ {x}, పేరు రవి: పికప్ కోడ్ {a}.",
            ),
            "backend",
        )
        add(
            tri(
                f"Customer {y}, display name Ravi: pickup code {b}. Shared phone; accounts are separate.",
                f"ग्राहक {y}, नाम रवि: पिकअप कोड {b}। फोन साझा है; खाते अलग हैं।",
                f"కస్టమర్ {y}, పేరు రవి: పికప్ కోడ్ {b}. ఫోన్ ఉమ్మడి; ఖాతాలు వేరు.",
            ),
            "backend",
        )
        target = x if i % 3 == 0 else y
        ambiguous = i % 3 == 2
        who = "Ravi" if ambiguous else target
        answer = None if ambiguous else (a if target == x else b)
        clarification = ambiguous
        question = tri(
            f"What is the pickup code for {who}? No other identity information is available.",
            f"{who} का पिकअप कोड क्या है? पहचान की और जानकारी उपलब्ध नहीं है।",
            f"{who} యొక్క పికప్ కోడ్ ఏమిటి? ఇతర గుర్తింపు సమాచారం అందుబాటులో లేదు.",
        )
    elif family == "arithmetic":
        qty, price, refunded = 3 + i % 5, 125 + i * 7, i % 3
        discount = 20 if yes else 0
        add(
            tri(
                f"Order {order}: {qty} items at INR {price} each. {refunded} items were cancelled and are not charged. Deduct a single INR {discount} order discount after cancellations. No other charges.",
                f"ऑर्डर {order}: {qty} वस्तुएँ, हर एक की कीमत INR {price}। {refunded} वस्तुएँ रद्द हुईं और उनका शुल्क नहीं लगेगा। रद्द करने के बाद पूरे ऑर्डर पर एक बार INR {discount} की छूट घटाएँ। और कोई शुल्क नहीं।",
                f"ఆర్డర్ {order}: {qty} వస్తువులు, ఒక్కొక్కటి INR {price}. {refunded} వస్తువులు రద్దయ్యాయి, వాటికి ఛార్జీ లేదు. రద్దుల తర్వాత మొత్తం ఆర్డర్‌పై ఒక్కసారి INR {discount} తగ్గింపు తీసివేయండి. ఇతర ఛార్జీలు లేవు.",
            ),
            "backend",
        )
        answer = str((qty - refunded) * price - discount)
        question = tri(
            "What is the payable amount in INR? Return digits without currency symbols.",
            "चुकाने की राशि INR में कितनी है? मुद्रा चिह्न के बिना अंक दें।",
            "చెల్లించాల్సిన మొత్తం INR లో ఎంత? కరెన్సీ గుర్తులు లేకుండా అంకెలు ఇవ్వండి.",
        )
    elif family == "distractors":
        count = [12, 36, 72, 108][i % 4]
        target_position = [0, count // 2, count - 1][i % 3]
        for j in range(count):
            oid = order if j == target_position else f"OTHER-{i}-{j}"
            code = b if j == target_position else f"P{j+111}"
            add(
                tri(
                    f"Order {oid}: active pickup code {code}. Status ready; packaging checked.",
                    f"ऑर्डर {oid}: सक्रिय पिकअप कोड {code}। तैयार है; पैकिंग जाँची गई।",
                    f"ఆర్డర్ {oid}: ప్రస్తుత పికప్ కోడ్ {code}. సిద్ధంగా ఉంది; ప్యాకింగ్ తనిఖీ అయింది.",
                ),
                "backend",
            )
        add(
            tri(
                f"Archived note for {order}: old code {a}, now invalid.",
                f"{order} का पुराना नोट: पुराना कोड {a}, अब अमान्य है।",
                f"{order} కోసం పాత గమనిక: పాత కోడ్ {a}, ఇప్పుడు చెల్లదు.",
            ),
            "backend",
        )
        answer = b
        question = tri(
            f"What is the active pickup code for {order}?",
            f"{order} का सक्रिय पिकअप कोड क्या है?",
            f"{order} కోసం ప్రస్తుత పికప్ కోడ్ ఏమిటి?",
        )
    elif family == "ambiguity":
        add(
            tri(
                f"There are two repair tickets: phone ticket {a}, laptop ticket {b}.",
                f"मरम्मत के दो टिकट हैं: फोन का {a}, लैपटॉप का {b}।",
                f"రిపేర్ టికెట్లు రెండు ఉన్నాయి: ఫోన్ టికెట్ {a}, ల్యాప్‌టాప్ టికెట్ {b}.",
            )
        )
        specified = i % 3
        question = [
            tri("Which ticket is for my phone?", "मेरे फोन का टिकट कौन सा है?", "నా ఫోన్ టికెట్ ఏది?"),
            tri("Which ticket is for my laptop?", "मेरे लैपटॉप का टिकट कौन सा है?", "నా ల్యాప్‌టాప్ టికెట్ ఏది?"),
            tri(
                "Which ticket is for my device? No device type was specified.",
                "मेरे डिवाइस का टिकट कौन सा है? डिवाइस का प्रकार नहीं बताया गया।",
                "నా పరికరం టికెట్ ఏది? పరికరం రకం చెప్పలేదు.",
            ),
        ][specified]
        answer = [a, b, None][specified]
        clarification = specified == 2
    elif family in ("action_update", "action_retry"):
        key = f"REQ-{i+700}"
        closed = i % 4 == 3
        approved = i % 4 != 1
        already = family == "action_retry" and yes
        backend["closed"] = closed
        if already:
            backend["date"] = desired
            backend["version"] = 5
            backend["operations"] = [{"key": key, "date": desired}]
        add(
            tri(
                f"Appointment {order}: date {backend['date']}, version {backend['version']}, closed={str(closed).lower()}.",
                f"अपॉइंटमेंट {order}: तारीख {backend['date']}, संस्करण {backend['version']}, closed={str(closed).lower()}।",
                f"అపాయింట్‌మెంట్ {order}: తేదీ {backend['date']}, వెర్షన్ {backend['version']}, closed={str(closed).lower()}.",
            ),
            "backend",
        )
        add(
            tri(
                "Manager approval has arrived."
                if approved
                else "Manager approval has NOT arrived.",
                "मैनेजर की मंज़ूरी आ गई है।" if approved else "मैनेजर की मंज़ूरी नहीं आई है।",
                "మేనేజర్ ఆమోదం వచ్చింది." if approved else "మేనేజర్ ఆమోదం రాలేదు.",
            ),
            "operator",
        )
        if already:
            add(
                tri(
                    f"Receipt: request {key} was already applied to {order} with date {desired}. Do not create a second operation.",
                    f"रसीद: अनुरोध {key} पहले ही {order} पर तारीख {desired} के साथ लागू हो चुका है। दूसरा ऑपरेशन न बनाएँ।",
                    f"రసీదు: అభ్యర్థన {key} ఇప్పటికే {order} పై తేదీ {desired} తో అమలైంది. రెండో ఆపరేషన్ సృష్టించవద్దు.",
                ),
                "backend",
            )
        answer = desired if (approved and not closed) or already else current
        writes = int(approved and not closed and not already)
        action_status = (
            "already_applied"
            if already
            else "not_performed"
            if not approved
            else "rejected"
            if closed
            else "confirmed"
        )
        question = tri(
            f"Only if manager approval has arrived, use request key {key} to change appointment {order} to {desired}. If already applied, report already_applied. Return the actual final date, and the action status. Never claim success without a tool receipt. If approval is absent, report not_performed. A closed appointment rejects changes. If approved and not already applied, attempt the tool even when closed and report its actual result.",
            f"मैनेजर की मंज़ूरी आ गई हो तभी अनुरोध कुंजी {key} से अपॉइंटमेंट {order} को {desired} करें। पहले लागू हो चुका हो तो already_applied बताएँ। वास्तविक अंतिम तारीख और कार्रवाई की स्थिति दें। टूल की रसीद के बिना सफलता न बताएँ। मंज़ूरी न हो तो not_performed बताएँ। बंद अपॉइंटमेंट में बदलाव अस्वीकार होता है। मंज़ूरी हो और पहले लागू न हुआ हो तो बंद होने पर भी टूल का प्रयास करें और उसका वास्तविक परिणाम बताएँ।",
            f"మేనేజర్ ఆమోదం వచ్చి ఉంటేనే అభ్యర్థన కీ {key} తో అపాయింట్‌మెంట్ {order} ను {desired} కు మార్చండి. ఇప్పటికే అమలైతే already_applied చెప్పండి. నిజమైన చివరి తేదీ మరియు చర్య స్థితి ఇవ్వండి. టూల్ రసీదు లేకుండా విజయం చెప్పవద్దు. ఆమోదం లేకపోతే not_performed చెప్పండి. మూసివేసిన అపాయింట్‌మెంట్ మార్పులను తిరస్కరిస్తుంది. ఆమోదం ఉండి, ఇప్పటికే అమలు కాకపోతే, మూసివేసినా టూల్‌ను ప్రయత్నించి దాని నిజమైన ఫలితాన్ని చెప్పండి.",
        )
    else:
        raise ValueError(family)
    return (
        records,
        question,
        backend,
        {
            "answer": answer,
            "needs_clarification": clarification,
            "action_status": action_status,
            "writes": writes,
            "forbid_mutation": action_status == "not_performed",
        },
    )


def build():
    runtime, gold = [], {}
    for family in FAMILIES:
        for i in range(20):
            records, question, backend, expected = scenario(family, i)
            base = f"{family}-{i:02}"
            template_group = f"{family}-pattern-{i%10}"
            split = "confirmation" if i % 10 >= 7 else "discovery"
            renderings = list(LANGS) + (["mixed"] if i % 4 in (0, 3) else [])
            for lang in renderings:
                cid = base + "-" + lang
                rows = [
                    {**r, "text": r["text"][LANGS[j % 3] if lang == "mixed" else lang]}
                    for j, r in enumerate(records)
                ]
                # Native Hindi question with evidence rotating English/Hindi/Telugu.
                q = question["hi" if lang == "mixed" else lang]
                case = {
                    "id": cid,
                    "base_id": base,
                    "template_group": template_group,
                    "family": family,
                    "language": lang,
                    "split": split,
                    "question": q,
                    "records": rows,
                    "backend": backend,
                    "review": "ASSISTANT_TEMPLATE_REVIEWED_NOT_INDEPENDENT_NATIVE_VALIDATED",
                }
                runtime.append(case)
                gold[cid] = {**expected, "runtime_sha256": digest(case)}
    # Schedule order cannot reveal target position/outcome, and cases aren't renamed after outputs.
    random.Random(20261011).shuffle(runtime)
    return runtime, gold


def prepare(directory):
    directory = Path(directory)
    directory.mkdir()
    runtime, gold = build()
    save(directory / "runtime.json", runtime)
    save(directory / "gold.json", gold)
    save(
        directory / "review.json",
        {
            "runtime_sha256": digest(runtime),
            "gold_sha256": digest(gold),
            "renderings": len(runtime),
            "underlying_scenarios": 200,
            "template_groups": 100,
            "families": list(FAMILIES),
            "discovery_renderings": sum(c["split"] == "discovery" for c in runtime),
            "confirmation_renderings": sum(c["split"] == "confirmation" for c in runtime),
            "review": "Assistant authored and checked parallel templates; no independent native validation",
            "limitations": [
                "Parameterized synthetic scenarios, not 200 independent real customer workflows",
                "Template families overlap across splits; confirmation is internal, not external validation",
                "Mixed cases switch source languages; no Romanized-input or speech claim",
                "Longer histories here do not test the full advertised context windows",
                "No generated memory and no AgentMem runtime",
            ],
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    prepare(p.parse_args().directory)
