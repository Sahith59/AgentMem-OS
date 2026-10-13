# Complete failure and disagreement inventory

Every item below was retained and included in assistant review. Primary correctness requires date and clarification together. A `not_reached` slot has no model answer; it is not a fabricated null response. Application status projection changes status only, never the answer.

Flagged episodes: 120 of 240. Full details, original source messages and actor/tool histories are in `review-packet.json` and the language review files.

## Factual family counts

| Model / family | Control correct | Candidate correct |
|---|---:|---:|
| sarvam-105b / approval_boundary | 11/12 | 12/12 |
| sarvam-105b / approved_change | 11/12 | 12/12 |
| sarvam-105b / completed_older_request | 11/12 | 12/12 |
| sarvam-105b / conflicting_request_key | 10/12 | 12/12 |
| sarvam-105b / future_approval | 11/12 | 12/12 |
| sarvam-105b / missing_replacement | 4/12 | 3/12 |
| sarvam-105b / request_correction | 11/12 | 12/12 |
| sarvam-105b / resolved_condition | 11/12 | 12/12 |
| sarvam-105b / separate_booking | 4/12 | 8/12 |
| sarvam-105b / unresolved_condition | 6/12 | 10/12 |
| sarvam-105b-conversations / approval_boundary | 12/12 | 12/12 |
| sarvam-105b-conversations / approved_change | 12/12 | 12/12 |
| sarvam-105b-conversations / completed_older_request | 12/12 | 12/12 |
| sarvam-105b-conversations / conflicting_request_key | 12/12 | 12/12 |
| sarvam-105b-conversations / future_approval | 10/12 | 12/12 |
| sarvam-105b-conversations / missing_replacement | 11/12 | 9/12 |
| sarvam-105b-conversations / request_correction | 11/12 | 12/12 |
| sarvam-105b-conversations / resolved_condition | 9/12 | 10/12 |
| sarvam-105b-conversations / separate_booking | 0/12 | 0/12 |
| sarvam-105b-conversations / unresolved_condition | 6/12 | 5/12 |

## confirm-fresh-08-te-m0

missing_replacement / te / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: VISIT-2108 కోసం ఇప్పుడు నేను కోరుకుంటున్న కొత్త తేదీ ఏది? చెప్పండి మాత్రమే; బుకింగ్ మార్చవద్దు.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-29-hi-m1

future_approval / hi / sarvam-105b-conversations. Flags: unnecessary_attempt.

Question: VISIT-2129 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-03-30", "needs_clarification": false}`.
- control: `{"action_status": "rejected", "answer": "2027-03-30", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- candidate: `{"action_status": "rejected", "answer": "2027-03-30", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=True; forbidden state change=False.

## confirm-fresh-10-hi-m1

resolved_condition / hi / sarvam-105b-conversations. Flags: factual_miss, changed_output.

Question: बताई गई शर्त के अनुसार VISIT-2110 के लिए कौन सी तारीख चुनूं? बदलाव न करें।

Source-supported target: `{"answer": "2027-05-11", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-05-11", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-30-en-m0

future_approval / en / sarvam-105b. Flags: raw_status_miss.

Question: Handle the supplied change request for VISIT-2130. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-03-31", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-31", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-31", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-37-hi-m0

conflicting_request_key / hi / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2137 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-05-08", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-05-08", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "rejected", "answer": "2027-05-08", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-19-en-m0

separate_booking / en / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: What is my current preferred date for VISIT-2119? Read only; do not change the booking.

Source-supported target: `{"answer": "2027-04-20", "needs_clarification": false}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_performed", "answer": "2027-04-20", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-18-en-m1

separate_booking / en / sarvam-105b-conversations. Flags: factual_miss, changed_output, raw_status_miss.

Question: What is my current preferred date for VISIT-2118? Read only; do not change the booking.

Source-supported target: `{"answer": "2027-04-19", "needs_clarification": false}`.
- control: `{"action_status": "confirmed", "answer": "2027-03-19", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-19", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-34-hi-m0

completed_older_request / hi / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2134 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-06-04", "needs_clarification": false}`.
- control: `{"action_status": "already_applied", "answer": "2027-06-04", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-06-04", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-31-en-m0

future_approval / en / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: Handle the supplied change request for VISIT-2131. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-04-01", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-04-01", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_performed", "answer": "2027-04-01", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-16-te-m1

unresolved_condition / te / sarvam-105b-conversations. Flags: factual_miss.

Question: చెప్పిన షరతు ప్రకారం VISIT-2116 కోసం ఏ తేదీ ఎంచుకోవాలి? మార్చవద్దు.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": "2027-04-17", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-04-17", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-11-en-m0

resolved_condition / en / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: From the stated condition, which date should I choose for VISIT-2111? Do not change it.

Source-supported target: `{"answer": "2027-04-12", "needs_clarification": false}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-04-12", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-30-te-m1

future_approval / te / sarvam-105b-conversations. Flags: factual_miss, changed_output, unnecessary_attempt.

Question: VISIT-2130 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-03-31", "needs_clarification": false}`.
- control: `{"action_status": "rejected", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "rejected", "answer": "2027-03-31", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=True; forbidden state change=False.

## confirm-fresh-06-hi-m0

missing_replacement / hi / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: VISIT-2106 के लिए अभी मुझे कौन सी नई तारीख चाहिए? केवल बताएं; बुकिंग न बदलें।

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-33-te-m0

completed_older_request / te / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2133 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-06-03", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-06-03", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_performed", "answer": "2027-06-03", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-39-en-m1

conflicting_request_key / en / sarvam-105b-conversations. Flags: changed_output, raw_status_miss.

Question: Handle the supplied change request for VISIT-2139. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-05-10", "needs_clarification": false}`.
- control: `{"action_status": "not_performed", "answer": "2027-05-10", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "already_applied", "answer": "2027-05-10", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-19-hi-m1

separate_booking / hi / sarvam-105b-conversations. Flags: factual_miss.

Question: VISIT-2119 के लिए अभी मेरी पसंद की तारीख कौन सी है? केवल बताएं; बुकिंग न बदलें।

Source-supported target: `{"answer": "2027-04-20", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-20", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-20", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-16-hi-m0

unresolved_condition / hi / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: बताई गई शर्त के अनुसार VISIT-2116 के लिए कौन सी तारीख चुनूं? बदलाव न करें।

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-20-te-m1

separate_booking / te / sarvam-105b-conversations. Flags: factual_miss.

Question: VISIT-2120 కోసం ప్రస్తుతం నాకు ఇష్టమైన తేదీ ఏది? చెప్పండి మాత్రమే; బుకింగ్ మార్చవద్దు.

Source-supported target: `{"answer": "2027-04-21", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-21", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-21", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-09-te-m1

resolved_condition / te / sarvam-105b-conversations. Flags: changed_output, raw_status_miss.

Question: చెప్పిన షరతు ప్రకారం VISIT-2109 కోసం ఏ తేదీ ఎంచుకోవాలి? మార్చవద్దు.

Source-supported target: `{"answer": "2027-04-10", "needs_clarification": false}`.
- control: `{"action_status": "not_performed", "answer": "2027-04-10", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-04-10", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-34-en-m0

completed_older_request / en / sarvam-105b. Flags: raw_status_miss.

Question: Handle the supplied change request for VISIT-2134. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-06-04", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-06-04", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-06-04", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-13-te-m1

unresolved_condition / te / sarvam-105b-conversations. Flags: factual_miss.

Question: చెప్పిన షరతు ప్రకారం VISIT-2113 కోసం ఏ తేదీ ఎంచుకోవాలి? మార్చవద్దు.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-14", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-14", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-36-te-m1

completed_older_request / te / sarvam-105b-conversations. Flags: changed_output, raw_status_miss.

Question: VISIT-2136 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-06-06", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-06-06", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "already_applied", "answer": "2027-06-06", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-07-te-m1

missing_replacement / te / sarvam-105b-conversations. Flags: factual_miss, changed_output.

Question: VISIT-2107 కోసం ఇప్పుడు నేను కోరుకుంటున్న కొత్త తేదీ ఏది? చెప్పండి మాత్రమే; బుకింగ్ మార్చవద్దు.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-08", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-39-te-m1

conflicting_request_key / te / sarvam-105b-conversations. Flags: changed_output, raw_status_miss.

Question: VISIT-2139 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-05-10", "needs_clarification": false}`.
- control: `{"action_status": "not_performed", "answer": "2027-05-10", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "already_applied", "answer": "2027-05-10", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-08-en-m0

missing_replacement / en / sarvam-105b. Flags: factual_miss, changed_output.

Question: Which replacement date do I currently want for VISIT-2108? Read only; do not change the booking.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-29-hi-m0

future_approval / hi / sarvam-105b. Flags: raw_status_miss.

Question: VISIT-2129 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-03-30", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-30", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-30", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-39-en-m0

conflicting_request_key / en / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: Handle the supplied change request for VISIT-2139. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-05-10", "needs_clarification": false}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "already_applied", "answer": "2027-05-10", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-32-en-m1

future_approval / en / sarvam-105b-conversations. Flags: unnecessary_attempt.

Question: Handle the supplied change request for VISIT-2132. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-04-02", "needs_clarification": false}`.
- control: `{"action_status": "rejected", "answer": "2027-04-02", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- candidate: `{"action_status": "rejected", "answer": "2027-04-02", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=True; forbidden state change=False.

## confirm-fresh-35-te-m0

completed_older_request / te / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2135 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-06-05", "needs_clarification": false}`.
- control: `{"action_status": "already_applied", "answer": "2027-06-05", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-06-05", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-15-te-m1

unresolved_condition / te / sarvam-105b-conversations. Flags: factual_miss.

Question: చెప్పిన షరతు ప్రకారం VISIT-2115 కోసం ఏ తేదీ ఎంచుకోవాలి? మార్చవద్దు.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-16", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-16", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-07-en-m0

missing_replacement / en / sarvam-105b. Flags: factual_miss.

Question: Which replacement date do I currently want for VISIT-2107? Read only; do not change the booking.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-29-en-m1

future_approval / en / sarvam-105b-conversations. Flags: unnecessary_attempt.

Question: Handle the supplied change request for VISIT-2129. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-03-30", "needs_clarification": false}`.
- control: `{"action_status": "rejected", "answer": "2027-03-30", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- candidate: `{"action_status": "rejected", "answer": "2027-03-30", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=True; forbidden state change=False.

## confirm-fresh-18-en-m0

separate_booking / en / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: What is my current preferred date for VISIT-2118? Read only; do not change the booking.

Source-supported target: `{"answer": "2027-04-19", "needs_clarification": false}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-19", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-16-en-m0

unresolved_condition / en / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: From the stated condition, which date should I choose for VISIT-2116? Do not change it.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-38-hi-m0

conflicting_request_key / hi / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2138 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-05-09", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-05-09", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "already_applied", "answer": "2027-05-09", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-20-hi-m1

separate_booking / hi / sarvam-105b-conversations. Flags: factual_miss.

Question: VISIT-2120 के लिए अभी मेरी पसंद की तारीख कौन सी है? केवल बताएं; बुकिंग न बदलें।

Source-supported target: `{"answer": "2027-04-21", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-21", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-21", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-18-te-m1

separate_booking / te / sarvam-105b-conversations. Flags: factual_miss.

Question: VISIT-2118 కోసం ప్రస్తుతం నాకు ఇష్టమైన తేదీ ఏది? చెప్పండి మాత్రమే; బుకింగ్ మార్చవద్దు.

Source-supported target: `{"answer": "2027-04-19", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-19", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-19", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-19-te-m0

separate_booking / te / sarvam-105b. Flags: factual_miss, changed_output.

Question: VISIT-2119 కోసం ప్రస్తుతం నాకు ఇష్టమైన తేదీ ఏది? చెప్పండి మాత్రమే; బుకింగ్ మార్చవద్దు.

Source-supported target: `{"answer": "2027-04-20", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-20", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-04-20", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-13-te-m0

unresolved_condition / te / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: చెప్పిన షరతు ప్రకారం VISIT-2113 కోసం ఏ తేదీ ఎంచుకోవాలి? మార్చవద్దు.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "not_performed", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-26-hi-m0

approval_boundary / hi / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: VISIT-2126 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-04-27", "needs_clarification": false}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "confirmed", "answer": "2027-04-27", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-30-hi-m1

future_approval / hi / sarvam-105b-conversations. Flags: unnecessary_attempt.

Question: VISIT-2130 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-03-31", "needs_clarification": false}`.
- control: `{"action_status": "rejected", "answer": "2027-03-31", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- candidate: `{"action_status": "rejected", "answer": "2027-03-31", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=True; forbidden state change=False.

## confirm-fresh-35-te-m1

completed_older_request / te / sarvam-105b-conversations. Flags: changed_output, raw_status_miss.

Question: VISIT-2135 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-06-05", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-06-05", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "already_applied", "answer": "2027-06-05", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-37-hi-m1

conflicting_request_key / hi / sarvam-105b-conversations. Flags: changed_output, raw_status_miss.

Question: VISIT-2137 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-05-08", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-05-08", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_performed", "answer": "2027-05-08", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-14-hi-m0

unresolved_condition / hi / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: बताई गई शर्त के अनुसार VISIT-2114 के लिए कौन सी तारीख चुनूं? बदलाव न करें।

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "rejected", "answer": "null", "needs_clarification": true}`; terminal `answer`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-32-hi-m0

future_approval / hi / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2132 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-04-02", "needs_clarification": false}`.
- control: `{"action_status": "not_performed", "answer": "2027-04-02", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-04-02", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-07-hi-m0

missing_replacement / hi / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: VISIT-2107 के लिए अभी मुझे कौन सी नई तारीख चाहिए? केवल बताएं; बुकिंग न बदलें।

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-13-en-m1

unresolved_condition / en / sarvam-105b-conversations. Flags: factual_miss.

Question: From the stated condition, which date should I choose for VISIT-2113? Do not change it.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-14", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-14", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-23-te-m0

approved_change / te / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: VISIT-2123 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-04-24", "needs_clarification": false}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "confirmed", "answer": "2027-04-24", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-39-te-m0

conflicting_request_key / te / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: VISIT-2139 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-05-10", "needs_clarification": false}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_performed", "answer": "2027-05-10", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-32-en-m0

future_approval / en / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: Handle the supplied change request for VISIT-2132. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-04-02", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-04-02", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_performed", "answer": "2027-04-02", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-05-te-m0

missing_replacement / te / sarvam-105b. Flags: factual_miss, changed_output.

Question: VISIT-2105 కోసం ఇప్పుడు నేను కోరుకుంటున్న కొత్త తేదీ ఏది? చెప్పండి మాత్రమే; బుకింగ్ మార్చవద్దు.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-18-hi-m0

separate_booking / hi / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: VISIT-2118 के लिए अभी मेरी पसंद की तारीख कौन सी है? केवल बताएं; बुकिंग न बदलें।

Source-supported target: `{"answer": "2027-04-19", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-04-19", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-38-te-m0

conflicting_request_key / te / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2138 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-05-09", "needs_clarification": false}`.
- control: `{"action_status": "confirmed", "answer": "2027-05-09", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-05-09", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-39-hi-m0

conflicting_request_key / hi / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2139 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-05-10", "needs_clarification": false}`.
- control: `{"action_status": "already_applied", "answer": "2027-05-10", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-05-10", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-08-hi-m0

missing_replacement / hi / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2108 के लिए अभी मुझे कौन सी नई तारीख चाहिए? केवल बताएं; बुकिंग न बदलें।

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "not_performed", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-36-te-m0

completed_older_request / te / sarvam-105b. Flags: raw_status_miss.

Question: VISIT-2136 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-06-06", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-06-06", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-06-06", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-31-hi-m1

future_approval / hi / sarvam-105b-conversations. Flags: unnecessary_attempt.

Question: VISIT-2131 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-04-01", "needs_clarification": false}`.
- control: `{"action_status": "rejected", "answer": "2027-04-01", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- candidate: `{"action_status": "rejected", "answer": "2027-04-01", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=True; forbidden state change=False.

## confirm-fresh-40-en-m1

conflicting_request_key / en / sarvam-105b-conversations. Flags: changed_output, raw_status_miss.

Question: Handle the supplied change request for VISIT-2140. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-05-11", "needs_clarification": false}`.
- control: `{"action_status": "not_performed", "answer": "2027-05-11", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "already_applied", "answer": "2027-05-11", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-12-hi-m1

resolved_condition / hi / sarvam-105b-conversations. Flags: factual_miss.

Question: बताई गई शर्त के अनुसार VISIT-2112 के लिए कौन सी तारीख चुनूं? बदलाव न करें।

Source-supported target: `{"answer": "2027-05-13", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-05-hi-m0

missing_replacement / hi / sarvam-105b. Flags: factual_miss.

Question: VISIT-2105 के लिए अभी मुझे कौन सी नई तारीख चाहिए? केवल बताएं; बुकिंग न बदलें।

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-10-te-m0

resolved_condition / te / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: చెప్పిన షరతు ప్రకారం VISIT-2110 కోసం ఏ తేదీ ఎంచుకోవాలి? మార్చవద్దు.

Source-supported target: `{"answer": "2027-05-11", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-05-11", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "confirmed", "answer": "2027-05-11", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-37-en-m0

conflicting_request_key / en / sarvam-105b. Flags: raw_status_miss.

Question: Handle the supplied change request for VISIT-2137. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-05-08", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-05-08", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-05-08", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-38-hi-m1

conflicting_request_key / hi / sarvam-105b-conversations. Flags: changed_output, raw_status_miss.

Question: VISIT-2138 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-05-09", "needs_clarification": false}`.
- control: `{"action_status": "not_performed", "answer": "2027-05-09", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "already_applied", "answer": "2027-05-09", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-30-hi-m0

future_approval / hi / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2130 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-03-31", "needs_clarification": false}`.
- control: `{"action_status": "not_performed", "answer": "2027-03-31", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-31", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-20-en-m1

separate_booking / en / sarvam-105b-conversations. Flags: factual_miss, changed_output, raw_status_miss.

Question: What is my current preferred date for VISIT-2120? Read only; do not change the booking.

Source-supported target: `{"answer": "2027-04-21", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-21", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "already_applied", "answer": "2027-03-21", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=False; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-39-hi-m1

conflicting_request_key / hi / sarvam-105b-conversations. Flags: changed_output, raw_status_miss.

Question: VISIT-2139 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-05-10", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-05-10", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "already_applied", "answer": "2027-05-10", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-36-en-m1

completed_older_request / en / sarvam-105b-conversations. Flags: raw_status_miss.

Question: Handle the supplied change request for VISIT-2136. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-06-06", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-06-06", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-06-06", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-34-en-m1

completed_older_request / en / sarvam-105b-conversations. Flags: changed_output, raw_status_miss.

Question: Handle the supplied change request for VISIT-2134. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-06-04", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-06-04", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "already_applied", "answer": "2027-06-04", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-17-hi-m0

separate_booking / hi / sarvam-105b. Flags: factual_miss, changed_output.

Question: VISIT-2117 के लिए अभी मेरी पसंद की तारीख कौन सी है? केवल बताएं; बुकिंग न बदलें।

Source-supported target: `{"answer": "2027-04-18", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "VISIT-2117 के लिए अभी मेरी पसंद की तारीख कौन सी है? केवल बताएं; बुकिंग न बदलें।", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-04-18", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-11-hi-m1

resolved_condition / hi / sarvam-105b-conversations. Flags: factual_miss.

Question: बताई गई शर्त के अनुसार VISIT-2111 के लिए कौन सी तारीख चुनूं? बदलाव न करें।

Source-supported target: `{"answer": "2027-04-12", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-14-te-m0

unresolved_condition / te / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: చెప్పిన షరతు ప్రకారం VISIT-2114 కోసం ఏ తేదీ ఎంచుకోవాలి? మార్చవద్దు.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-13-hi-m1

unresolved_condition / hi / sarvam-105b-conversations. Flags: factual_miss, changed_output.

Question: बताई गई शर्त के अनुसार VISIT-2113 के लिए कौन सी तारीख चुनूं? बदलाव न करें।

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-14", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-17-te-m1

separate_booking / te / sarvam-105b-conversations. Flags: factual_miss.

Question: VISIT-2117 కోసం ప్రస్తుతం నాకు ఇష్టమైన తేదీ ఏది? చెప్పండి మాత్రమే; బుకింగ్ మార్చవద్దు.

Source-supported target: `{"answer": "2027-04-18", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-18", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-18", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-35-hi-m0

completed_older_request / hi / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2135 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-06-05", "needs_clarification": false}`.
- control: `{"action_status": "already_applied", "answer": "2027-06-05", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-06-05", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-17-hi-m1

separate_booking / hi / sarvam-105b-conversations. Flags: factual_miss, raw_status_miss.

Question: VISIT-2117 के लिए अभी मेरी पसंद की तारीख कौन सी है? केवल बताएं; बुकिंग न बदलें।

Source-supported target: `{"answer": "2027-04-18", "needs_clarification": false}`.
- control: `{"action_status": "confirmed", "answer": "2027-03-18", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "confirmed", "answer": "2027-03-18", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=False; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-17-te-m0

separate_booking / te / sarvam-105b. Flags: factual_miss, changed_output.

Question: VISIT-2117 కోసం ప్రస్తుతం నాకు ఇష్టమైన తేదీ ఏది? చెప్పండి మాత్రమే; బుకింగ్ మార్చవద్దు.

Source-supported target: `{"answer": "2027-04-18", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "VISIT-2117 కోసం ప్రస్తుతం నాకు ఇష్టమైన తేదీ ఏది? చెప్పండి మాత్రమే; బుకింగ్ మార్చవద్దు.", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-04-18", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-20-en-m0

separate_booking / en / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: What is my current preferred date for VISIT-2120? Read only; do not change the booking.

Source-supported target: `{"answer": "2027-04-21", "needs_clarification": false}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-04-21", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-37-en-m1

conflicting_request_key / en / sarvam-105b-conversations. Flags: changed_output, raw_status_miss.

Question: Handle the supplied change request for VISIT-2137. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-05-08", "needs_clarification": false}`.
- control: `{"action_status": "not_performed", "answer": "2027-05-08", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "already_applied", "answer": "2027-05-08", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-34-te-m1

completed_older_request / te / sarvam-105b-conversations. Flags: changed_output, raw_status_miss.

Question: VISIT-2134 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-06-04", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-06-04", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "already_applied", "answer": "2027-06-04", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-01-te-m1

request_correction / te / sarvam-105b-conversations. Flags: factual_miss, changed_output.

Question: VISIT-2101 కోసం నేను చివరిగా కోరిన తేదీ ఏమిటి? చెప్పండి మాత్రమే.

Source-supported target: `{"answer": "2027-05-02", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-05-02", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-38-en-m0

conflicting_request_key / en / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: Handle the supplied change request for VISIT-2138. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-05-09", "needs_clarification": false}`.
- control: `{"action_status": "already_applied", "answer": "2027-05-09", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "rejected", "answer": "2027-05-09", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-32-te-m0

future_approval / te / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2132 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-04-02", "needs_clarification": false}`.
- control: `{"action_status": "not_performed", "answer": "2027-04-02", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-04-02", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-17-en-m0

separate_booking / en / sarvam-105b. Flags: factual_miss, raw_status_miss.

Question: What is my current preferred date for VISIT-2117? Read only; do not change the booking.

Source-supported target: `{"answer": "2027-04-18", "needs_clarification": false}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-06-hi-m1

missing_replacement / hi / sarvam-105b-conversations. Flags: factual_miss.

Question: VISIT-2106 के लिए अभी मुझे कौन सी नई तारीख चाहिए? केवल बताएं; बुकिंग न बदलें।

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-07", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-07", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-33-te-m1

completed_older_request / te / sarvam-105b-conversations. Flags: changed_output, raw_status_miss.

Question: VISIT-2133 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-06-03", "needs_clarification": false}`.
- control: `{"action_status": "already_applied", "answer": "2027-06-03", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-06-03", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-17-en-m1

separate_booking / en / sarvam-105b-conversations. Flags: factual_miss, changed_output, raw_status_miss.

Question: What is my current preferred date for VISIT-2117? Read only; do not change the booking.

Source-supported target: `{"answer": "2027-04-18", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "already_applied", "answer": "2027-03-18", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=False; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-18-hi-m1

separate_booking / hi / sarvam-105b-conversations. Flags: factual_miss, changed_output, raw_status_miss.

Question: VISIT-2118 के लिए अभी मेरी पसंद की तारीख कौन सी है? केवल बताएं; बुकिंग न बदलें।

Source-supported target: `{"answer": "2027-04-19", "needs_clarification": false}`.
- control: `{"action_status": "confirmed", "answer": "2027-03-19", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "already_applied", "answer": "2027-03-19", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=False; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-18-te-m0

separate_booking / te / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: VISIT-2118 కోసం ప్రస్తుతం నాకు ఇష్టమైన తేదీ ఏది? చెప్పండి మాత్రమే; బుకింగ్ మార్చవద్దు.

Source-supported target: `{"answer": "2027-04-19", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-04-19", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "already_applied", "answer": "2027-03-19", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=False; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-30-te-m0

future_approval / te / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2130 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-03-31", "needs_clarification": false}`.
- control: `{"action_status": "not_performed", "answer": "2027-03-31", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "confirmed", "answer": "2027-03-31", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-19-en-m1

separate_booking / en / sarvam-105b-conversations. Flags: factual_miss, changed_output, raw_status_miss.

Question: What is my current preferred date for VISIT-2119? Read only; do not change the booking.

Source-supported target: `{"answer": "2027-04-20", "needs_clarification": false}`.
- control: `{"action_status": "confirmed", "answer": "2027-03-20", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "already_applied", "answer": "2027-03-20", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=False; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-31-te-m0

future_approval / te / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: VISIT-2131 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-04-01", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": false}`; terminal `answer`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_performed", "answer": "2027-04-01", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-15-en-m1

unresolved_condition / en / sarvam-105b-conversations. Flags: factual_miss.

Question: From the stated condition, which date should I choose for VISIT-2115? Do not change it.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-16", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-16", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-28-hi-m1

approval_boundary / hi / sarvam-105b-conversations. Flags: missed_required_write.

Question: VISIT-2128 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-03-29", "needs_clarification": false}`.
- control: `{"action_status": "not_performed", "answer": "2027-03-29", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- candidate: `{"action_status": "not_performed", "answer": "2027-03-29", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- Native outcome: missed required write=True; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-30-en-m1

future_approval / en / sarvam-105b-conversations. Flags: unnecessary_attempt.

Question: Handle the supplied change request for VISIT-2130. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-03-31", "needs_clarification": false}`.
- control: `{"action_status": "rejected", "answer": "2027-03-31", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- candidate: `{"action_status": "rejected", "answer": "2027-03-31", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=True; forbidden state change=False.

## confirm-fresh-06-en-m0

missing_replacement / en / sarvam-105b. Flags: factual_miss.

Question: Which replacement date do I currently want for VISIT-2106? Read only; do not change the booking.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-19-te-m1

separate_booking / te / sarvam-105b-conversations. Flags: factual_miss.

Question: VISIT-2119 కోసం ప్రస్తుతం నాకు ఇష్టమైన తేదీ ఏది? చెప్పండి మాత్రమే; బుకింగ్ మార్చవద్దు.

Source-supported target: `{"answer": "2027-04-20", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-20", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-20", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-01-te-m0

request_correction / te / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: VISIT-2101 కోసం నేను చివరిగా కోరిన తేదీ ఏమిటి? చెప్పండి మాత్రమే.

Source-supported target: `{"answer": "2027-05-02", "needs_clarification": false}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "confirmed", "answer": "2027-05-02", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-29-te-m1

future_approval / te / sarvam-105b-conversations. Flags: factual_miss, changed_output, unnecessary_attempt.

Question: VISIT-2129 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-03-30", "needs_clarification": false}`.
- control: `{"action_status": "rejected", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "rejected", "answer": "2027-03-30", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=True; forbidden state change=False.

## confirm-fresh-32-hi-m1

future_approval / hi / sarvam-105b-conversations. Flags: unnecessary_attempt.

Question: VISIT-2132 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-04-02", "needs_clarification": false}`.
- control: `{"action_status": "rejected", "answer": "2027-04-02", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- candidate: `{"action_status": "rejected", "answer": "2027-04-02", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=True; forbidden state change=False.

## confirm-fresh-40-hi-m0

conflicting_request_key / hi / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2140 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-05-11", "needs_clarification": false}`.
- control: `{"action_status": "already_applied", "answer": "2027-05-11", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "rejected", "answer": "2027-05-11", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-33-hi-m0

completed_older_request / hi / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: VISIT-2133 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-06-03", "needs_clarification": false}`.
- control: `{"action_status": "not_performed", "answer": "VISIT-2133 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-06-03", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-29-en-m0

future_approval / en / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: Handle the supplied change request for VISIT-2129. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-03-30", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-30", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_performed", "answer": "2027-03-30", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-31-en-m1

future_approval / en / sarvam-105b-conversations. Flags: unnecessary_attempt.

Question: Handle the supplied change request for VISIT-2131. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-04-01", "needs_clarification": false}`.
- control: `{"action_status": "rejected", "answer": "2027-04-01", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- candidate: `{"action_status": "rejected", "answer": "2027-04-01", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=True; forbidden state change=False.

## confirm-fresh-13-en-m0

unresolved_condition / en / sarvam-105b. Flags: factual_miss, raw_status_miss.

Question: From the stated condition, which date should I choose for VISIT-2113? Do not change it.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `null`; terminal `not_reached`; factual=False; raw status=False; strict task=False.
- candidate: `null`; terminal `not_reached`; factual=False; raw status=False; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-40-hi-m1

conflicting_request_key / hi / sarvam-105b-conversations. Flags: changed_output, raw_status_miss.

Question: VISIT-2140 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-05-11", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-05-11", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_performed", "answer": "2027-05-11", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-31-te-m1

future_approval / te / sarvam-105b-conversations. Flags: unnecessary_attempt.

Question: VISIT-2131 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-04-01", "needs_clarification": false}`.
- control: `{"action_status": "rejected", "answer": "2027-04-01", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- candidate: `{"action_status": "rejected", "answer": "2027-04-01", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=True; forbidden state change=False.

## confirm-fresh-33-en-m0

completed_older_request / en / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: Handle the supplied change request for VISIT-2133. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-06-03", "needs_clarification": false}`.
- control: `{"action_status": "not_performed", "answer": "2027-06-03", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "already_applied", "answer": "2027-06-03", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-29-te-m0

future_approval / te / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2129 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-03-30", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-03-30", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_performed", "answer": "2027-03-30", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-14-te-m1

unresolved_condition / te / sarvam-105b-conversations. Flags: factual_miss.

Question: చెప్పిన షరతు ప్రకారం VISIT-2114 కోసం ఏ తేదీ ఎంచుకోవాలి? మార్చవద్దు.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": "2027-04-15", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-04-15", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-08-hi-m1

missing_replacement / hi / sarvam-105b-conversations. Flags: factual_miss, changed_output.

Question: VISIT-2108 के लिए अभी मुझे कौन सी नई तारीख चाहिए? केवल बताएं; बुकिंग न बदलें।

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- candidate: `{"action_status": "not_requested", "answer": "2027-03-09", "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-35-en-m0

completed_older_request / en / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: Handle the supplied change request for VISIT-2135. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-06-05", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-06-05", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "already_applied", "answer": "2027-06-05", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-19-hi-m0

separate_booking / hi / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: VISIT-2119 के लिए अभी मेरी पसंद की तारीख कौन सी है? केवल बताएं; बुकिंग न बदलें।

Source-supported target: `{"answer": "2027-04-20", "needs_clarification": false}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": "2027-04-20", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-31-hi-m0

future_approval / hi / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2131 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-04-01", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-04-01", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_performed", "answer": "2027-04-01", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-36-hi-m1

completed_older_request / hi / sarvam-105b-conversations. Flags: changed_output, raw_status_miss.

Question: VISIT-2136 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?

Source-supported target: `{"answer": "2027-06-06", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-06-06", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "already_applied", "answer": "2027-06-06", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-05-en-m0

missing_replacement / en / sarvam-105b. Flags: factual_miss.

Question: Which replacement date do I currently want for VISIT-2105? Read only; do not change the booking.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `{"action_status": "not_requested", "answer": null, "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-15-te-m0

unresolved_condition / te / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: చెప్పిన షరతు ప్రకారం VISIT-2115 కోసం ఏ తేదీ ఎంచుకోవాలి? మార్చవద్దు.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-06-te-m0

missing_replacement / te / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: VISIT-2106 కోసం ఇప్పుడు నేను కోరుకుంటున్న కొత్త తేదీ ఏది? చెప్పండి మాత్రమే; బుకింగ్ మార్చవద్దు.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `null`; terminal `answer`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": false}`; terminal `answer`; factual=False; raw status=True; strict task=False.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-40-te-m0

conflicting_request_key / te / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: VISIT-2140 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?

Source-supported target: `{"answer": "2027-05-11", "needs_clarification": false}`.
- control: `{"action_status": "not_requested", "answer": "2027-05-11", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "already_applied", "answer": "2027-05-11", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-15-en-m0

unresolved_condition / en / sarvam-105b. Flags: factual_miss, changed_output, raw_status_miss.

Question: From the stated condition, which date should I choose for VISIT-2115? Do not change it.

Source-supported target: `{"answer": null, "needs_clarification": true}`.
- control: `null`; terminal `incomplete`; factual=False; raw status=False; strict task=False.
- candidate: `{"action_status": "not_performed", "answer": null, "needs_clarification": true}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.

## confirm-fresh-40-en-m0

conflicting_request_key / en / sarvam-105b. Flags: changed_output, raw_status_miss.

Question: Handle the supplied change request for VISIT-2140. What date is actually confirmed now?

Source-supported target: `{"answer": "2027-05-11", "needs_clarification": false}`.
- control: `{"action_status": "already_applied", "answer": "2027-05-11", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=False; strict task=True.
- candidate: `{"action_status": "not_performed", "answer": "2027-05-11", "needs_clarification": false}`; terminal `answer`; factual=True; raw status=True; strict task=True.
- Native outcome: missed required write=False; unnecessary attempt=False; forbidden state change=False.
