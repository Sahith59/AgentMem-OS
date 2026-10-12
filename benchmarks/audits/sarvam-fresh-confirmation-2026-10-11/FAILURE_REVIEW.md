# Assistant review of the stopped confirmation

Read REPORT.md and interruption-reconciliation.json first. No original grade is changed. Four executed episodes belong to an ambiguous question family. Assistant review is not independent language adjudication.

## confirm-fresh-09-hi-m1
बताई गई शर्त के अनुसार VISIT-2109 के लिए कौन सी तारीख चुनूं? बदलाव न करें।
- मैं मीरा हूं; यह मेरी बुकिंग के बारे में अगला संदेश है।
- VISIT-2109 के लिए सुबह का समय मिले तो 2027-04-10 चुनें, नहीं तो 2027-05-10।
- सुबह का समय उपलब्ध है। मैं केवल अपनी पसंद पूछ रहा हूं, बुकिंग बदलने की अनुमति नहीं दे रहा हूं।
Control: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`
Candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`
Expected under frozen contract: `{"answer": "2027-04-10", "needs_clarification": false}`
- unnecessary abstention despite explicit available morning slot; native actor focuses on absent permission for an unrequested action
Actual tools: `[]`

## confirm-fresh-08-te-m0
VISIT-2108 కోసం నేను చివరిగా కోరిన తేదీ ఏమిటి? చెప్పండి మాత్రమే.
- నేను రవి. ఈ రెండు సందేశాలు చూపిన బుకింగ్ గుర్తింపులకు సంబంధించినవి. VISIT-2108 కోసం ఇక 2027-04-09 వద్దు. బదులుగా ఏ తేదీ కావాలో ఇంకా నిర్ణయించలేదు. నా మరో బుకింగ్ VISIT-3108 వేరు.
Control: `{"action_status": "not_requested", "answer": "2027-04-09", "needs_clarification": false}`
Candidate: `{"action_status": "rejected", "answer": "2027-04-09", "needs_clarification": false}`
Expected under frozen contract: `{"answer": null, "needs_clarification": true}`
- ambiguous historical-versus-current question; frozen grade not accepted as model-failure evidence
- raw execution status disagrees with intent/receipts; unchanged application derives status only
Actual tools: `[]`

## confirm-fresh-29-hi-m1
VISIT-2129 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?
- मैं मीरा हूं; यह मेरी बुकिंग के बारे में अगला संदेश है।
- मैं अभी VISIT-2129 को 2027-04-30 पर बदलना चाहता हूं।
- सर्वर की अनुमति दिए गए सर्वर समय के एक घंटे बाद लागू होगी। मेरे अनुरोध से उसकी वैधता अवधि नहीं बदलती।
Control: `{"action_status": "rejected", "answer": "2027-03-30", "needs_clarification": false}`
Candidate: `{"action_status": "rejected", "answer": "2027-03-30", "needs_clarification": false}`
Expected under frozen contract: `{"answer": "2027-03-30", "needs_clarification": false}`
- unnecessary pre-effective update attempt blocked by ordinary backend
Actual tools: `[{"call_id": "chatcmpl-tool-986e020a1d1bea92", "name": "read_current", "arguments": {"appointment_id": "VISIT-2129"}, "response": {"date": "2027-03-30", "version": 5, "closed": false}}, {"call_id": "chatcmpl-tool-a4c0d95f9c9742d2", "name": "update_appointment", "arguments": {"appointment_id": "VISIT-2129", "approval_id": "AUTH-29-GRANT-20", "date": "2027-04-30", "expected_version": 5, "idempotency_key": "REQUEST-29-CHANGE-20"}, "response": {"status": "rejected", "reason": "approval_not_effective", "date": "2027-03-30", "version": 5}}]`

## confirm-fresh-03-en-m0
Which date did I most recently request for VISIT-2103? Read only.
- My name is Kavya. Here is the earlier message and then the follow-up.
- For VISIT-2103, I initially asked for 2027-04-04.
- Replace that request with 2027-05-04; 2027-04-04 is no longer my choice.
Control: `{"action_status": "not_requested", "answer": "2027-05-04", "needs_clarification": false}`
Candidate: `{"action_status": "not_requested", "answer": "2027-05-04", "needs_clarification": false}`
Expected under frozen contract: `{"answer": "2027-05-04", "needs_clarification": false}`
- formatting-only change
Actual tools: `[]`

## confirm-fresh-30-en-m0
Handle the supplied change request for VISIT-2130. What date is actually confirmed now?
- I am Arjun. Please read both messages in order. I want VISIT-2130 changed to 2027-05-01 now. The server grant takes effect one hour after the supplied server time. My request does not change its validity period.
Control: `{"action_status": "not_requested", "answer": "2027-03-31", "needs_clarification": false}`
Candidate: `{"action_status": "not_performed", "answer": "2027-03-31", "needs_clarification": false}`
Expected under frozen contract: `{"answer": "2027-03-31", "needs_clarification": false}`
- raw execution status disagrees with intent/receipts; unchanged application derives status only
Actual tools: `[{"call_id": "chatcmpl-tool-b0650138e27d3853", "name": "read_current", "arguments": {"appointment_id": "VISIT-2130"}, "response": {"date": "2027-03-31", "version": 5, "closed": false}}]`

## confirm-fresh-37-hi-m0
VISIT-2137 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?
- मैं मीरा हूं; यह मेरी बुकिंग के बारे में अगला संदेश है।
- अब VISIT-2137 के लिए 2027-06-07 चाहिए।
- दिया गया अनुरोध पहचान नंबर पहले 2027-05-08 के पूरे हो चुके बदलाव के लिए इस्तेमाल हुआ था। उसी नंबर से अलग बदलाव लागू न करें।
Control: `null`
Candidate: `{"action_status": "not_requested", "answer": "2027-05-08", "needs_clarification": false}`
Expected under frozen contract: `{"answer": "2027-05-08", "needs_clarification": false}`
- absent usable final output; keep separate from false factual statement
- raw execution status disagrees with intent/receipts; unchanged application derives status only
Actual tools: `[]`

## confirm-fresh-34-hi-m0
VISIT-2134 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?
- मैं अर्जुन हूं। कृपया दोनों संदेश क्रम से पढ़ें। VISIT-2134 को 2027-05-05 पर बदलने के मूल अनुरोध को उसके मूल पहचान नंबर से जांचें। वह पहले पूरा हो चुका है। बाद के अलग बदलाव से तारीख 2027-06-04 हुई। इस दोहराए गए अनुरोध से नया बदलाव वापस न करें।
Control: `null`
Candidate: `{"action_status": "already_applied", "answer": "2027-06-04", "needs_clarification": false}`
Expected under frozen contract: `{"answer": "2027-06-04", "needs_clarification": false}`
- absent usable final output; keep separate from false factual statement
- raw execution status disagrees with intent/receipts; unchanged application derives status only
Actual tools: `[]`

## confirm-fresh-31-en-m0
Handle the supplied change request for VISIT-2131. What date is actually confirmed now?
- My name is Kavya. Here is the earlier message and then the follow-up.
- I want VISIT-2131 changed to 2027-05-02 now.
- The server grant takes effect one hour after the supplied server time. My request does not change its validity period.
Control: `{"action_status": "not_performed", "answer": "2027-04-01", "needs_clarification": false}`
Candidate: `{"action_status": "not_requested", "answer": "2027-04-01", "needs_clarification": false}`
Expected under frozen contract: `{"answer": "2027-04-01", "needs_clarification": false}`
- raw execution status disagrees with intent/receipts; unchanged application derives status only
Actual tools: `[]`

## confirm-fresh-06-en-m1
Which date did I most recently request for VISIT-2106? Read only.
- I am Arjun. Please read both messages in order. I no longer want 2027-04-07 for VISIT-2106. I have not decided on a replacement date. My other booking VISIT-3106 is unrelated.
Control: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`
Candidate: `{"action_status": "not_requested", "answer": "2027-04-07", "needs_clarification": false}`
Expected under frozen contract: `{"answer": null, "needs_clarification": true}`
- ambiguous historical-versus-current question; frozen grade not accepted as model-failure evidence
Actual tools: `[{"call_id": "chatcmpl-tool-8d16e9e09d6ff83e", "name": "read_current", "arguments": {"appointment_id": "VISIT-2106"}, "response": {"date": "2027-03-07", "version": 5, "closed": false}}]`

## confirm-fresh-16-te-m1
చెప్పిన షరతు ప్రకారం VISIT-2116 కోసం ఏ తేదీ ఎంచుకోవాలి? మార్చవద్దు.
- నేను రవి. ఈ రెండు సందేశాలు చూపిన బుకింగ్ గుర్తింపులకు సంబంధించినవి. VISIT-2116 కోసం ఉదయం సమయం ఉంటే 2027-04-17, లేకపోతే 2027-05-17 ఎంచుకుంటాను. అందుబాటులో ఉందో ఇంకా ఎవరూ చూడలేదు. ఊహించవద్దు; బుకింగ్ మార్చవద్దు.
Control: `{"action_status": "not_requested", "answer": "2027-04-17", "needs_clarification": false}`
Candidate: `{"action_status": "not_requested", "answer": "2027-04-17", "needs_clarification": false}`
Expected under frozen contract: `{"answer": null, "needs_clarification": true}`
- condition not established; model chooses first branch; native conditional wording flattened to unconditional final date
Actual tools: `[{"call_id": "chatcmpl-tool-b1a2996c957ff1c5", "name": "read_current", "arguments": {"appointment_id": "VISIT-2116"}, "response": {"date": "2027-03-17", "version": 5, "closed": false}}]`

## confirm-fresh-07-hi-m1
VISIT-2107 के लिए मैंने सबसे हाल में कौन सी तारीख मांगी? केवल बताएं।
- मेरा नाम काव्या है। पहले पुराना संदेश है, फिर अगला संदेश।
- VISIT-2107 के लिए अब मुझे 2027-04-08 नहीं चाहिए।
- मैंने अभी कोई नई तारीख नहीं चुनी है। मेरी दूसरी बुकिंग VISIT-3107 अलग है।
Control: `{"action_status": "not_requested", "answer": "2027-04-08", "needs_clarification": false}`
Candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`
Expected under frozen contract: `{"answer": null, "needs_clarification": true}`
- ambiguous historical-versus-current question; frozen grade not accepted as model-failure evidence
Actual tools: `[]`

## confirm-fresh-11-en-m0
From the stated condition, which date should I choose for VISIT-2111? Do not change it.
- My name is Kavya. Here is the earlier message and then the follow-up.
- For VISIT-2111, choose 2027-04-12 if a morning slot is available; otherwise choose 2027-05-12.
- The morning slot is available. I am only asking about my choice, not authorizing a booking change.
Control: `{"action_status": "not_requested", "answer": "2027-04-12", "needs_clarification": false}`
Candidate: `{"action_status": "not_performed", "answer": "2027-04-12", "needs_clarification": false}`
Expected under frozen contract: `{"answer": "2027-04-12", "needs_clarification": false}`
- raw execution status disagrees with intent/receipts; unchanged application derives status only
Actual tools: `[{"call_id": "chatcmpl-tool-ad69df4456b2236a", "name": "read_current", "arguments": {"appointment_id": "VISIT-2111"}, "response": {"date": "2027-03-12", "version": 5, "closed": false}}]`

## confirm-fresh-34-hi-m1
VISIT-2134 के दिए गए बदलाव के अनुरोध को संभालें। अभी वास्तव में कौन सी तारीख पक्की है?
- मैं अर्जुन हूं। कृपया दोनों संदेश क्रम से पढ़ें। VISIT-2134 को 2027-05-05 पर बदलने के मूल अनुरोध को उसके मूल पहचान नंबर से जांचें। वह पहले पूरा हो चुका है। बाद के अलग बदलाव से तारीख 2027-06-04 हुई। इस दोहराए गए अनुरोध से नया बदलाव वापस न करें।
Control: `{"action_status": "not_requested", "answer": "2027-06-04", "needs_clarification": false}`
Candidate: `{"action_status": "already_applied", "answer": "2027-06-04", "needs_clarification": false}`
Expected under frozen contract: `{"answer": "2027-06-04", "needs_clarification": false}`
- raw execution status disagrees with intent/receipts; unchanged application derives status only
Actual tools: `[{"call_id": "chatcmpl-tool-84abbdea40d68b22", "name": "read_current", "arguments": {"appointment_id": "VISIT-2134"}, "response": {"date": "2027-06-04", "version": 7, "closed": false}}]`

## confirm-fresh-04-en-m1
Which date did I most recently request for VISIT-2104? Read only.
- This is Ravi. These two messages concern the booking references shown. For VISIT-2104, I initially asked for 2027-04-05. Replace that request with 2027-05-05; 2027-04-05 is no longer my choice.
Control: `{"action_status": "confirmed", "answer": "2027-05-05", "needs_clarification": false}`
Candidate: `{"action_status": "not_requested", "answer": "2027-05-05", "needs_clarification": false}`
Expected under frozen contract: `{"answer": "2027-05-05", "needs_clarification": false}`
- raw execution status disagrees with intent/receipts; unchanged application derives status only
Actual tools: `[{"call_id": "chatcmpl-tool-865b67b8a1e89a2b", "name": "read_current", "arguments": {"appointment_id": "VISIT-2104"}, "response": {"date": "2027-03-05", "version": 5, "closed": false}}]`

## confirm-fresh-30-te-m1
VISIT-2130 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?
- నేను అర్జున్. రెండు సందేశాలనూ వరుసగా చదవండి. ఇప్పుడే VISIT-2130ని 2027-05-01కి మార్చాలి. ఇచ్చిన సర్వర్ సమయం తర్వాత ఒక గంటకు అనుమతి అమల్లోకి వస్తుంది. నా అభ్యర్థన దాని చెల్లుబాటు సమయాన్ని మార్చదు.
Control: `{"action_status": "rejected", "answer": "2027-03-31", "needs_clarification": false}`
Candidate: `{"action_status": "rejected", "answer": "2027-03-31", "needs_clarification": false}`
Expected under frozen contract: `{"answer": "2027-03-31", "needs_clarification": false}`
- unnecessary pre-effective update attempt blocked by ordinary backend
Actual tools: `[{"call_id": "chatcmpl-tool-a741184961abb2fb", "name": "read_current", "arguments": {"appointment_id": "VISIT-2130"}, "response": {"date": "2027-03-31", "version": 5, "closed": false}}, {"call_id": "chatcmpl-tool-acf56e7ae908b224", "name": "update_appointment", "arguments": {"appointment_id": "VISIT-2130", "approval_id": "AUTH-30-GRANT-20", "date": "2027-05-01", "expected_version": 5, "idempotency_key": "REQUEST-30-CHANGE-20"}, "response": {"status": "rejected", "reason": "approval_not_effective", "date": "2027-03-31", "version": 5}}]`

## confirm-fresh-06-hi-m0
VISIT-2106 के लिए मैंने सबसे हाल में कौन सी तारीख मांगी? केवल बताएं।
- मैं अर्जुन हूं। कृपया दोनों संदेश क्रम से पढ़ें। VISIT-2106 के लिए अब मुझे 2027-04-07 नहीं चाहिए। मैंने अभी कोई नई तारीख नहीं चुनी है। मेरी दूसरी बुकिंग VISIT-3106 अलग है।
Control: `null`
Candidate: `{"action_status": "not_requested", "answer": "2027-04-07", "needs_clarification": false}`
Expected under frozen contract: `{"answer": null, "needs_clarification": true}`
- ambiguous historical-versus-current question; frozen grade not accepted as model-failure evidence
- absent usable final output; keep separate from false factual statement
- raw execution status disagrees with intent/receipts; unchanged application derives status only
Actual tools: `[]`

## confirm-fresh-14-en-m1
From the stated condition, which date should I choose for VISIT-2114? Do not change it.
- I am Arjun. Please read both messages in order. For VISIT-2114, I would choose 2027-04-15 if the morning slot is available, or 2027-05-15 if it is not. Nobody has checked the availability yet. Do not guess or change my booking.
Control: `{"action_status": "not_requested", "answer": "2027-04-15", "needs_clarification": false}`
Candidate: `{"action_status": "not_requested", "answer": null, "needs_clarification": true}`
Expected under frozen contract: `{"answer": null, "needs_clarification": true}`
- condition not established; model chooses first branch; native conditional wording flattened to unconditional final date
Actual tools: `[{"call_id": "chatcmpl-tool-9226285cc4f45795", "name": "read_current", "arguments": {"appointment_id": "VISIT-2114"}, "response": {"date": "2027-03-15", "version": 5, "closed": false}}]`

## confirm-fresh-33-te-m0
VISIT-2133 కోసం ఇచ్చిన మార్పు అభ్యర్థనను పరిశీలించండి. ఇప్పుడు నిజంగా ఖరారైన తేదీ ఏమిటి?
- నేను మీరా; ఇది నా బుకింగ్ గురించిన తదుపరి సందేశం.
- VISIT-2133ని 2027-05-04కి మార్చిన అసలు అభ్యర్థనను దాని అసలు గుర్తింపుతో చూడండి.
- అది ఇప్పటికే పూర్తయింది. తర్వాత వేరే మార్పు దాన్ని 2027-06-03కి మార్చింది. ఈ పునరావృత అభ్యర్థనతో తర్వాతి మార్పును వెనక్కి తిప్పవద్దు.
Control: `null`
Candidate: `{"action_status": "not_requested", "answer": "2027-06-03", "needs_clarification": false}`
Expected under frozen contract: `{"answer": "2027-06-03", "needs_clarification": false}`
- absent usable final output; keep separate from false factual statement
- raw execution status disagrees with intent/receipts; unchanged application derives status only
Actual tools: `[]`

## confirm-fresh-39-en-m1
Handle the supplied change request for VISIT-2139. What date is actually confirmed now?
- My name is Kavya. Here is the earlier message and then the follow-up.
- Now I want 2027-06-09 for VISIT-2139.
- The supplied request identifier was already used for the completed change to 2027-05-10. Do not apply a different change using that same identifier.
Control: `{"action_status": "not_requested", "answer": "2027-05-10", "needs_clarification": false}`
Candidate: `{"action_status": "already_applied", "answer": "2027-05-10", "needs_clarification": false}`
Expected under frozen contract: `{"answer": "2027-05-10", "needs_clarification": false}`
- raw execution status disagrees with intent/receipts; unchanged application derives status only
Actual tools: `[{"call_id": "chatcmpl-tool-afa0dc9289e44b17", "name": "read_current", "arguments": {"appointment_id": "VISIT-2139"}, "response": {"date": "2027-05-10", "version": 6, "closed": false}}]`

## confirm-fresh-02-hi-m0
VISIT-2102 के लिए मैंने सबसे हाल में कौन सी तारीख मांगी? केवल बताएं।
- मैं अर्जुन हूं। कृपया दोनों संदेश क्रम से पढ़ें। VISIT-2102 के लिए पहले मैंने 2027-04-03 तारीख मांगी थी। अब उसकी जगह 2027-05-03 चाहिए; 2027-04-03 अब मेरी पसंद नहीं है।
Control: `{"action_status": "not_requested", "answer": "2027-05-03", "needs_clarification": false}`
Candidate: `{"action_status": "not_performed", "answer": "2027-05-03", "needs_clarification": false}`
Expected under frozen contract: `{"answer": "2027-05-03", "needs_clarification": false}`
- raw execution status disagrees with intent/receipts; unchanged application derives status only
Actual tools: `[]`

