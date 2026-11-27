# English event-memory feasibility

This research module measures event mention extraction and event identity on the public ECB+ news corpus. It runs separately from AgentMem OS production and LongMemEval scoring. It implements a reduced writer with two decisions: link a mention to an existing occurrence or create a new occurrence. Corpus labels provide supervisory targets and conditional input spans for the identity experiment. The detector experiment receives ordinary sentence tokens.

Use the Cybulska & Vossen [ECB+ author release](https://github.com/cltl/ecbPlus/tree/3aa071e12fd08fc0705b16bb1841e7a9c5c22420/ECB%2B_LREC2014). Place the original ECB+.zip and ECBplus_coreference_sentences.csv side by side. The license supplied by the authors covers their annotations. Their news texts stay outside this repository.

From this directory use an environment with Python, numpy, scipy, scikit-learn, PyTorch CPU and spaCy with en_core_web_sm. Development execution used sklearn 1.5.2, spaCy 3.8.14, en_core_web_sm 3.8.0, torch 2.13.0 and numpy 2.1.3. The commands below assume DATA points to the downloaded release directory and OUT to a new local output directory.

```sh
python prepare_inputs.py "$DATA" "$OUT"
python detect.py train "$OUT/token.joblib" --data "$DATA"
python detect.py detect "$OUT/token.joblib" < "$OUT/ecb_runtime.jsonl" > "$OUT/token_predictions.jsonl"
python detect_span.py train "$OUT/span.joblib" --data "$DATA"
python detect_span.py detect "$OUT/span.joblib" < "$OUT/ecb_runtime.jsonl" > "$OUT/span_predictions.jsonl"
python score_inputs.py "$OUT" "$OUT/token_predictions.jsonl"
python score_inputs.py "$OUT" "$OUT/span_predictions.jsonl"
python identity.py "$DATA" "$OUT"
python verify_identity.py "$OUT"
```

The identity experiment fits three seeds and three training objectives on topic-disjoint training data. Each original triple is paired with a view that omits its middle observation. Gold event identities are used only for targets. The learned scorer uses the observed words and their local preceding context. The probe checks the partition of the two or three observed mentions and their distinct-event counts. A missed observation does not alter events in the world. Results include marginal final-partition decisions and online decisions which preserve the first link after later input arrives.

Development comprises eight topics. Topic 36 onward is excluded before loading its XML. Triples overlap within documents and 769 of 859 validation triples have three distinct event IDs. Report separate performance for repeats and bootstrap by topic. A whole-document outcome is a different measure from a triple outcome.

On the eight development topics the token detector found 891 of 1,245 exact event mentions and the calibrated span detector found 921. Exact antecedents reachable in the first eight candidates rose from 136 to 140 of 254. These exposed development results include a follow-up motivated by the first detector failures. In the matched conditional identity comparison, online absolute-state supervision scored 802 of 859 at all three seeds. Adding paired state-change supervision scored 800, 802 and 802. This does not meet the prespecified advancement criterion. Temporal field revisions, unknown references and ordinary conversation QA need separate evidence.
