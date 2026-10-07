# Presence mapping correction before independent certification

The first audit is preserved as audit.json/packets.jsonl/freeze.json. Its mechanical budgets, raw offsets and 7/77miss outcome passed independent recalculation, but the source-presence identity claim did not:7actualshorter turns could be clipped prefixes of longer same-line sources. It is not an independently certified result.

Independent adversarial checks also found fake additional-source frames embedded in future source text or non-source sections. These injection cases are latent in the fixed dataset (no future raw reserved-marker occurrences observed); they still matter for core reliability.

Implementationv2 rejects every proper source-text prefix, parses the complete outer section envelope before trusting an additional-source marker, and screens allscoped original sources for reserved framing/prefix ambiguity. Only time-eligible sources may be matched/ranked/admitted. Allscope safety checks may conservatively disable a certificate when unsafe future text exists; they never admit it as evidence or alter retrieval scores. This supersedes the initial overly broad statement that no future text can affect runtimepresence at all. The baseline itself is frozen and was not presumed temporally/structurally trustworthy without checking.

Same candidate/control policy, thresholds, source/embedding input, budgets and eight anchors; no score-driven retuning. Original output/code retained; audit_v2.py writes separate files. The restored whole-section parser intentionally rejects unknown packet envelopes instead of guessing.68focusedtests (38new,30existing) pass. Final independent review and all500 outcome still pending at this note's creation.
