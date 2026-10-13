# Measuring one observed reporting recovery

The corrected 240-episode Sarvam study exposed one integration failure: the native actor exhausted its output cap, returned no content or tool calls, and the pipeline never reached its final reporter. The original result stays failed.

The diagnostic in `benchmarks.sarvam_handoff_diagnostic` checks eligibility across every saved episode and prepares the two existing no-tools reporters for the sole eligible history. It uses the existing opt-in handoff, the same Sarvam-105B low profile, and unchanged source evidence. It executes no actor or backend action. The original control and candidate reporters are each called once; the candidate is the primary outcome, fixed before execution.

Preparation archives source inputs and hashes separate evaluator bytes without reading labels. Paid execution requires an explicit flag, frozen source/input/budget checks, an exclusive started marker, durable dispatch claim and the existing request/receipt meter. Infrastructure or unknown-usage failure stops further provider dispatch. Started experiments cannot be resumed or retried. Offline audit reads labels only after execution and checks raw responses, request identity, receipt-owned status and accounting.

The two reporters share one English example. They are not two independent cases, a multilingual benchmark, a production reliability estimate or a memory advantage. The handoff adds reporting compute compared with the old early exit; it does not establish an equal-actual-compute gain. Even successful recovery cannot change the original study's failed acceptance gate or reopen its blocked 108-call screen.

The phase allows at most two provider calls, a ₹6.144 conservative full-context reservation and a ₹8 phase ceiling within the standing total ₹6,000 native research budget. The original native call remains counted in earlier accounting. A missing or invalid report remains a failure. No result is selected after seeing the outputs.

Published result and offline replay instructions will live in the dedicated audit directory. Use those instructions for reproduction; do not restart a historical paid manifest. A separate larger source-interpretation comparison must be frozen and announced before any further paid execution.
