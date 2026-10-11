# First Sarvam discovery pilot: stopped before a usable extraction

October 10, 2026 (America/New_York). Frozen source commit `4656fbc`;
package `acf3956980d4acdc4cc8b740aaff6e4dd1f1908f310d98ed247ca2b2ef178b0c`.
This is a runtime configuration result, **not a memory accuracy result**.

The approved comparison planned 228 calls across 38 development renderings and two
repeats. It stopped at the first extraction (`r1-D01-en-extract`), following the
predeclared no-retry rule. The other 227 calls were never dispatched.

| Observation | Result |
| --- | --- |
| HTTP status / returned model | 200 / sarvam-105b |
| Reasoning setting / output limit | low / 4,096 tokens |
| Prompt tokens | 362 |
| Completion tokens | 4,096, including 4,095 reported reasoning tokens |
| Finish reason / final content | length / null |
| Elapsed request time | 17.087 seconds |
| Attempts / accepted extractions / final answers | 1 / 0 / 0 |
| Automatic retries | 0 |
| Known usage at frozen uncached rates | ₹0.31042656 |
| Known usage at current uncached published rates | ₹0.25119 |
| Reservation retained for this attempt | ₹4.13761536 |

No answer pairs exist, so accuracy, improvement and comparative cost-per-correct-answer
are undefined. This does not establish a storage, retrieval, multilingual reasoning,
AgentMem architecture or product-market gap. The key authenticated successfully for
this request, but no balance or service-tier claim follows.

## Diagnosis

The immediate failure is unambiguous: the provider's completion budget was exhausted
and no final content was returned. Sarvam documents that reasoning consumes the same
completion allowance and can leave no visible answer; its suggested remedies include
more output allowance or explicitly disabling reasoning. Our frozen choice of `low`
with 4,096 completion tokens proved unsuitable for this call. This is a limitation
of our configuration that must be corrected before evaluating memory. We do not know
from one receipt why this particular simple extraction used so much reasoning, or
whether every later case would do the same. [Official model limitations](https://docs.sarvam.ai/api/getting-started/models/sarvam-105b).

We preserved the checkpoint and raw response. Replayed validation confirms the exact
request, frozen settings and input hashes; independent arithmetic checks the token
sum, costs and retained reservation. The separate evaluator labels the run
`INCOMPLETE_NO_HEADLINE_SCORE`. It does not turn failure into abstention or a zero
accuracy score. Final-content-only parsing rejected the truncated response as designed.

## Costs and rate change

At preparation, rates were ₹29.28 input / ₹73.20 output per million tokens. The
pre-dispatch check found ₹15 input / ₹60 output per million instead. We kept the
approved higher reservation and frozen accounting unchanged, and calculated the
current-rate amount separately from the returned usage. Neither estimate is a
verified invoice; cached discounts and taxes are not included. [Published pricing](https://docs.sarvam.ai/api/getting-started/pricing).

## Prepared next check — not executed or approved

`proposed-thinking-off-request.json` changes exactly one field of the failed request:
`reasoning_effort` from `"low"` to JSON `null`. Model, endpoint, original source,
extractor prompt, output cap and JSON format remain unchanged. Sarvam documents
explicit null as disabling reasoning. We have not verified that behavior on this
account yet. [Reasoning control](https://docs.sarvam.ai/api/api-guides-tutorials/chat-completion/how-to/adjust-the-models-thinking-level).

Proposed scope: one diagnostic request, no retry, at most ₹4.13761536 reserved using
the same conservative rates. This is a new paid configuration and needs explicit
approval; the original checkpoint will never be rewritten or resumed with changed
settings. Success requires a completed response, usable final JSON, valid source
citations and a faithful extraction from the one source. Report returned reasoning
usage separately. An empty fact list alone is not evidence that useful extraction
works. This would establish only this one extraction's viability, not the full
pilot or answer quality. A later broader comparison remains separately scoped.

Public artifacts include the exact credential-free request, proposed request and
receipt-derived summary with hashes. Raw private API receipts remain locally retained;
provider receipt IDs/account details and credentials are not published. The existing
frozen pilot code and prior evidence are unchanged.
