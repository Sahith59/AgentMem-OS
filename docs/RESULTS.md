# Every Number We Have Ever Produced, In One Table

Every row below comes from a committed JSON artifact in `benchmarks/`
that records its own configuration. Nothing here is projected, rounded
up, or quietly superseded. Columns that matter for comparability:

- **Harness**: `truncated` = before fix F-17 our loader silently cut
  42.8% of conversation turns at 800 chars (numbers are valid for that
  harness and labelled, never deleted); `full` = the honest harness,
  benchmark delivered verbatim.
- **Write-time extraction**: the model that reads conversations and
  writes the fact tier. All rows to date: **llama3.1-8B, local** — the
  upgrade to a frontier extractor is the next measured arc.
- **Memory backbone**: what the answer packet is primarily built from.
- **Answerer / Judge**: the judge is frozen (GPT-4o, the benchmark's
  official per-type prompts) in every row. We never tune the judge.

## Headline numbers (current)

| # | What | Harness | n | Backbone | Answerer | Score | Mean ctx tokens | Artifact |
|---|---|---|---|---|---|---|---|---|
| H1 | **Number of record (Luna column), pooled 3 runs** | full | 3×500 | verbatim+facts+profile | gpt-5.6-luna | **80.0% ± 0.5** (399/403/398) | 8,536 | `_500q_40k_fullturns_luna{,_r2,_r3}` |
| H2 | GPT-4o column, run 1 | full | 500 | verbatim+facts+profile | gpt-4o | 73.2% (366) | 8,561 | `_500q_40k_fullturns_r1` |
| H3 | **Measured ceiling** (oracle: perfect evidence, no haystack) | full | 150 | gold sessions only | gpt-5.6-luna | **89.3%** (134) | 4,557 | `_oracle150_luna_fullturns` |
| H4 | No memory system at all (benchmark authors' full-context figure) | — | 500 | entire haystack ~115k tokens | gpt-4o | 60.2% | ~115,000 | LongMemEval paper |

The one-variable lesson of H1 vs H2: same memory, same packets, same
judge — the answerer alone is worth +6.6 points. Any vendor comparison
that does not disclose the answerer is comparing answerers, not
memories. The lesson of H3: this exam's roof for a live system is the
high 80s; published numbers in the 90s are not this protocol.

## Main-line history (chronological)

| Era | Harness | n | Answerer | Score | Mean ctx tokens | What changed | Artifact |
|---|---|---|---|---|---|---|---|
| Jul 2026 | truncated | 150 | gpt-4o-mini | 63.3% (95) | n/r | first clean reproducible baseline | `_TRUE150` |
| Aug 2026 | truncated | 150 | gpt-4o | 72.0% (108) | n/r | answerer upgrade (+8.7, p=0.011) | `_TRUE150_4o` |
| Aug 2026 | truncated | 150 | gpt-4o | 73.3% (110) | n/r | fact-tier budget retune | `_cov35` |
| Aug 2026 | truncated | 3×150 | gpt-4o | 76.9% ± 1.0 (114/115/117) | 5,698 | dense retrieval | `_dense{,_repeat,_run3}` |
| Aug 2026 | truncated | 3×150 | gpt-4o | 79.3% ± 1.2 (120/120/117) | ~9,800 | context 24k→40k chars | `_ctx40k{,_r2,_r3}` |
| Aug 2026 | truncated | 500 | gpt-4o | 74.8% (374) | 8,600 | full question set (150-sample had overestimated; disclosed ±6.8 CI) | `_500q_40k_r1` |
| Aug 2026 | **full** | 500 | gpt-4o | 73.2% (366) | 8,561 | honest harness + F-19 packing (forensics in FAILURES.md) | `_500q_40k_fullturns_r1` |
| Aug 2026 | **full** | 3×500 | gpt-5.6-luna | **80.0% ± 0.5** | 8,536 | answerer column; gains concentrate in counting/temporal | `_500q_40k_fullturns_luna*` |

## Category smokes on the honest harness (validation runs, small n)

| Category | n | Answerer | Score | Context | Artifact |
|---|---|---|---|---|---|
| assistant-recall, pre-packing-fix | 56 | gpt-4o | 94.6% (53) | validates F-17 evidence restoration (was 78.6% truncated) | `_ssa56_fullturns` |
| assistant-recall, F-19 adaptive packing | 56 | gpt-4o | **96.4% (54)** | validates F-19 | `_ssa56_f19adaptive` |
| knowledge-update, flat snippets | 78 | gpt-4o | 80.8% (63) | the F-18 regression, kept on record | `_ku78_fullturns` |
| knowledge-update, tuned snippets | 78 | gpt-4o | 87.2% (68) | recovery to baseline | `_ku78_fullturns_snip800` |
| preference | 30 | gpt-4o | 53.3% (16) | judge-rubric-bound category (evidence coverage measured 0.87) | `_pref30_fullturns` |

## Probes that failed and stay published (truncated era, n=150 unless noted)

| Probe | Score | Verdict | Artifact |
|---|---|---|---|
| Structured answerer v1 (code computes answers) | 57.3% | refuted — computed path loses to free reasoning | `_structured` |
| Structured answerer v2 (code computes arithmetic only) | 65.3% | refuted | `_structured_v2` |
| Aggregation routing as default | 77.3% | failed pre-registered bar; ships opt-in | `_aggroute` |
| ctx0 span width | 75.3% | trade, not gain | `_ctx0` |
| Breadth-then-depth d=4 | 76.7% | trade, not gain | `_btd4` |
| Extracted-facts-primary memory (llama3.1-8B extraction) | 48.7% | 8B-extracted notes cannot carry the packet alone — the measured reason the backbone is verbatim, and the motivation for the frontier-extractor arc | `_gate_d_full150` |

## Reading rules

1. Never quote a single run where a pooled mean exists.
2. Never compare rows across different answerers, judges, splits, or
   harnesses without saying so.
3. Anything in this file can be reproduced from the repo; the eval
   command for every row is in the artifact's own config block.
