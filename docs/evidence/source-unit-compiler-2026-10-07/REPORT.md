# Source-unit compiler: implemented and independently verified

October7,2026, America/New_York. Implementationc95f25b. **Mechanical contract complete; semantic selection and answer gains remain unproved.** No paid calls, new answers, embeddings, model changes, corpus edits or runtime-default changes. English remains423/500 (84.6%, once); Terra425/500 grades the same saved answers. English closure remains open; no full500 date is established.

## What works now

The compiler represents a small original quote and its declared context as one question-bound unit. Each quote carries original source identity, speaker, observation timestamp, full-source hash and exact character offsets. Typed edges connect it to declared conditions, negations, corrections, dates, referents or other dependencies. A qualifier can be outside the immediate neighbor window and shared by several quotes without being emitted twice.

The entire caller-supplied ordered snapshot, exact question and cutoff are bound by hashes. Unknown IDs, modified sources/metadata, malformed ranges, overlapping/duplicate spans, invalid roots, cycles and orphan spans fail. A future source, unresolved dependency or oversized unit refuses the whole preview; it cannot emit just the attractive fact after dropping its declared qualification. Adjacent ranges coalesce; omitted ranges partition the rest of every referenced source. Unreferenced history is not represented as reviewed.

JSONL encoding preserves the original text on decoding, including Unicode, literal role labels, fake source frames and line separators. Receipt positions identify encoded JSON strings plus decoded substring offsets. Roles/dates come only from the original snapshot. Edges and unresolved claims stay in the companion report, not as generated historical facts. Preserve preview and report together: report binds the preview hash, exact unit, snapshot, question and cutoff. This is integrity binding, not external source authentication.

The canonical input is llm/source_unit_contract.py. The implementation is llm/source_unit_compiler.py; compile_unit returns an audit preview and report. No answer-packet attachment, model request, automatic proposer or production promotion exists. The shared typed interface is enough for this Python boundary; no duplicate schema/generator system was introduced.

## Three separate checks

1. **103 focused tests pass** across the new compiler and existing source-aware/atomic/source-packet contracts. They exercise adversarial source examples, whole-unit refusal, source/hash/time/question mutation, graph errors, Unicode framing, exact budget boundaries and receipt/omission integrity. Tests preserve distinctions such as planned/completed, advice/action and prior/current; they do not ask a model to classify those distinctions. An intentionally omitted negation still passes structural checks and remains NOT_CERTIFIED: this is a required limitation test, not a semantic success.
2. **Independent code review plus700 adversarial probe cases pass.** These cover500 Unicode/framing/offset/budget cases and200 multi-span coalescence/partition cases. No blocking structural finding. Source hashes are recorded in independent-review.json.
3. **Real-source structural replay passes all500 snapshots and2,008 batches.** All231,575 eligible original turns decode with exact role/date/text and quote receipts.499 exact budget-refusal probes and76 future-source refusal probes pass; one question has no eligible sources. A separate verifier recomputed all snapshot hashes/eligibility/batch coverage and every preview/report hash and receipt. It counted246,750 scoped source occurrences, including15,175 future-by-current-cutoff occurrences, and21 batches containing Unicode line-separator hazards.

The real-source replay deliberately uses whole original turns in batches, not a retrieval selector or compact candidate. Its100,000,000-character stress ceiling merely permits large batches; the largest preview was276,069characters. **This is not evidence that those batches fit the4k/40k answer budget, improve accuracy or preserve all semantic context.** No answer budget changed. No annotation labels, references, historical grades or oracle-capacity witnesses entered this audit. Runtime source input and code were frozen before outputs; audit execution took18.054seconds locally.

Ruff and diff checks pass. Initial lint/import formatting issues were corrected before freezing the structural audit; all103 tests then reran successfully. Existing asyncio_mode warning reflects disabled pytest plugin autoload. No historical artifact was overwritten.

## What remains before a paid comparison

A compiler can enforce dependencies it receives; it cannot discover missing dependencies or prove they are complete. The deliberately incomplete negation example makes this visible. Every successful preview therefore retains semantic_completeness=NOT_CERTIFIED and answer_path_eligible=false. This is not a new paid candidate or a positive answer-score result.

The next unresolved engineering task is a **general label-blind proposer and semantic review method** that finds a useful quote plus all necessary qualifications. Do not route by benchmark IDs, import evaluator labels into runtime, silently clip neighbors, reuse the failed fact-source bridge, or treat the former Luna-v3 role labels as validated. Any model-assisted proposer introduces calls which must be explicitly costed and separately approved; no free/offline guarantee can be inferred from this compiler.

Only a defensible proposer supports a new frozen all500 paired source audit with meaningful gains and controlled losses. The existing10misscase/5net source thresholds remain spending-triage gates, not proof of answers. A reviewed positive offline candidate must precede the concrete fixed-Luna/Terra matched answer comparison, and actual answer gains must precede separately approved full500 and repeat. Do not claim English closed or90% imminent from this engineering step. Founder now emphasizes starting Sarvam quickly; that urgency changes prioritization, not validation or current scope authorization.

Evidence: POLICY.md, freeze.json, audit.py, audit.json, batches.jsonl, validation.json, independent-review.json and stress-verification.json. All500 remain exposed development data; independent review is by an assistant, not independent human annotation. Baseline timestamp-contract differences remain unchanged and need separate resolution before any future paid package.
