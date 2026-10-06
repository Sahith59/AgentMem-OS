# English architecture: diagnosis, research and a failed offline candidate

October 6, 2026, America/New_York. Base commit `dc66cda`. Fixed Luna answerer, Terra evaluator and existing extraction configuration. No paid inference or new answers in this work. Sarvam remains parked. Research and engineering below do not establish a new benchmark score.

## The result we actually have

The historical full benchmark is **423/500 = 84.6%, measured once**. The preceding configuration scored 418/500 twice. Terra's 425/500 evaluation of the same saved answers is a different judging scale, not an architecture gain. Reaching 90% from 423 requires 27 net gains; exceeding 90% requires 28. With no regressions, 27 is 35.1% of the 77 historical misses. We cannot promise this is achievable with the fixed models.

The latest paid experiment completed 32 development pairs: **29 correct baseline, 28 candidate, zero gains and one loss**. It failed its frozen gate. Adding original source openings did not improve answers. See [the preserved result](SESSION_OPENING_RESULT_2026-10-06.md). The founder's one-run authorization was consumed there; this session spent nothing on inference.

This is a valid reason to change the engineering approach. Many recent repairs improved request validity, audit labels or source receipts. Those are necessary checks, but they are not the user outcome. Neither passing tests nor retrieving a relevant phrase earns an accuracy claim.

## What is wrong, and what remains uncertain

| Failure | Direct observation | Implication |
| --- | --- | --- |
| Wrong source returned | Session-opening retrieval ranks a session using its best matching user turn but emits its first user turn. The workshop detail is in a later turn. | Preserve the identity of the actual matched turn. |
| Missing related facts | The tenure answer needs two different intervals: company tenure and time until promotion. The paid expansion delivered only the promotion interval. | Matching one topical passage is insufficient for a relation or calculation. Correct arithmetic cannot repair missing operands. |
| Evidence delivered, answer still wrong | The coffee-maker donation/upgrade appeared in the paid candidate packet; Luna still answered four instead of five. | Retrieval alone did not solve that miss. Attention, interpretation and item membership remain competing explanations. |
| Context identity is flattened | The dense benchmark adapter flattens rows; neighbor expansion can operate across synthetic rather than original session boundaries. The assembler receives strings and can truncate joined sections. | A source record, session boundary and delivery receipt should survive retrieval and packing. This audit identifies risk paths, not a measured count of affected answers. |
| Overfitting to small exposed screens | All 32 screen cases are development-exposed; its baseline missed only three. The >=3 net gain gate therefore required fixing all three without losses. | Preserve the failed gate. Design the next population prospectively; do not infer full-set improvement from a few familiar cases. |

The historical 77-miss audit assigned 25 cases all annotated raw turns delivered, 11 sufficiency/judge issues, 2 no linked fact, 10 linked facts absent from the packet and 29 linked facts delivered but raw evidence missing. These are lineage categories, **not 77 independently proven causes or guaranteed recoverable points**. They support separating evidence availability from answer use, not an estimated 90% ceiling.

Earlier structured ledgers and fixed counting heuristics caused substantial regressions. We will not resurrect a count parser that cannot determine whether an event occurred, belongs to the question's category, or is merely advice. Prior span-window experiments also had mixed outcomes; adding neighbors is not a new invention.

## Research that changes the design

Primary sources were reviewed by root and GPT-6 Sol research/audit agents. No entire-internet or reproduction claim is made. These papers support hypotheses, not a transfer of their scores to our system.

| Source | Relevant evidence | Boundary for this project |
| --- | --- | --- |
| [LongMemEval](https://arxiv.org/html/2410.10813v2), [official implementation](https://github.com/xiaowu0162/LongMemEval) | Studies indexing, retrieval and reading separately. Round-level access, preserving original information and a reader that first extracts relevant details can matter. More context is not uniformly helpful across readers. | Test source delivery and answer use separately. Chain-of-note style reading is a hypothesis; JSON formatting alone is not a cure. |
| [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/) | Relevant information can be used less reliably depending on its position in long context. | Appending more text is not automatically better. This does not prove position caused our kitchen miss or establish Luna's behavior. |
| [LeanMem](https://arxiv.org/html/2608.03463v1) | Query-specific access to profiles/events/original records and source pointers. Reported LongMemEval S results differ strongly by reader: 91.8 with GPT-4.1-mini versus 77.4 with Qwen3-8B. | Evidence that architecture and reader interact, not evidence that Luna will reach 90%. We have not reproduced its pipeline; reviewed HTML did not supply the claimed supplementary detail/code for full verification. |
| [EXIT](https://arxiv.org/html/2412.12559v2) | Query-conditioned sentence selection uses document context, then assembles original sentences. | Supports separating selection from source-preserving packing. Its trained classifier and QA datasets differ from this fixed-model conversational setting; pruning can lose evidence. |
| [ECoRAG](https://aclanthology.org/2025.findings-acl.1365.pdf) | Distinguishes strong, weak and distracting evidence and expands context according to sufficiency. | An evidence sufficiency check is useful, but a compressor cannot recover facts absent from its retrieved documents. Extra evaluator/training cost is not free. |
| [FinQA](https://aclanthology.org/2021.emnlp-main.300.pdf) | Separates finding facts from generating and executing operations over them. | For tenure, require source-linked operands and relation labels before calculating. Its financial task and scores do not transfer to conversational event counting. |
| [MemR3](https://arxiv.org/html/2512.20237v1) | A controller tracks evidence gaps and chooses retrieval, reflection or answering. | A bounded gap search is a candidate, with all additional calls charged to a matched-compute comparison. Different dataset/model results do not prove our gain. |

The existing `MultiVectorRetriever` already combines dense and lexical ranks using reciprocal-rank fusion. Hybrid retrieval is therefore not a missing invention. The repair must carry its actual source identities and boundaries into packing; the lexical-only adapter tested here cannot replace that existing recall path.

The useful architecture is: **trusted source snapshot → candidate retrieval → exact source hits → evidence packet → fixed reader → independent evaluation**. Each boundary has a separate failure signal. A source's role, original session, time and identity should survive the full path. A relevant match is a candidate, not certified truth or complete support.

No novelty claim follows from this design. A research contribution would require controlled ablations showing which component improves answers at a stated cost, replicated outcomes and validation beyond repeatedly inspected development questions.

## Implemented now: a reusable source-preserving boundary

`llm/evidence_packet.py` accepts immutable, explicitly scoped source records and ranked source IDs bound to text hashes. It returns the actual hit; it does not substitute a session opening. It filters future sources, packs whole turns under a single character budget, skips an oversized hit without blocking smaller later hits, and admits optional neighbors only from the original session. Distinct events with identical text retain distinct source IDs. Every delivered span has an exact hash/offset receipt.

The packer accepts external source-bound hits. `llm/source_turn_retrieval.py` supplies a minimal lexical adapter for this experiment; its cutoff is applied before fitting TF-IDF. Both user and assistant turns can be candidates. The shared opt-in `ContextAssembler.assemble_source_packet()` calls these same modules. The normal assembler, MCP flow and existing paid packages are unchanged. This is **not a deployed retrieval replacement**.

Completeness remains explicitly uncertified. Anchor-first packing can omit a necessary qualifier. The report distinguishes selection skips from final omissions, because a skipped anchor can later be delivered as a neighbor. Omissions cover considered candidates/neighbors, not every unretrieved source. Caller-supplied scope consistency is not independent tenant authorization. Character caps are not exact tokenizer or compute equivalence.

## Offline result: do not promote

The real shared assembler entrypoint replayed all 32 saved development questions with a 12,000-character ceiling, eight anchors and one neighbor on either side. Runtime inputs contained questions and original date/role/text records, not reference answers or evaluator target phrases. The three known failure checks below were applied only after packing. They are diagnostic inclusion checks, not a semantic-recall metric over 32 questions.

| Known failure | Original-source inclusion in new packet | Diagnosis |
| --- | --- | --- |
| Kitchen donation | Missing | The required turn has zero lexical similarity and never becomes a hit. The older session-opening candidate delivered it. |
| Workshop dates | Delivered | The actual later turn ranks seventh and survives packing. This verifies the identity correction for this case, not a correct new answer. |
| Both tenure intervals | Both missing | Both required turns have zero lexical similarity. More packing space would not make this ranker retrieve them. |

**This compact lexical replacement is rejected for promotion and is not ready for a paid comparison.** It is not acceptable to fix the workshop while silently losing the kitchen evidence. Do not tune keywords, question IDs or gold-directed queries to these three cases.

Independent code review found that the first report listed some delivered neighbors as omitted in 25/32 cases. The report was repaired and the original artifact preserved. A second replay confirms identical packet bytes for all 32 questions and zero delivered/omitted overlaps. The unsuccessful retrieval outcome is unchanged.

See [committed compact evidence](evidence/source-evidence-packet-2026-10-06.json). Full immutable local replays and scripts are in `../codex-memory-2026-09-08/plans/2026-10-06-architecture-reset/`. Input package byte hash: `4a2d1d8e2fcad6241b1806df92ac2883ca4edcfdd0ebad2aa2438ff6801f725b`. The compact evidence records implementation and artifact hashes.

Validation: **129 focused offline tests passed**, including 11 new source-packet tests; Ruff and `git diff --check` passed. Tests use synthetic source boundaries, qualifiers, duplicate events, over-budget turns, future text, forged source hashes and the real assembler entrypoint. The isolated pytest run disabled plugins/conftest and reported one expected unknown `asyncio_mode` configuration warning. It did not collect the legacy paid API test. No new model answers were generated.

## The next bounded task

1. **Replace the losing lexical-only retrieval adapter, not the models.** Preserve the existing dense-plus-lexical fusion and fact candidates with original source identities; add complementary candidates only when justified by a measured coverage gap. Inspect adapters that flatten sessions. Use one shared packet budget and record which source was absent at retrieval versus rejected during packing. First compare source preservation across the existing 500 questions offline; annotations are evaluator-only and cannot drive runtime admission. This is the next engineering task, not completed work.
2. **Test evidence use separately.** Once source delivery is credible, compare the same Luna with the same candidate evidence and a bounded source-cited reading step against an equal-call revision control. Do not add a stronger reader or judge and call that an architecture gain. Record realized tokens and latency as well as call caps. Model inference needs a separately notified, concrete approved package.
3. **Require actual paired answer gains before a full benchmark.** Freeze population, prompt/model versions, costs, treatment/control and success rule before outputs. Include sufficient existing misses and stable successes, disclose development exposure, and audit every gain/loss. Do not reinterpret the previous gate after failure. A passing screen is followed by the full 500, a repeat and a defensible generalization check; it is not itself English closure.

We recommend one coherent candidate through these gates, not indefinite prompt/schema micro-experiments. If it fails to produce answer gains, record the negative result and reconsider the 90% target and remaining budget with the founder. There is no evidence-backed date for 90%, and no authority here to silently close English or switch to Sarvam.
