# Focused competitor and paper review

October6,2026. Primary docs/code reviewed by root and GPT-6 Sol researcher. Prepared while the local embedding/audit job runs; this is not an outcome claim or another evaluated candidate. No hosted competitor calls, paid inference, installations of competing services, or reproduced leaderboard claims.

| System / primary source | Mechanism relevant to our gap | Transfer boundary |
| --- | --- | --- |
| [Hindsight recall](https://github.com/vectorize-io/hindsight/blob/main/skills/hindsight-docs/references/developer/api/recall.md) | Semantic, keyword, graph and temporal retrieval; fusion/reranking; original fact/chunk pointers. Optional `prefer_observations` suppresses raw facts already supported by returned observations and backfills slots. | Source-linked redundancy control is useful. Consolidation can lag. Its fact budget excludes metadata and can return an oversized first fact; our hard total budget must include attribution and never silently overflow. Temporal ranking is not our strict future-source filter. |
| [Graphiti search recipes](https://github.com/getzep/graphiti/blob/main/graphiti_core/search/search_config_recipes.py), [edge maintenance](https://github.com/getzep/graphiti/blob/main/graphiti_core/utils/maintenance/edge_operations.py) | Hybrid search, episode-linked facts and temporal validity; configurable fusion/diversity/reranking. | Transfer original-episode links and qualifier access. A new graph plus ingestion-time extraction would change several variables and cost; it is not an automatic fixed-model improvement. |
| [Mem0 Platform graph](https://github.com/mem0ai/mem0/blob/main/docs/platform/features/graph-memory.mdx), [current architecture](https://github.com/mem0ai/mem0/blob/main/docs/core-concepts/how-it-works.mdx) | Entity-linked context can strengthen retrieval beyond one similarity score. Platform and OSS capabilities differ. | Shared entity mentions do not establish a typed relationship or completed event. Do not infer an event count or tenure relation merely from connectivity, or conflate current platform behavior with the older paper. |
| [Letta memory architecture](https://github.com/letta-ai/skills/blob/main/letta/agent-development/references/memory-architecture.md) | Bounded always-visible memory, separate conversation history and archival retrieval. | Treat memory budget as an explicit resource. Archive search is an agent action; adopting its loop changes calls and latency and requires a matched-compute control. |

These are documented mechanisms, not comparable scores on our Luna/extraction/Terra setup. No competitor's90%claim proves our target achievable. We reviewed mechanisms and their limits, not every competitor release or the entire internet.

## Papers that constrain the next hypothesis

- [LongMemEval](https://arxiv.org/html/2410.10813v2): indexing, retrieval and reading need separate assessment. Recovering an original turn and using it correctly are distinct outcomes. Query-focused reading has precedent; rigid output formatting alone is not a solution.
- [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/): more context and favorable retrieval coverage do not guarantee reliable answer use. Its older-model findings are not a causal diagnosis of a specific Luna miss.
- [EXIT](https://arxiv.org/html/2412.12559v2): context-aware query-conditioned source selection can reduce distracting text. Its trained classifier and datasets differ; pruning must be audited for lost qualifiers.
- [ECoRAG](https://aclanthology.org/2025.findings-acl.1365.pdf): evidence quality/sufficiency can guide how much context to admit. Its additional learned components are not free replacements under our fixed-model budget.
- [MemR3](https://arxiv.org/html/2512.20237v1): a bounded evidence-gap controller is precedented. Any planning/retrieval/reading calls must be charged and compared with an equal-call control; gains on another benchmark do not transfer automatically.

## Falsifiable response if the current addition policy fails

Do not switch to a larger model or a graph rewrite and label that the success of this intervention. First use the observed ledger to determine the missing stage:

1. If the needed source never ranks in the candidate pool, packing cannot repair it. Inspect candidate recall and query formulation; preserve original links and temporal scope.
2. If useful candidates rank but duplicates or whole-turn size consume the budget, test source-aware allocation. Certify baseline source presence only where an exact scoped source/date/role/body mapping is unambiguous; retain ambiguous or distinct dated events. A semantic paraphrase match must not silently merge events.
3. If source support is delivered but a qualification is missing, admit the linked qualifier as part of a source bundle, or report that the complete bundle does not fit. Relevance alone does not establish completeness.
4. If the complete source is delivered and Luna still fails, stop adding retrieval variants for that case. A fixed-Luna reading intervention needs its own actual-answer comparison and compute control.

The possible research contribution is a measured source-aware selection and qualification contract, including explicit failure/abstention behavior. It is not a novelty claim today: retrieval fusion, source pointers, deduplication and evidence selection already exist. A publishable claim would require ablations, qualified matched-model/cost comparisons, replication, and validation beyond the repeatedly inspected500-question development set.

Do not implement output-driven parameter sweeps after a negative audit. Preserve that result, explain the measured mechanism, and freeze the next materially different candidate before generating answers. No paid proposal follows from this review alone.
