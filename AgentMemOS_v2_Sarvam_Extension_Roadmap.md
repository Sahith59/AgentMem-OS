# AgentMem-OS v2: Cross-Lingual Federated Memory — Revised Plan

**v3 of this document.** v1 (your original draft) proposed a 5-area extension mapped 1:1 against 5 open Sarvam job postings, centered on fine-tuning a cross-lingual embedding adapter "on a GPU cluster." v2 (Claude, live research) pivoted the centerpiece to a cross-lingual knowledge graph instead, corrected two factual errors, and gave real 2026 cost estimates. v3 (this version) incorporates Codex's independent adversarial review (§10) — which found real problems with v2's timeline and technical scoping — and answers the founder's follow-up questions about Phase 2 criticality, zero-budget sequencing, and the combined path to both the AAMAS deadline and Sarvam.

Author: Sahith Reddy Thummala · Planning partner: Claude, cross-checked by Codex · Status: v3, active plan

---

## 1. Sequencing decision — LOCKED 2026-08-01: Option C

Founder confirmed **Option C — one coherent track, not two**. AgentMem-OS's existing AAMAS 2027 paper push (`LAUNCH_ROADMAP.md`, author registration hard-gated **Sep 17 2026**, full paper **Oct 8 2026**) stays alive, and the Sarvam-facing work below is sequenced to serve both goals rather than compete with them.

**Correction from v2:** v2 claimed Phase 2's paid competitor-baseline evaluation run was "already-AAMAS-required" and should happen first. Codex's review challenged this directly, and it's right — that framing overstated the case. The paper's actual headline claim is MFP (dynamic trust + memory forking), not the competitor comparison. **The MFP-specific evaluation harness — not yet started at all — is more critical-path to AAMAS than the baseline run.** See §5 for the corrected sequencing.

**This still locks sequencing, not spend.** Both the baseline run and any MFP harness runs that cost real API money remain separate financial decisions, gated on their own explicit go-ahead — see §6.

---

## 2. Corrections to the original draft (verified 2026-08-01)

**Chanakya is not an MLOps/serving product.** It's Sarvam's sovereign/defense/government AI platform — "MLOps Engineer, Chanakya" is an internal infra role *under* that program, not a public product you can build "on top of." An air-gapped, fully local-first memory system is still a strong thematic fit for sovereign-AI positioning — just say it accurately: you're demonstrating the kind of engineering Chanakya's mission needs, not integrating with a public Chanakya API (there isn't one).

**"GPU cluster" is the wrong word for what this actually costs**, and using it risks becoming exactly the kind of overclaim your own draft's §8 warns against. See §6 — the real number is $0–40, on a single rented consumer GPU or a free tier, not a cluster.

**Sarvam's API access is real and self-serve** — sign up, get a key, pay-as-you-go, no sales call needed. Published pricing: chat ₹4/1M input tokens, Saaras ASR ₹30/hr (₹45/hr with diarization), Bulbul TTS ₹30/10K chars, Vision ₹0.5/page. Free signup credits exist but the exact amount is inconsistent across their own docs pages (₹100 vs ₹1,000) — budget for real spend regardless, it'll be small (see §6).

**I could not verify** that anyone has been hired at Sarvam (or a comparable company) specifically because of an open-source side project. Codex's review sharpened this into an explicit warning (see §7): don't treat "build this and they'll come to me" as a plan on its own.

---

## 3. The core idea, revised

**New centerpiece: a cross-lingual, temporal, federated knowledge graph**, not a fine-tuned embedding adapter. Three things converge to make this the better bet:

1. **It's a real, unsolved, well-documented gap.** Zep/Graphiti — the most credible funded competitor in this space (arXiv 2501.13956) — has four separate open, unresolved GitHub issues asking for exactly this: [#1141](https://github.com/getzep/graphiti/issues/1141), [#1380](https://github.com/getzep/graphiti/issues/1380), [#312](https://github.com/getzep/graphiti/issues/312), [#434](https://github.com/getzep/graphiti/issues/434). Their own FAQ says it's "on the roadmap," not shipped. Mem0/LangMem/Letta show no evidence of tackling it either.
2. **It's cheap, not a training project.** A January 2026 paper (arXiv 2601.00814) demonstrates the practical version: embed each extracted entity mention with an off-the-shelf multilingual encoder (LaBSE or multilingual-e5, zero fine-tuning), merge nodes across languages by cosine-similarity threshold into `same_as` edges, optionally anchored to a Wikidata QID via a free API call.
3. **You already own most of the substrate.** X-MemoryArch (a sibling project) already has a working bi-temporal knowledge graph, already scoped for reuse into AgentMem-OS but never pulled in.

**Codex's technical warning — take this seriously, don't skip it:** cosine-threshold alias merging is fragile in exactly the ways that would matter in front of a Sarvam engineer specifically. Likely false positives: polysemous product/brand names ("Sarvam," "Saaras," "Bulbul" as product vs. casual reference), common places, contextual entities ("my manager," "the client") that embed similarly across languages but mean different things. Likely false negatives: cross-script transliteration variants, code-mixed Hinglish/Tanglish fragments, honorifics and morphology breaking surface-form matching. **Presented as general-purpose cross-lingual resolution, this will likely get picked apart fast by people who know exactly where it breaks. Presented as a tightly-scoped, evaluated, honestly-labeled feature with visible confidence scores and human-reviewable merges, it's a real, credible, demoable result.** §5 resequences the build to measure this before shipping it live, per Codex's recommendation.

This directly answers your "invest significant time in the knowledge graph" instruction — it's not a side feature next to the KG, the KG *is* the extension.

---

## 4. Architecture

```
                         ┌─────────────────────────┐
                         │   MCP Server Interface   │  ← ALREADY BUILT (Phase 5A,
                         └────────────┬─────────────┘     6 tools, 2 transports —
                                      │                    confirm coverage, don't rebuild)
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
            Speech Ingest       Multimodal Ingest   Text Ingest
            (Saaras ASR +       (Vision pipeline,   (existing path)
             speaker ID)         reuses Nexus work)
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      ▼
                    ┌───────────────────────────────────────┐
                    │  Cross-Lingual Temporal Knowledge Graph │  ← THE NOVEL CORE
                    │  • X-MemoryArch's bi-temporal KG        │     (near-$0, no
                    │    (valid_from/valid_until, supersession)│     GPU cluster —
                    │  • + entity-mention embedding (LaBSE/    │     see §6)
                    │    multilingual-e5, off-the-shelf)       │
                    │  • + cosine-threshold alias merging,     │
                    │    VALIDATED against hand-labeled data   │
                    │    before going live (see §5)            │
                    │  • + optional Wikidata QID anchor         │
                    └────────────────┬──────────────────────┘
                                      ▼
                Existing 4-tier memory core (working cache → episodic → vector → graph)
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  Production Serving (lower priority)  │
                    │  vLLM/llama.cpp, quantized (GGUF/AWQ),│
                    │  Docker, air-gapped mode              │
                    └───────────────────────────────────────┘
```

---

## 5. Combined timeline — one track, both goals (rewritten after Codex's review)

Codex's review of v2 found real problems here: it crammed ~10 weeks of work into a 9.7-week window that also needed to fit paper *writing* time (not budgeted at all), it had a sequencing contradiction (§1 said the baseline eval "happens first," but the phase list put KG work first), and it jumped straight to live entity-merging without first measuring whether it actually works. All three are fixed below.

### Right now — zero budget, nothing here spends money

- **Port X-MemoryArch's Temporal KG** (`graph_builder.py`/`entity_registry.py`/`graph_retrieval.py`) into AgentMem-OS. Codex confirmed this is exactly right for a $0 window — local code, no API spend, no GPU, useful for both tracks.
- **Build a hand-labeled cross-lingual entity-alias eval set** — 20–50 entities across 3–5 languages/scripts, positive pairs (same entity, different language) plus *hard negatives* specifically covering the failure modes in §3 (polysemous Sarvam product names, transliteration near-misses, code-mixed fragments). Run off-the-shelf LaBSE/multilingual-e5 locally (CPU is fine) and measure precision/recall at a few cosine thresholds. **Do not wire live auto-merging into the graph until this step says the threshold actually works** — this is the resequencing Codex specifically recommended, and it's the difference between a credible feature and an embarrassing demo.
- **Scaffold the MFP-specific evaluation harness.** This is the AAMAS paper's actual headline-claim experiment (dynamic trust updates, parent-child forking, confidence decay) and hasn't been started at all, despite the main roadmap flagging it as needing to start "Week 1." Both Codex's read and mine agree: this is more critical-path to the paper than the competitor-baseline run. Building the harness code costs nothing — only running it at scale does.
- **Draft the AAMAS paper sections that don't depend on final numbers** — introduction, related work, system architecture, MFP design. No reason to wait for experiments to start writing these.

### Once budget arrives — no rush to a specific day, just don't let it slip past early September

- **Pilot the paid competitor-baseline eval** (Mem0/Graphiti/Letta/LangMem vs. AgentMem OS) small first — `--n 20-30`, roughly $5-10 per this project's own cost model — before committing to the full run.
- **Run the MFP harness for real**, once built.
- **Wire the cross-lingual alias merging into the live KG**, using the threshold the eval set validated — then build the **X-CRS** metric on real data.
- **Sarvam API integration** (Saaras/Bulbul/Vision) — small real spend, explicit go-ahead required first, same discipline as everywhere else in this project.

### Administrative, not experimental

**Author registration (Sep 17) doesn't require the paper to be finished** — register early in the window regardless of where the numbers stand.

### If time runs short, cut in this order

Production serving's air-gapped mode (stretch goal only) → multimodal ingestion (already proven elsewhere) → Sarvam speech integration (still valuable, but the KG work carries the differentiation alone if it has to). **Never cut: the MFP harness (it's the whole paper) or the Temporal KG + validated cross-lingual layer (it's the whole Sarvam differentiator).**

---

## 6. Cost and GPU breakdown

| Item | Real cost | Notes |
|---|---|---|
| Temporal KG port + cross-lingual entity resolution (hand-labeled eval first) | **$0–15** | Off-the-shelf LaBSE/multilingual-e5, zero fine-tuning for v1. CPU-feasible or free-tier Colab/Kaggle. Optional LoRA fine-tune later: +$0–15 on a rented RTX 4090/L4, 8–20 GPU-hours. **No A100/H100 ever justified here.** |
| MFP evaluation harness runs | Real Anthropic API money once run at scale — amount depends on scenario count, not yet estimated since the harness doesn't exist yet | Build the harness free; get a cost estimate before the first real run, same as everywhere else. |
| Competitor-baseline pilot run | **~$5–10** for a small pilot (`--n 20-30`) | Full run afterward if the pilot numbers look sane — see this repo's own `real_baseline_eval.py --dry-run-cost`. |
| Sarvam Saaras/Bulbul/Vision API calls | **Real, small money — ₹ hundreds to low thousands** for a full demo, scales with usage. Free signup credits partially cover it (exact amount unverified). | **Ask before spending here.** |
| Quantize + serve + benchmark | **$0–10** | GGUF quantization is CPU-only. AWQ needs a GPU briefly for calibration only. |
| **Total realistic budget, done frugally but for real** | **~$25–75 all in** for the engineering side, plus whatever Sarvam API and MFP-harness spend gets separately approved | Nothing here requires institutional compute. |

---

## 7. Outreach — staged plan, starting from zero (updated 2026-08-01)

Codex's sharpest pushback on the earlier version of this plan: "build for ~10 weeks, then launch and hope to be noticed" is too passive given the actual stakes. Confirmed with the founder: no existing LinkedIn/X posting habit, no existing warm contacts at Sarvam or in Indian AI/ML — outreach channels get built from scratch, in parallel with the product, not after it. Founder's own call on pacing: **the Sarvam-facing push leads daily attention; the AAMAS paper track (MFP harness paid run, Table writing) fills the gaps around it**, not the reverse.

Because there's no existing audience to post into, the first artifacts matter more than usual — each one needs to stand on its own with zero prior credibility to borrow from. Sequenced so nothing waits on something slower:

### Stage A — this week (Aug 1–8), $0, starts immediately, nothing blocks anything else

1. **Apply directly to relevant open Sarvam roles today**, via their careers page. Doesn't wait on any of the below — application/interview timelines run weeks regardless, so starting now buys runway rather than losing it. Reference the GitHub repo as "actively building, recent commits" even before the demo is polished; a visibly active repo is itself a signal.
2. **Post the GitHub discussion comparing this project's cross-lingual entity-resolution numbers against Graphiti's own open, unresolved issues** (#1141, #1380, #312, #434) — framed as contribution/collaboration, not a takedown. This can go out *today*: the real precision/recall numbers already exist (`cross_lingual_kg_eval_results.json`), no demo video or README rewrite needed first. This is the single strongest artifact available right now precisely because it's a measured result compared against a named, funded competitor's own admitted gap — not a claim, a fact with a citation.
3. **Create the LinkedIn (and optionally X) presence now if it doesn't exist, and post #1 today** — a short, honest "building this in public, here's why" post linking the GitHub discussion from #2. Starting from zero means the first posts won't get much reach, and that's fine and expected — the goal this week is starting the habit and having a visible history by the time anyone checks the profile, not going viral. Calibrate expectations against this project's own staged trust-signal targets (see `agentmem_os_gtm_positioning.md` in project memory): 1–3 unsolicited mentions is a *Month 1* target, not a Week 1 one.

### Stage B — next 1–2 weeks (Aug 8–22), still $0

4. **Wire the validated τ=0.90 threshold (+ a secondary check for the Chennai/China-class gap) into the live KG**, then build the actual end-to-end demo through the real product: a fact stored in Hindi, correctly recalled when queried in English — the single most visually compelling proof point per §6. This is the technical work that turns Stage A's "here's a number" artifact into "here's it actually working."
5. **Rewrite the README** to lead with that demo, the benchmark table (X-CRS alongside CRS/TES/LCS), and an explicit "built on Sarvam" section — per §6's Phase 6 deliverables, pulled forward because the outreach in Stage C needs somewhere credible to land a click.
6. **Weekly build-log posts continue**, one concrete artifact each time — this week's is the live demo, not just the eval script's numbers.

### Stage C — once the demo + README are real (roughly Aug 22 onward, adjust based on actual pace, not the calendar)

7. **Direct, narrow outreach to specific Sarvam engineers** — likely the Agentic/Orchestration or Sarvam Studio teams first, per the original job-role mapping in §1 of this doc's history. With zero warm network, this is cold outreach, so the message needs to lead with the concrete artifact, not the ask: *"I built a local-first cross-lingual memory demo that solves [the exact Graphiti gap], using your APIs — here's a 90-second demo — would value your technical feedback."* Not "hire me." The demo does the selling; the message just gets it in front of the right person.
8. Continue weekly posts, layer in the Sarvam API integration (Phase 3, needs budget) once funds arrive — voice-in/voice-out closes the loop and gives Stage C's outreach an even stronger follow-up artifact.

**What I can do directly, whenever you're ready for each:** draft the GitHub discussion post (Stage A.2), draft the first build-in-public post (Stage A.3), draft the Sarvam outreach message template (Stage C.7), and do the README rewrite (Stage B.5) myself since that's pure engineering/writing work, not something only you can do. Applying and actually clicking "post" are yours.

---

## 8. What NOT to claim

- Say "embedded with an off-the-shelf multilingual encoder and built a validated cross-lingual entity-resolution layer" — not "trained a cross-lingual foundation model."
- Say "cross-lingual entity resolution, evaluated against hand-labeled hard negatives, verified against Graphiti/Zep's own open feature requests" — not "general-purpose cross-lingual KG" (per §3's fragility warning) and not a vague "better than the competition."
- Say "quantized and served on a single consumer GPU" — not "GPU cluster."
- Say "designed a data curation pipeline mirroring production discipline at a documented, smaller scope" — not "petabyte-scale data engineering."
- Say "Chanakya-aligned sovereign/air-gapped design" — not "integrates with Chanakya" (no public API exists).

---

## 9. My verdict

The KG-centered pivot is still the right call — cheaper, more novel (a documented gap in a funded competitor's own product), more coherent with work already built, more thematically precise for Sarvam. But Codex's review changed the plan materially, and it was right to: the original timeline didn't budget paper-writing time and quietly assumed two ten-week tracks could coexist inside one nine-week window; the entity-merging feature needed an evaluation step before going live, not after; and the whole plan was too passive about actually getting in front of Sarvam. All three are fixed in §5–7. The version you're looking at now is materially more honest about difficulty than the one I handed you a few hours ago, and better for it.

---

## 10. Codex's independent review — received 2026-08-01

Full adversarial review completed after re-login. Its core disagreements with my v2, and what changed as a result:

1. **Timeline was not executable as written** — under 10 weeks of scheduled work inside a 9.7-week window that also needed paper-writing time, which wasn't budgeted at all. *Fixed in §5.*
2. **Sequencing contradiction** — §1 said the baseline run "happens first," §5's phase list put KG work first. *Fixed — §1 corrected, MFP harness now explicitly the priority.*
3. **Cosine-threshold entity merging is technically fragile** and risked embarrassment if shipped live without validation — specific, concrete failure modes identified (polysemous Sarvam product names, transliteration, code-mixing). *Fixed — §5 now requires a hand-labeled precision/recall pass before any live merging.*
4. **The "Phase 2 baseline run is AAMAS-required" framing was unjustified** — the MFP harness is the actual headline claim and more critical-path. *Agreed and corrected in §1/§5 — this directly answers your question below.*
5. **The overall strategy was too passive** — "build for 10 weeks then launch and hope to be noticed" needs real parallel outreach, starting now. *New §7, directly from this finding.*

I didn't find anything in its review I disagreed with — all five points are incorporated above, not just reported.

---

## 11. Direct answers to your questions

**"How critical is Phase 2's baseline evaluation run?"** Less critical than I originally told you, and Codex's independent read agrees: it matters for a complete paper (reviewers will expect a real competitor comparison), but it is *not* the most urgent blocking item, and a delay of a few days changes nothing material. The MFP evaluation harness — which hasn't been started at all — is the more urgent piece, because it's the paper's actual headline claim and it's currently just... not built yet. Good news given your situation: the harness itself costs nothing to build, only running it at scale costs money.

**"What can we start next with no money?"** Three concrete things, all $0, all useful regardless of A/B/C or how the money timing shakes out: (1) port X-MemoryArch's Temporal KG, (2) build the hand-labeled cross-lingual eval set and measure precision/recall before wiring anything live, (3) scaffold the MFP evaluation harness. I can start on any or all of these right now.

**"What's the plan to meet both deadlines?"** §5 above, in full. Short version: build the two free things now, don't spend anything until you're ready, register for AAMAS by Sep 17 regardless of experiment status (it's administrative), run the paid work (baseline pilot, MFP harness, Sarvam APIs) once budget lands, write the paper in parallel rather than after. If time gets tight, the MFP harness and the Temporal KG/cross-lingual work are the two things that never get cut — everything else (speech, vision, serving, air-gapped mode) is real but secondary.

---

## 12. All three $0 items — done, with real results (2026-08-01)

Built and verified all three, in the order: Temporal KG port → MFP evaluation harness → cross-lingual entity-alias precision/recall. Each surfaced real bugs before producing trustworthy numbers — noted here because the bugs themselves are informative, not just the fixes.

**Temporal KG** (`db/knowledge_graph.py`, `db/models.py`): bi-temporal `KnowledgeGraphEdge` schema (relation_type/confidence/valid_from/valid_until/superseded_by) live, typed relation extraction (WORKS_AT/LIVES_AT/STUDIES_AT) reusing `conflict_detector.py`'s own vocabulary, deterministic supersession, `as_of` point-in-time retrieval. `tests/test_temporal_kg.py`, 5/5 passing. Found and fixed 4 real bugs along the way — the most important: a single stated fact ("X works at Y") could never actually surface in retrieval at all, because the subgraph BFS gated traversal on a co-occurrence-frequency threshold that a one-off typed relation could never clear. See the commit message for the other three.

**MFP evaluation harness** (`benchmarks/mfp_eval.py`): exercises the real `AgentTrustNetwork`/`MemoryFederationProtocol`/`AgentNamespaceManager` classes directly — the paper's actual critical-path item, not started before this. Real, sensible, non-degenerate numbers: trust-weighting has a large, honest effect (0.951 vs 0.625 retrieval precision without it); the trust trajectory correctly shows an adversarial agent's trust declining 0.50→0.27 over rounds (the system learns); a correctly-preloaded static-tier baseline scores slightly *above* dynamic trust in this particular scenario, which is a genuine finding worth a paper footnote (static tiers don't pay a learning cost if someone already got the assignment right — the real test dynamic trust wins is when a static tier is *wrong* or an agent's behavior *changes* after assignment, not tested yet). Also fixed `trust_network.py`'s docstring/code mismatch (50/50 vs. the actual 70/30 blend) — the exact thing the main roadmap flagged as needing resolution before Section 3 gets written.

**Cross-lingual entity-alias resolution** (`benchmarks/cross_lingual_kg_eval.py`): hand-labeled 10 real-world entities × English/Hindi/Tamil (30 positive pairs) plus 6 hard negatives specifically targeting Codex's named failure modes (polysemous brand names, similar-sounding-but-different places, same-surname-different-person), embedded with `intfloat/multilingual-e5-small` (local, ~470MB one-time download, $0). **Real result: best F1 at τ=0.90 (precision 0.762, recall 0.533)** — genuinely usable, but with an honest, specific gap: "Chennai" vs. "China" still incorrectly merges even at that threshold (similarity 0.901), and recall at the precision-safe threshold leaves nearly half of genuine cross-lingual matches undetected. **This is the finding to lead with, not hide**: cosine-threshold merging alone gets you most of the way, not all of it — a secondary signal (entity-type agreement, or the Wikidata QID anchoring §3 already mentioned) is a real next step before calling this solved, not a hypothetical one. This number is *more* credible for being imperfect — a suspiciously clean 1.0/1.0 result would be the one to distrust.

**Cost incurred: $0.** New local dependency: `sentence-transformers` (pulls `torch`, CPU-only, ~470MB model download) — added to `requirements.txt`/`pyproject.toml`'s `benchmarks` extra, scoped to this project's own venv, not a global install.

---

## 13. What I'd do next

The three $0 foundations are in place. Real next steps, in rough priority order:
1. **Wire the validated threshold into the live KG** — a `same_as` alias-merge step in `db/knowledge_graph.py` using τ=0.90 plus a secondary check for the Chennai/China-class failure mode (e.g. require entity-type agreement, or hold out any pair whose similarity sits in the ambiguous 0.85-0.92 band for a QID-anchor tiebreak instead of auto-merging).
2. **X-CRS metric** — same fact stored in one language, recall accuracy when queried in another, now measurable end to end since the KG foundation exists.
3. **MFP harness paid run** — needs your go-ahead on spend; the harness itself is done and free to re-run as many more free iterations as useful before that.
4. Whenever you're ready to spend: Phase 2's competitor-baseline pilot, Sarvam API integration (Phase 3), and the MFP harness's real run all become unblocked at once.

Tell me which to start on, or say go and I'll pick the highest-leverage one.
