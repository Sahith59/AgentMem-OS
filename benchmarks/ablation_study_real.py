#!/usr/bin/env python3
"""
Real-code ablation — exercises the actual ContextAssembler /
ConversationStore / EntityKnowledgeGraph / ProceduralMemory, not a
hand-rolled simulation of them.

Every other ablation/head-to-head script in this directory
(ablation_study.py, phase1_multi_run.py, phase2_long_horizon.py,
phase3_baselines.py, head_to_head.py) is a self-contained proxy that never
imports agentmem_os internals — see their own module docstrings ("zero
dependency on memnai.* internals"). That means the ablation study has never
actually ablated AgentMem OS; it's ablated a toy model of it. This script
closes that specific gap for the semantic / knowledge-graph / procedural
tiers, using the `disable` parameter added to
ContextAssembler.assemble() for exactly this purpose.

Scope, deliberately narrower than the simulation scripts:
  - Measures CRS only (context-relevance / assembled-context quality).
    Does NOT measure LCS (needs a live LLM answering questions — already
    covered by tests/test_e2e_claude.py against real Claude calls) or TES
    (needs real Sleep Consolidation summaries, which have their own known
    macOS/Python 3.13 BLAS thread-safety issue — see LAUNCH_ROADMAP.md
    Phase 3 Bug 12 — out of scope for this script).
  - No NO_SLEEP variant: ContextAssembler.assemble() has no
    sleep/consolidation toggle of its own — consolidation changes what's
    INSIDE the episodic/recent turns (raw turns vs. a compressed summary),
    not which section of assemble() runs — so there's no clean "disable
    sleep" equivalent via the same `disable` mechanism as the other tiers.
  - Ingests turns via a synchronous, deterministic path rather than
    save_turn()'s normal daemon-thread KG ingestion, so results don't
    depend on background-thread timing. Production save_turn() behavior
    is unchanged by this script.
  - Swaps ChromaDB for a TF-IDF semantic retriever (same swap
    tests/test_e2e_claude.py makes), so this runs with zero external
    dependencies — no Ollama, no Chroma server, no API key.

Cost: $0 — no LLM calls at all, pure retrieval/assembly.

Usage:
    python3 benchmarks/ablation_study_real.py

Output: benchmarks/ablation_real_results.json
"""
import sys
import json
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tier_lib import ok, warn, hdr, sub, CONVERSATION, PROBE_RECALLS, crs_from_probe_contexts  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agentmem_os.storage.store import ConversationStore  # noqa: E402
from agentmem_os.llm.context_assembler import ContextAssembler  # noqa: E402
from agentmem_os.llm.procedural_memory import ProceduralMemory  # noqa: E402
from agentmem_os.db.engine import get_session as get_db  # noqa: E402
from real_code_utils import install_tfidf_chroma  # noqa: E402

install_tfidf_chroma(ContextAssembler)


VARIANTS = [
    ("FULL",        "Full real system (all tiers)",         frozenset()),
    ("NO_SEMANTIC", "w/o Semantic Retrieval (real Tier 3)",  frozenset({"semantic"})),
    ("NO_KG",       "w/o Entity Knowledge Graph (real)",     frozenset({"global"})),
    ("NO_PROC",     "w/o Procedural Memory (real)",          frozenset({"procedural"})),
    ("RECENT_ONLY", "All 3 optional tiers disabled",         frozenset({"semantic", "global", "procedural"})),
]

_ACK_REPLY = "Understood."  # canned — this ablation is retrieval-only, no LLM calls


def ingest_conversation(store: ConversationStore, session_id: str) -> None:
    """
    Persist CONVERSATION into a fresh session synchronously — real SQLite
    writes, real KG entity/edge extraction, real pattern mining, no
    daemon-thread timing dependency.
    """
    from agentmem_os.db.models import Turn

    store.get_or_create_session(session_id, name="ablation-real")

    for role, content in [
        pair for msg in CONVERSATION for pair in
        (("user", msg), ("assistant", _ACK_REPLY))
    ]:
        tokens = store.token_counter.count(content)
        turn = Turn(session_id=session_id, role=role, content=content, token_count=tokens)
        store.db.add(turn)
        store.db.commit()
        try:
            store._ingest_kg(session_id, None, content)
        except Exception as e:
            warn(f"KG ingestion failed for a turn: {e}")

    # Procedural pattern mining is a separate step from ingestion in the
    # real system (mine_patterns() vs. save_turn()) — run it explicitly so
    # the NO_PROC variant has something real to disable.
    pm = ProceduralMemory(get_db)
    n_patterns = pm.mine_patterns(session_id)
    if n_patterns == 0:
        warn("mine_patterns() found 0 patterns for this conversation — "
             "NO_PROC will legitimately show no CRS difference from FULL.")


def run_variant(assembler: ContextAssembler, session_id: str, disable: frozenset) -> dict:
    probe_contexts = {}
    for i in sorted(PROBE_RECALLS):
        query = CONVERSATION[i]
        ctx = assembler.assemble(session_id, query, disable=disable)
        probe_contexts[i] = (query, ctx)
    crs = crs_from_probe_contexts(probe_contexts)
    return {"crs": crs}


def main():
    hdr("AgentMem OS — Real-Code Ablation (CRS only, $0 cost)")
    store = ConversationStore()
    assembler = ContextAssembler()

    results = []
    for vname, label, disable in VARIANTS:
        sub(f"{vname} — {label}")
        session_id = f"ablation-real-{vname.lower()}-{uuid.uuid4().hex[:8]}"
        ingest_conversation(store, session_id)
        r = run_variant(assembler, session_id, disable)
        ok(f"{vname}: CRS={r['crs']:.4f}")
        results.append({"variant": vname, "label": label,
                         "disable": sorted(disable), "metrics": {"CRS": r["crs"]}})

    out_path = Path(__file__).parent / "ablation_real_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    ok(f"Results → {out_path}")

    hdr("REAL-CODE ABLATION SUMMARY")
    full = next(r for r in results if r["variant"] == "FULL")
    print(f"  {'Variant':<15} {'CRS':>8} {'ΔvsFULL':>10}")
    print(f"  {'─'*15} {'─'*8} {'─'*10}")
    for r in results:
        d = r["metrics"]["CRS"] - full["metrics"]["CRS"]
        marker = "  ← reference" if r["variant"] == "FULL" else ""
        print(f"  {r['variant']:<15} {r['metrics']['CRS']:>8.4f} {d:>+10.4f}{marker}")

    identical = [r["variant"] for r in results[1:]
                 if abs(r["metrics"]["CRS"] - full["metrics"]["CRS"]) < 1e-9]
    if identical:
        warn(f"These variants scored IDENTICAL CRS to FULL: {identical} — "
             f"check whether the disabled tier actually had content to "
             f"contribute for this conversation (e.g. KG/procedural tiers "
             f"need enough repeated signal within the 25-turn script).")
    else:
        ok("Every real-code variant produced a measurably different CRS — "
           "the tier-blindness bug from the simulation scripts is fixed here too.")


if __name__ == "__main__":
    main()
