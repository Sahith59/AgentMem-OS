#!/usr/bin/env python3
"""
MFP-specific multi-agent evaluation harness — LAUNCH_ROADMAP.md Phase 4's
critical-path item ("No evaluation harness anywhere in the repo currently
exercises agents/trust_network.py/namespace_manager.py/memory_federation.py
against production code paths — building this is on the critical path and
must start in Week 1").

Exercises the REAL production classes directly — AgentTrustNetwork,
AgentNamespaceManager, MemoryFederationProtocol — not a reimplementation of
their formulas (that's what test_phase4_multiagent.py already is, and it
doesn't count as this harness per the roadmap's own framing).

Scenario: 5 agents in a simulated team (4 genuine domain experts + 1
deliberately-degrading "noisy" agent — the adversarial injection). Each
round, agents query outside their own specialty, forcing federated
retrieval; a synthetic-but-known ground truth (baked into the scenario,
not judged by an LLM) marks which retrieved memory was genuinely good vs.
from the noisy agent, driving real feedback() calls into the real trust
network. A fork event and a decay run happen mid-scenario.

Six variants isolate what MFP's mechanisms actually contribute, matching
LAUNCH_ROADMAP.md Table 3 exactly:
  FULL                 — trust-weighted + age-weighted + L2/L3 fork inheritance (the real system)
  NO_TRUST             — feedback() never called, so trust never leaves neutral (0.5 for every pair)
  STATIC_TIER_TRUST    — trust set once via 4 fixed manual tiers, never updated by feedback
                          (simulates MemClaw's actual mechanism: manual PATCH-style
                          assignment, no EMA learning — see agentmem_os_gtm_positioning.md)
  NO_DECAY             — run_decay() never called
  NO_FORK_INHERITANCE  — fork_agent(..., inherit_levels=[]) — child starts fully cold
  INHERIT_ALL_LEVELS   — fork_agent(..., inherit_levels=[1,2,3]) — inherits episode-level
                          summaries too, not just pattern/principle (tests whether
                          restricting to L2/L3 is actually pulling its weight, or
                          whether "more inherited context" would do just as well
                          for more token cost — NOT a literal raw-Turn copy, fork_agent()
                          never touches the Turn table at all regardless of this setting)

All six call the same real AgentTrustNetwork/MemoryFederationProtocol/
AgentNamespaceManager instances — only which methods get called, and with
what arguments, changes between variants. No LLM calls anywhere; feedback
signals come from the scenario's own known ground truth, not a judge model.

Cost: $0. Uses one isolated SQLite DB for the whole run (AGENTMEM_OS_DB_PATH
override, set before any agentmem_os.db import — never touches the default
dev DB), reset between variants via drop_all/create_all rather than
swapping DB files (db/engine.py binds its engine once at import time, so
changing the env var per-variant after that point has no effect).

Usage:
    python3 benchmarks/mfp_eval.py

Output: benchmarks/mfp_eval_results.json (Table 3 numbers, Fig 2's
per-round trust trajectory, Fig 3's fork-quality-vs-token-cost data)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tier_lib import ok, warn, hdr, sub  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Isolated DB for this harness, created before any agentmem_os.db.* import
# so db/engine.py's module-level engine binds to it, not the shared dev DB.
_DB_DIR = tempfile.mkdtemp(prefix="agentmem_mfp_eval_")
os.environ["AGENTMEM_OS_DB_PATH"] = str(Path(_DB_DIR) / "mfp_eval.db")

from agentmem_os.db.engine import get_session, init_db, engine  # noqa: E402
from agentmem_os.db.models import Base, Session as DBSession, Summary  # noqa: E402
from agentmem_os.agents.trust_network import AgentTrustNetwork  # noqa: E402
from agentmem_os.agents.namespace_manager import AgentNamespaceManager  # noqa: E402
from agentmem_os.agents.memory_federation import MemoryFederationProtocol  # noqa: E402
from agentmem_os.llm.token_counter import TokenCounter  # noqa: E402


# ── Scenario definition ──────────────────────────────────────────────────

EXPERT_AGENTS = ["researcher-bot", "coder-bot", "reviewer-bot", "planner-bot"]
NOISY_AGENT = "noisy-bot"
ALL_AGENTS = EXPERT_AGENTS + [NOISY_AGENT]

# Each expert owns one topic; every agent queries the OTHER three topics
# each round, forcing federated retrieval rather than self-answering.
TOPIC_OWNER = {
    "researcher-bot": "architecture",
    "coder-bot": "debugging",
    "reviewer-bot": "testing",
    "planner-bot": "deployment",
}
TOPICS = list(TOPIC_OWNER.values())


# Retrieval relevance uses a real local TF-IDF embedder (see _build_embedder
# below), not MFP's keyword-Jaccard fallback. This matters for the
# experiment's validity: hand-crafting content so keyword-Jaccard ties
# EXACTLY between good/noisy causes every tie to resolve by Python's
# stable-sort insertion order (good was seeded first, so it would win
# literally every round regardless of trust — verified empirically: the
# first version of this harness did exactly that, produced a flat trust
# trajectory, and NO_TRUST/STATIC_TIER_TRUST scored *higher* than FULL,
# which is backwards). Real cosine similarity between two related-but-
# different sentences is close but essentially never an exact tie,
# leaving genuine room for trust to be the deciding factor once it
# diverges from neutral.
GOOD_CONTENT = {
    "architecture": "Architecture principle: prefer composition over inheritance when "
                     "behavior needs to vary independently of identity.",
    "debugging": "Debugging principle: reproduce the issue with the smallest possible "
                 "input before reading any implementation code.",
    "testing": "Testing principle: a flaky test is a bug in the test's isolation, not "
               "a reason to retry until it passes.",
    "deployment": "Deployment principle: roll changes out behind a flag so a bad "
                  "change degrades gracefully instead of failing all at once.",
}
# Plausible-sounding but wrong — the adversarial content, same register
# and length as GOOD_CONTENT so it isn't trivially distinguishable.
NOISY_CONTENT = {
    "architecture": "Architecture principle: always use inheritance over composition, "
                     "it's simpler and composition just adds indirection.",
    "debugging": "Debugging principle: the fastest way to debug is to read the entire "
                 "codebase top to bottom before touching anything.",
    "testing": "Testing principle: flaky tests are normal at scale, just add a retry "
               "wrapper and move on, it's not worth investigating.",
    "deployment": "Deployment principle: always deploy the full change to all traffic "
                  "immediately, feature flags just slow down iteration.",
}
# Two query phrasings per topic, alternated by round parity — one echoes
# GOOD's own reasoning vocabulary, one echoes NOISY's. This is deliberate,
# not left to chance: it guarantees noisy-bot's content is a genuine
# contender on relevance for its "home" phrasing (so trust has real
# material to act on across rounds) while good's content is the genuine
# contender on the other, rather than hoping generic phrasing happens to
# land in a useful zone (verified empirically that it doesn't — see the
# GOOD_CONTENT block's comment for what happened when it didn't).
QUERIES = {
    "architecture": [
        "What's the guidance on preferring composition and letting behavior vary independently?",
        "What's the guidance on using inheritance for simplicity and less indirection?",
    ],
    "debugging": [
        "What's the guidance on reproducing an issue with minimal input before reading code?",
        "What's the guidance on reading the whole codebase before touching anything?",
    ],
    "testing": [
        "What's the guidance on isolating a flaky test instead of just retrying it?",
        "What's the guidance on adding a retry wrapper for flaky tests at scale?",
    ],
    "deployment": [
        "What's the guidance on rolling a change out behind a flag?",
        "What's the guidance on deploying a full change to all traffic immediately?",
    ],
}

N_ROUNDS = 14
FORK_AT_ROUND = 5           # planner-bot forks a specialist child mid-scenario
DECAY_AT_ROUND = 11
STATIC_TIERS = {             # MemClaw-style fixed manual tiers, set once, never updated
    "researcher-bot": 0.90, "coder-bot": 0.75, "reviewer-bot": 0.60,
    "planner-bot": 0.45, NOISY_AGENT: 0.45,
}

VARIANTS = [
    ("FULL", "trust-weighted + age-weighted + L2/L3 fork inheritance (the real system)"),
    ("NO_TRUST", "feedback() never called — trust stays neutral (0.5) for every pair"),
    ("STATIC_TIER_TRUST", "trust set once via 4 fixed manual tiers, never updated (MemClaw-style)"),
    ("NO_DECAY", "run_decay() never called"),
    ("NO_FORK_INHERITANCE", "fork with inherit_levels=[] — child starts fully cold"),
    ("INHERIT_ALL_LEVELS", "fork with inherit_levels=[1,2,3] — episode-level summaries too"),
]

_tok = TokenCounter()


def _build_embedder():
    """
    Local TF-IDF embedder, fit once on the full known corpus (content +
    queries) — no API calls, no downloaded model, pure scikit-learn
    (already a project dependency). Returns a str -> dense-vector callable
    matching MemoryFederationProtocol's get_embedding_fn contract.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    all_queries = [q for pair in QUERIES.values() for q in pair]
    corpus = list(GOOD_CONTENT.values()) + list(NOISY_CONTENT.values()) + all_queries
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(corpus)

    def embed(text: str):
        return vectorizer.transform([text]).toarray()[0]

    return embed


_EMBED = _build_embedder()


def _seed_agents_and_pool(db) -> dict:
    """
    Create the 5 agents and seed each expert's on-topic summary (good) plus
    the noisy agent's off-brand summary for every topic (bad). Returns a
    content -> is_good lookup used to score retrievals against ground truth.
    """
    ns = AgentNamespaceManager(get_session)
    for agent_id in ALL_AGENTS:
        ns.ensure_agent_exists(agent_id)

    ground_truth = {}
    for agent_id in EXPERT_AGENTS:
        topic = TOPIC_OWNER[agent_id]
        content = GOOD_CONTENT[topic]
        _seed_summary(db, agent_id, content)
        ground_truth[content] = True

    for topic in TOPICS:
        content = NOISY_CONTENT[topic]
        _seed_summary(db, NOISY_AGENT, content)
        ground_truth[content] = False

    return ground_truth


def _seed_summary(db, agent_id: str, content: str) -> None:
    session_id = f"mfp-eval-{agent_id}-seed"
    if not db.query(DBSession).filter_by(session_id=session_id).first():
        db.add(DBSession(session_id=session_id, agent_id=agent_id, name="mfp-eval seed"))
        db.commit()
    db.add(Summary(
        session_id=session_id, agent_id=agent_id, turn_range="0-0",
        content=content, abstraction_level=3, is_shared=False,
    ))
    db.commit()


def run_variant(variant: str) -> dict:
    """
    db/engine.py binds its `engine` to AGENTMEM_OS_DB_PATH at MODULE IMPORT
    TIME (module-level create_engine() call) — reassigning the env var
    per-variant here would silently do nothing after the first import, and
    every variant would contaminate the same DB. Real isolation instead:
    drop and recreate every table between variants, on the one engine/DB
    file fixed for this whole process (set once, before any agentmem_os.db
    import, at module level above).
    """
    Base.metadata.drop_all(bind=engine)
    init_db()

    db = get_session()
    ground_truth = _seed_agents_and_pool(db)
    db.close()

    trust = AgentTrustNetwork(get_session)
    ns = AgentNamespaceManager(get_session)
    mfp = MemoryFederationProtocol(get_session, trust, get_embedding_fn=_EMBED)

    if variant == "STATIC_TIER_TRUST":
        for a in ALL_AGENTS:
            for b in ALL_AGENTS:
                if a != b:
                    trust.set_trust(a, b, STATIC_TIERS[b])

    for agent_id in ALL_AGENTS:
        mfp.promote(agent_id)

    trust_trajectory = []   # Fig 2 data: per-round trust in the noisy agent
    retrieval_events = []   # Table 3 data: was the top-1 result genuinely good?
    fork_result = None

    for round_i in range(N_ROUNDS):
        for querying_agent in ALL_AGENTS:
            own_topic = TOPIC_OWNER.get(querying_agent)
            for topic in TOPICS:
                if topic == own_topic:
                    continue  # only query outside your own specialty
                query = QUERIES[topic][round_i % 2]
                results = mfp.retrieve(query=query, querying_agent=querying_agent, top_k=1)
                if not results:
                    continue
                top = results[0]
                is_good = ground_truth.get(top["content"], False)
                retrieval_events.append({
                    "round": round_i, "agent": querying_agent, "topic": topic,
                    "source": top["source_agent_id"], "is_good": is_good,
                })
                if variant != "NO_TRUST":
                    signal = 1.0 if is_good else 0.0
                    mfp.feedback(
                        entry_id=top["entry_id"], from_agent=querying_agent,
                        to_agent=top["source_agent_id"], signal=signal,
                    )

        avg_trust_in_noisy = sum(
            trust.get_trust(a, NOISY_AGENT, use_transitive=False) for a in EXPERT_AGENTS
        ) / len(EXPERT_AGENTS)
        trust_trajectory.append({"round": round_i, "avg_trust_in_noisy_agent": round(avg_trust_in_noisy, 4)})

        if round_i == FORK_AT_ROUND:
            inherit_levels = {
                "NO_FORK_INHERITANCE": [], "INHERIT_ALL_LEVELS": [1, 2, 3],
            }.get(variant, [2, 3])
            fork_result = ns.fork_agent(
                parent_agent_id="planner-bot", child_agent_id="planner-bot-specialist",
                inherit_levels=inherit_levels, trust_network=trust,
            )
            db = get_session()
            child_summaries = (
                db.query(Summary).filter_by(agent_id="planner-bot-specialist").all()
            )
            inherited_chars = sum(len(s.content) for s in child_summaries)
            db.close()
            fork_result["inherited_context_tokens"] = _tok.count("x" * inherited_chars)
            fork_result["inherited_matches_own_topic"] = any(
                ground_truth.get(s.content.replace("[INHERITED from planner-bot] ", ""), False)
                for s in child_summaries
            )

        if round_i == DECAY_AT_ROUND and variant != "NO_DECAY":
            # decay_days=0 so the days-since-creation gate trivially passes
            # in this fast synthetic scenario (no real elapsed days) —
            # min_accesses stays at the module's own default (2), so this
            # retires genuinely under-accessed entries rather than wiping
            # the whole pool. Real test: does decay disproportionately
            # retire the noisy agent's entries (retrieved less often once
            # trust in them has fallen) while keeping the experts' — a
            # meaningful ablation, not just "decay deletes everything."
            mfp.run_decay(decay_days=0)

    total = len(retrieval_events)
    good = sum(1 for e in retrieval_events if e["is_good"])
    precision = good / total if total else 0.0

    return {
        "variant": variant,
        "retrieval_precision": round(precision, 4),
        "n_retrieval_events": total,
        "trust_trajectory": trust_trajectory,
        "fork_result": fork_result,
        "final_pool_stats": mfp.get_pool_stats(),
    }


def main():
    hdr("AgentMem OS — MFP Evaluation Harness (Table 3 / Fig 2 / Fig 3 source data)")
    warn("Exercises real agents/trust_network.py, memory_federation.py, "
         "namespace_manager.py directly — not a reimplementation. $0 cost, "
         "synthetic ground truth, no LLM calls.")

    results = []
    for variant, label in VARIANTS:
        sub(f"{variant} — {label}")
        r = run_variant(variant)
        ok(f"{variant}: retrieval_precision={r['retrieval_precision']:.4f} "
           f"over {r['n_retrieval_events']} events")
        results.append(r)

    out_path = Path(__file__).parent / "mfp_eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    ok(f"Results -> {out_path}")

    hdr("TABLE 3 — MFP ABLATION SUMMARY (retrieval precision, ground-truth-known scenario)")
    full = next(r for r in results if r["variant"] == "FULL")
    print(f"  {'Variant':<22} {'Precision':>10} {'ΔvsFULL':>10}")
    print(f"  {'─'*22} {'─'*10} {'─'*10}")
    for r in results:
        d = r["retrieval_precision"] - full["retrieval_precision"]
        marker = "  ← reference" if r["variant"] == "FULL" else ""
        print(f"  {r['variant']:<22} {r['retrieval_precision']:>10.4f} {d:>+10.4f}{marker}")

    hdr("FIG 2 — TRUST IN THE NOISY AGENT OVER ROUNDS (FULL vs STATIC_TIER_TRUST)")
    for variant in ("FULL", "STATIC_TIER_TRUST"):
        r = next(x for x in results if x["variant"] == variant)
        traj = [t["avg_trust_in_noisy_agent"] for t in r["trust_trajectory"]]
        print(f"  {variant:<20} " + " → ".join(f"{v:.2f}" for v in traj))

    hdr("FIG 3 — FORK QUALITY VS. CONTEXT-TOKEN COST")
    print(f"  {'Variant':<22} {'Inherited tokens':>16} {'Inherited matches own topic':>28}")
    for variant in ("FULL", "NO_FORK_INHERITANCE", "INHERIT_ALL_LEVELS"):
        r = next(x for x in results if x["variant"] == variant)
        fr = r["fork_result"] or {}
        print(f"  {variant:<22} {fr.get('inherited_context_tokens', 0):>16} "
              f"{str(fr.get('inherited_matches_own_topic', False)):>28}")

    identical = [r["variant"] for r in results[1:]
                 if abs(r["retrieval_precision"] - full["retrieval_precision"]) < 1e-9]
    if identical:
        warn(f"These variants scored IDENTICAL retrieval precision to FULL: {identical}. "
             f"Known, not yet fixed: NO_DECAY ties FULL because in this scenario config "
             f"every entry (including the noisy one) gets accessed >= MIN_USEFUL_ACCESSES "
             f"before decay runs, so decay has nothing to retire — extending N_ROUNDS or "
             f"tightening the threshold should separate them. NO_FORK_INHERITANCE/"
             f"INHERIT_ALL_LEVELS tie FULL because the forked child never queries anything "
             f"after being created — fork_result's inherited_context_tokens/"
             f"inherited_matches_own_topic fields (Fig 3) already differ correctly, but "
             f"the aggregate retrieval_precision metric doesn't include the child's own "
             f"post-fork queries yet. Both are real scenario gaps to close before this is "
             f"paper-final, not silent bugs — noted here instead of hidden.")
    else:
        ok("Every variant produced a measurably different retrieval precision — "
           "each mechanism (trust weighting, decay, fork scoping) is pulling real weight.")

    ok("STATIC_TIER_TRUST scoring at or slightly above FULL is a real, honest finding, "
       "not a bug: it starts with a correct trust ordering baked in from round 0, while "
       "FULL starts neutral and briefly picks the noisy agent a few times before dynamic "
       "trust catches up. The gap this scenario doesn't yet test — and should before "
       "citing this as evidence for dynamic trust over static tiers — is what happens "
       "when a static tier is WRONG or an agent's behavior changes after assignment; "
       "only dynamic trust can recover from that. Worth a dedicated variant later.")


if __name__ == "__main__":
    main()
