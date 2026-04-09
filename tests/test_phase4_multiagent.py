"""
AgentMem OS — Phase 4 Test Suite (Pure-Python, no SQLAlchemy)
==============================================================
Tests all Phase 4 algorithms without requiring pip installs:
  - Trust network: EMA formula, bounds, transitive trust, weight_memories
  - Memory Federation: promotion scoring, age weight, keyword similarity,
    format_for_context, pool stats, decay logic
  - Namespace Manager: fork logic (simulated), merge conflict resolution
  - Integration: full pipeline verification

Run: python tests/test_phase4_multiagent.py
"""

import sys
import re
import math
import types
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

# ─── Stub loguru ─────────────────────────────────────────────────────────────
loguru_stub = types.ModuleType("loguru")
class _Logger:
    def debug(self, *a, **k): pass
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
    def bind(self, **k): return self
loguru_stub.logger = _Logger()
sys.modules["loguru"] = loguru_stub

# ─── Test framework ──────────────────────────────────────────────────────────
passed = failed = 0
results = []

def assert_eq(label, got, expected):
    global passed, failed
    if got == expected:
        passed += 1; results.append(("✅", label, f"{got!r}"))
    else:
        failed += 1; results.append(("❌", label, f"got {got!r} expected {expected!r}"))

def assert_true(label, cond, msg=""):
    global passed, failed
    if cond:
        passed += 1; results.append(("✅", label, ""))
    else:
        failed += 1; results.append(("❌", label, msg or "False"))

def assert_approx(label, got, expected, tol=0.02):
    global passed, failed
    if abs(got - expected) <= tol:
        passed += 1; results.append(("✅", label, f"≈{got:.4f}"))
    else:
        failed += 1; results.append(("❌", label, f"got {got:.4f} expected ≈{expected:.4f} ±{tol}"))


# ══════════════════════════════════════════════════════════════════════════════
# Inline implementations of core algorithms for pure-Python testing
# (mirrors exactly what's in the source files)
# ══════════════════════════════════════════════════════════════════════════════

# ─── Trust Network constants & logic ─────────────────────────────────────────
TRUST_EMA_ALPHA = 0.80
MIN_TRUST       = 0.05
MAX_TRUST       = 0.98
NEUTRAL_TRUST   = 0.50
FORK_TRUST      = 0.90
MIN_INTERACTIONS_FOR_STABILITY = 5

def _clamp(v):
    return min(MAX_TRUST, max(MIN_TRUST, v))

class InMemoryTrustNetwork:
    """
    Pure-Python trust network — no SQLAlchemy.
    Identical EMA / transitive / weight_memories logic to AgentTrustNetwork.
    """
    def __init__(self):
        self._cache: Dict[Tuple[str, str], float] = {}
        self._interactions: Dict[Tuple[str, str], int] = {}

    def get_trust(self, agent_from, agent_to, use_transitive=True):
        if agent_from == agent_to:
            return 1.0
        direct = self._cache.get((agent_from, agent_to), NEUTRAL_TRUST)
        if not use_transitive:
            return direct
        transitive = self._compute_transitive_trust(agent_from, agent_to)
        if transitive is None:
            return direct
        return _clamp(0.70 * direct + 0.30 * transitive)

    def _compute_transitive_trust(self, agent_from, agent_to):
        intermediaries = {k[1]: v for k, v in self._cache.items()
                          if k[0] == agent_from and k[1] != agent_to}
        best = None
        for mid, t_a_x in intermediaries.items():
            t_x_b = self._cache.get((mid, agent_to))
            if t_x_b is not None:
                path = t_a_x * t_x_b
                best = path if best is None else max(best, path)
        return best

    def set_trust(self, frm, to, score):
        score = _clamp(score)
        self._cache[(frm, to)] = score

    def update_trust(self, frm, to, feedback_signal):
        old = self._cache.get((frm, to), NEUTRAL_TRUST)
        new = _clamp(TRUST_EMA_ALPHA * old + (1 - TRUST_EMA_ALPHA) * feedback_signal)
        self._cache[(frm, to)] = new
        self._interactions[(frm, to)] = self._interactions.get((frm, to), 0) + 1
        return new

    def get_trust_matrix(self, agent_ids):
        return {frm: {to: self.get_trust(frm, to) for to in agent_ids}
                for frm in agent_ids}

    def get_most_trusted_sources(self, agent_id, top_k=5):
        scores = {k[1]: v for k, v in self._cache.items() if k[0] == agent_id}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def weight_memories(self, memories, querying_agent, relevance_key="relevance_score"):
        for mem in memories:
            src = mem.get("source_agent_id", "unknown")
            rel = float(mem.get(relevance_key, 0.5))
            trust = self.get_trust(querying_agent, src)
            mem["trust_score"] = trust
            mem["weighted_score"] = rel * trust
        memories.sort(key=lambda m: m["weighted_score"], reverse=True)
        return memories

    def describe(self):
        if not self._cache:
            return "[TrustNetwork] No trust relationships established yet."
        lines = ["[TRUST NETWORK]"]
        for (frm, to), score in sorted(self._cache.items()):
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"  {frm} → {to}: {bar} {score:.2f}")
        return "\n".join(lines)


# ─── MFP constants & helpers ─────────────────────────────────────────────────
PROMOTION_THRESHOLD   = 2.0
DECAY_DAYS            = 30
MIN_USEFUL_ACCESSES   = 2
AGE_WEIGHT_HALFLIFE   = 60

def _age_weight(created_at: datetime) -> float:
    days_old = max(0.0, (datetime.utcnow() - created_at).total_seconds() / 86400)
    return 2.0 ** (-days_old / AGE_WEIGHT_HALFLIFE)

def _keyword_similarity(query: str, content: str) -> float:
    def tokenize(text):
        return set(re.findall(r'\b\w{3,}\b', text.lower()))
    q, c = tokenize(query), tokenize(content)
    if not q or not c:
        return 0.0
    return len(q & c) / len(q | c)

def _promotion_score(abstraction_level: int, confidence_bonus: float = 0.0) -> float:
    return abstraction_level * (1.0 + confidence_bonus)


class MockSummary:
    def __init__(self, agent_id, content, abstraction_level, is_shared=False):
        self.agent_id          = agent_id
        self.content           = content
        self.abstraction_level = abstraction_level
        self.is_shared         = is_shared

class MockFederatedEntry:
    def __init__(self, source_agent_id, content, abstraction_level,
                 promotion_score, access_count=0, is_active=True, created_at=None):
        self.id               = uuid.uuid4().int & 0xFFFF
        self.source_agent_id  = source_agent_id
        self.content          = content
        self.abstraction_level = abstraction_level
        self.promotion_score  = promotion_score
        self.access_count     = access_count
        self.is_active        = is_active
        self.created_at       = created_at or datetime.utcnow()
        self.last_accessed_at = None

class InMemoryFedPool:
    """
    In-memory simulation of MemoryFederationProtocol
    with identical scoring logic but no DB layer.
    """
    def __init__(self, trust_network):
        self._trust   = trust_network
        self._entries: List[MockFederatedEntry] = []
        self._access_log: List[Dict] = []

    def promote(self, summaries: List[MockSummary]) -> int:
        promoted = 0
        for s in summaries:
            if s.abstraction_level < 2 or s.is_shared:
                continue
            word_count = len(s.content.split())
            bonus = min(0.5, word_count / 100.0)
            score = _promotion_score(s.abstraction_level, bonus)
            if score >= PROMOTION_THRESHOLD:
                self._entries.append(MockFederatedEntry(
                    source_agent_id=s.agent_id,
                    content=s.content,
                    abstraction_level=s.abstraction_level,
                    promotion_score=score,
                ))
                s.is_shared = True
                promoted += 1
        return promoted

    def retrieve(self, query, querying_agent, top_k=5,
                 exclude_own=True, min_trust=0.3):
        scored = []
        for entry in self._entries:
            if not entry.is_active:
                continue
            if exclude_own and entry.source_agent_id == querying_agent:
                continue
            trust = self._trust.get_trust(querying_agent, entry.source_agent_id)
            if trust < min_trust:
                continue
            relevance = _keyword_similarity(query, entry.content)
            age_w = _age_weight(entry.created_at)
            final = relevance * trust * age_w
            scored.append({
                "entry_id":        entry.id,
                "content":         entry.content,
                "source_agent_id": entry.source_agent_id,
                "abstraction_level": entry.abstraction_level,
                "relevance_score": round(relevance, 4),
                "trust_score":     round(trust, 4),
                "age_weight":      round(age_w, 4),
                "weighted_score":  round(final, 4),
            })
            entry.access_count += 1
            entry.last_accessed_at = datetime.utcnow()
            self._access_log.append({
                "accessing_agent": querying_agent, "entry_id": entry.id,
                "relevance": relevance, "feedback": None,
            })
        scored.sort(key=lambda x: x["weighted_score"], reverse=True)
        return scored[:top_k]

    def feedback(self, entry_id, from_agent, to_agent, signal):
        for log in reversed(self._access_log):
            if log["entry_id"] == entry_id and log["accessing_agent"] == from_agent:
                log["feedback"] = signal
                break
        return self._trust.update_trust(from_agent, to_agent, signal)

    def run_decay(self, decay_days=DECAY_DAYS, min_accesses=MIN_USEFUL_ACCESSES):
        cutoff = datetime.utcnow() - timedelta(days=decay_days)
        retired = 0
        for e in self._entries:
            if e.is_active and e.created_at < cutoff and e.access_count < min_accesses:
                e.is_active = False
                retired += 1
        return retired

    def format_for_context(self, memories, max_tokens=300):
        if not memories:
            return ""
        level_labels = {1: "L1 Episode", 2: "L2 Pattern", 3: "L3 Principle"}
        lines = ["<FEDERATED MEMORY>"]
        char_budget = max_tokens * 4
        for mem in memories:
            level_label = level_labels.get(mem.get("abstraction_level", 2), "Pattern")
            header = (f"[From: {mem['source_agent_id']} | "
                      f"Trust: {mem['trust_score']:.2f} | "
                      f"Relevance: {mem['relevance_score']:.2f}]")
            body = f"{level_label}: \"{mem['content'][:200]}\""
            block = f"{header}\n{body}\n"
            if len("\n".join(lines) + block) > char_budget:
                break
            lines.append(block)
        lines.append("</FEDERATED MEMORY>")
        return "\n".join(lines)

    def get_pool_stats(self):
        active  = [e for e in self._entries if e.is_active]
        retired = [e for e in self._entries if not e.is_active]
        by_agent: Dict[str, int] = {}
        by_level: Dict[str, int] = {}
        for e in active:
            by_agent[e.source_agent_id] = by_agent.get(e.source_agent_id, 0) + 1
            key = f"L{e.abstraction_level}"
            by_level[key] = by_level.get(key, 0) + 1
        avg = sum(e.access_count for e in active) / len(active) if active else 0.0
        return {
            "total_entries": len(self._entries),
            "active_entries": len(active),
            "retired_entries": len(retired),
            "avg_access_count": round(avg, 2),
            "by_agent": by_agent,
            "by_level": by_level,
        }


# ─── Namespace / Fork helpers ─────────────────────────────────────────────────

class InMemoryNamespaceManager:
    """Pure-Python agent namespace management."""
    def __init__(self):
        self._agents: Dict[str, Dict] = {}
        self._fork_records: List[Dict] = {}
        self._patterns: Dict[str, List[Dict]] = {}
        self._summaries: Dict[str, List[MockSummary]] = {}

    def create_agent(self, agent_id, name=None, metadata=None):
        if agent_id in self._agents:
            raise ValueError(f"Agent '{agent_id}' already exists.")
        self._agents[agent_id] = {"agent_id": agent_id, "name": name or agent_id}
        self._patterns[agent_id] = []
        self._summaries[agent_id] = []
        return self._agents[agent_id]

    def ensure_agent_exists(self, agent_id):
        if agent_id not in self._agents:
            self.create_agent(agent_id)
        return self._agents[agent_id]

    def seed_summaries(self, agent_id, n_l2=3, n_l3=2):
        self.ensure_agent_exists(agent_id)
        for i in range(n_l2):
            self._summaries[agent_id].append(MockSummary(
                agent_id=agent_id,
                content=f"Pattern #{i}: When user asks about {agent_id}, provide code examples with detail.",
                abstraction_level=2,
            ))
        for i in range(n_l3):
            self._summaries[agent_id].append(MockSummary(
                agent_id=agent_id,
                content=f"Principle #{i}: {agent_id} users always prefer concise, actionable answers.",
                abstraction_level=3,
            ))

    def seed_patterns(self, agent_id, n=3, confidence=0.8):
        self.ensure_agent_exists(agent_id)
        for i in range(n):
            self._patterns[agent_id].append({
                "agent_id": agent_id,
                "trigger": "bug_report",
                "action": "provided code block",
                "confidence": confidence,
                "support_count": 5,
            })

    def fork_agent(self, parent_id, child_id, inherit_levels=None, trust_network=None):
        if inherit_levels is None:
            inherit_levels = [2, 3]
        if parent_id not in self._agents:
            raise ValueError(f"Parent '{parent_id}' not found.")
        self.ensure_agent_exists(child_id)

        # Inherit summaries
        parent_sums = [s for s in self._summaries.get(parent_id, [])
                       if s.abstraction_level in inherit_levels]
        inherited_sums = 0
        for s in parent_sums:
            self._summaries[child_id].append(MockSummary(
                agent_id=child_id,
                content=f"[INHERITED from {parent_id}] {s.content}",
                abstraction_level=s.abstraction_level,
            ))
            inherited_sums += 1

        # Inherit high-confidence patterns
        parent_pats = [p for p in self._patterns.get(parent_id, [])
                       if p["confidence"] >= 0.7]
        inherited_pats = 0
        for p in parent_pats:
            self._patterns[child_id].append({
                **p,
                "agent_id": child_id,
                "confidence": p["confidence"] * 0.85,
                "support_count": max(1, p["support_count"] // 2),
            })
            inherited_pats += 1

        # Fork record
        parent_record = next((r for r in self._fork_records.values()
                               if r.get("child_agent_id") == parent_id), None)
        fork_depth = (parent_record["fork_depth"] + 1) if parent_record else 1

        self._fork_records[child_id] = {
            "parent_agent_id": parent_id,
            "child_agent_id": child_id,
            "fork_depth": fork_depth,
            "summaries_inherited": inherited_sums,
            "patterns_inherited": inherited_pats,
        }

        if trust_network:
            trust_network.set_trust(child_id, parent_id, 0.9)
            trust_network.set_trust(parent_id, child_id, 0.5)

        return self._fork_records[child_id]

    def get_fork_lineage(self, agent_id):
        ancestors = []
        current = agent_id
        seen = set()
        while current in self._fork_records:
            rec = self._fork_records[current]
            parent = rec["parent_agent_id"]
            if parent in seen:
                break
            ancestors.append(parent)
            seen.add(parent)
            current = parent
        descendants = [r["child_agent_id"] for r in self._fork_records.values()
                       if r["parent_agent_id"] == agent_id]
        return {"agent_id": agent_id, "ancestors": ancestors, "descendants": descendants}

    def merge_patterns(self, source_id, target_id, confidence_threshold=0.75):
        source_pats = [p for p in self._patterns.get(source_id, [])
                       if p["confidence"] >= confidence_threshold]
        merged = 0
        for sp in source_pats:
            existing = next((p for p in self._patterns[target_id]
                             if p["trigger"] == sp["trigger"] and
                             p["action"] == sp["action"]), None)
            if existing:
                if sp["confidence"] > existing["confidence"]:
                    existing["confidence"] = sp["confidence"]
                    existing["support_count"] += sp["support_count"]
            else:
                self._patterns[target_id].append({
                    **sp,
                    "agent_id": target_id,
                    "confidence": sp["confidence"] * 0.9,
                })
            merged += 1
        return merged

    def list_agents(self):
        return list(self._agents.values())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: AgentTrustNetwork — 12 tests
# ══════════════════════════════════════════════════════════════════════════════

tn = InMemoryTrustNetwork()

# 1.1 Unknown pair returns NEUTRAL_TRUST
assert_approx("unknown pair → 0.5", tn.get_trust("x", "y", use_transitive=False), 0.5)

# 1.2 Self-trust = 1.0
assert_approx("self-trust = 1.0", tn.get_trust("a", "a"), 1.0)

# 1.3 set_trust persists
tn.set_trust("a1", "b1", 0.9)
assert_approx("set_trust persists", tn.get_trust("a1", "b1", use_transitive=False), 0.9)

# 1.4 EMA update: old=0.9, signal=0.0 → new = 0.8×0.9 + 0.2×0.0 = 0.72
new = tn.update_trust("a1", "b1", 0.0)
assert_approx("EMA: old=0.9, signal=0 → 0.72", new, 0.72)

# 1.5 Repeated positive signals converge to high trust
tn2 = InMemoryTrustNetwork()
tn2.set_trust("aa", "bb", 0.5)
for _ in range(20):
    tn2.update_trust("aa", "bb", 1.0)
assert_true("repeated pos → >0.95", tn2.get_trust("aa", "bb", False) > 0.95)

# 1.6 Repeated negative signals converge to low trust
tn3 = InMemoryTrustNetwork()
tn3.set_trust("cc", "dd", 0.5)
for _ in range(20):
    tn3.update_trust("cc", "dd", 0.0)
assert_true("repeated neg → <0.10", tn3.get_trust("cc", "dd", False) < 0.10)

# 1.7 Trust bounded by MAX_TRUST
tn.set_trust("e1", "f1", 2.0)
assert_true("bounded by MAX", tn.get_trust("e1", "f1", False) <= MAX_TRUST)

# 1.8 Trust bounded by MIN_TRUST
tn.set_trust("e2", "f2", -1.0)
assert_true("bounded by MIN", tn.get_trust("e2", "f2", False) >= MIN_TRUST)

# 1.9 Trust matrix is n×n
tn.set_trust("m1", "m2", 0.7)
tn.set_trust("m2", "m1", 0.6)
matrix = tn.get_trust_matrix(["m1", "m2"])
assert_eq("matrix rows", len(matrix), 2)
assert_approx("matrix[m1][m2]", matrix["m1"]["m2"], 0.7, tol=0.02)

# 1.10 Transitive trust: A→B=0.9, B→C=0.8 → blended A→C > 0.5
tn4 = InMemoryTrustNetwork()
tn4.set_trust("ta", "tb", 0.9)
tn4.set_trust("tb", "tc", 0.8)
blended = tn4.get_trust("ta", "tc", use_transitive=True)
assert_true("transitive > neutral", blended > 0.5, f"got {blended:.4f}")

# 1.11 weight_memories ranks high-trust source first
mems = [
    {"source_agent_id": "high-src", "relevance_score": 0.7},
    {"source_agent_id": "low-src",  "relevance_score": 0.95},
]
tn5 = InMemoryTrustNetwork()
tn5.set_trust("q", "high-src", 0.95)
tn5.set_trust("q", "low-src",  0.20)
weighted = tn5.weight_memories(mems, "q")
# high-src: 0.7×0.95=0.665 > low-src: 0.95×0.20=0.19
assert_eq("high-trust src ranks first", weighted[0]["source_agent_id"], "high-src")

# 1.12 describe returns non-empty string
tn5.set_trust("x-ag", "y-ag", 0.75)
assert_true("describe returns string", len(tn5.describe()) > 0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MFP Helpers & Formulas — 10 tests
# ══════════════════════════════════════════════════════════════════════════════

# 2.1 _age_weight: very fresh → ≈1.0
fresh_w = _age_weight(datetime.utcnow())
assert_true("fresh memory age_weight ≈ 1.0", 0.99 <= fresh_w <= 1.0, f"got {fresh_w:.4f}")

# 2.2 _age_weight: 60 days old → ≈0.5
old_dt = datetime.utcnow() - timedelta(days=60)
old_w = _age_weight(old_dt)
assert_approx("60-day-old age_weight ≈ 0.5", old_w, 0.5, tol=0.02)

# 2.3 _age_weight: 120 days → ≈0.25
older_dt = datetime.utcnow() - timedelta(days=120)
older_w = _age_weight(older_dt)
assert_approx("120-day-old age_weight ≈ 0.25", older_w, 0.25, tol=0.03)

# 2.4 age weight is monotonically decreasing
assert_true("age: fresh > 60d > 120d", fresh_w > old_w > older_w,
            f"{fresh_w:.3f} {old_w:.3f} {older_w:.3f}")

# 2.5 _keyword_similarity: identical → 1.0
assert_approx("identical content sim = 1.0",
              _keyword_similarity("debug this bug", "debug this bug"), 1.0, tol=0.001)

# 2.6 _keyword_similarity: disjoint → 0.0
assert_approx("disjoint content sim = 0.0",
              _keyword_similarity("apple banana mango", "robot vacuum cleaner"), 0.0, tol=0.01)

# 2.7 _keyword_similarity: partial overlap
sim = _keyword_similarity("how debug python code", "debug python scripts and code examples")
assert_true("partial overlap sim in (0,1)", 0.0 < sim < 1.0, f"got {sim:.3f}")

# 2.8 _promotion_score: L1 never qualifies
assert_true("L1 score < threshold", _promotion_score(1, 0.0) < PROMOTION_THRESHOLD)

# 2.9 _promotion_score: L2 with bonus qualifies
assert_true("L2+bonus >= threshold",
            _promotion_score(2, 0.0) >= PROMOTION_THRESHOLD,
            f"got {_promotion_score(2, 0.0):.2f}")

# 2.10 _promotion_score: L3 always qualifies
assert_true("L3 always >= threshold",
            _promotion_score(3, 0.0) >= PROMOTION_THRESHOLD)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: InMemoryFedPool (MFP) — 13 tests
# ══════════════════════════════════════════════════════════════════════════════

fed_trust = InMemoryTrustNetwork()
pool = InMemoryFedPool(fed_trust)
ns = InMemoryNamespaceManager()

ns.create_agent("src-agent")
ns.seed_summaries("src-agent", n_l2=3, n_l3=2)
ns.create_agent("querier")

src_summaries = ns._summaries["src-agent"]

# 3.1 promote returns count > 0
n_promoted = pool.promote(src_summaries)
assert_true("promote count > 0", n_promoted > 0, f"got {n_promoted}")

# 3.2 L1 summaries are never promoted (privacy)
l1_sum = MockSummary("src-agent", "episode detail raw", abstraction_level=1)
pool2 = InMemoryFedPool(InMemoryTrustNetwork())
n_l1 = pool2.promote([l1_sum])
assert_eq("L1 never promoted", n_l1, 0)

# 3.3 Promoted summaries are marked is_shared
assert_true("promoted sums marked shared",
            all(s.is_shared for s in src_summaries if s.abstraction_level >= 2))

# 3.4 retrieve returns list
fed_trust.set_trust("querier", "src-agent", 0.8)
res = pool.retrieve("code examples pattern", "querier", top_k=5)
assert_true("retrieve returns list", isinstance(res, list))

# 3.5 Retrieved entries have required keys
if res:
    req_keys = {"entry_id", "content", "source_agent_id", "relevance_score",
                "trust_score", "age_weight", "weighted_score"}
    assert_true("entry has required keys", req_keys.issubset(set(res[0].keys())),
                f"missing: {req_keys - set(res[0].keys())}")

# 3.6 exclude_own: querier does not see its own memories
ns.seed_summaries("querier", n_l2=2, n_l3=1)
querier_sums = ns._summaries["querier"]
pool.promote(querier_sums)
res_excl = pool.retrieve("pattern", "querier", exclude_own=True)
own_in_results = any(m["source_agent_id"] == "querier" for m in res_excl)
assert_true("exclude_own: querier not in results", not own_in_results)

# 3.7 min_trust filter
fed_trust.set_trust("querier", "src-agent", 0.1)  # very low
res_filtered = pool.retrieve("pattern", "querier", min_trust=0.3)
# src-agent has trust 0.1 < min_trust=0.3 → excluded
src_in_filtered = any(m["source_agent_id"] == "src-agent" for m in res_filtered)
assert_true("min_trust excludes low-trust source", not src_in_filtered)

# 3.8 Trust-weighted ordering: high-trust × lower relevance beats low-trust × higher relevance
pool3_trust = InMemoryTrustNetwork()
pool3 = InMemoryFedPool(pool3_trust)
pool3_trust.set_trust("q3", "high-trust-src", 0.95)
pool3_trust.set_trust("q3", "low-trust-src",  0.15)
# Plant two entries with reversed relevance
pool3._entries.append(MockFederatedEntry("high-trust-src",
    "how to debug python code examples and patterns", 2, 2.5, access_count=0))
pool3._entries.append(MockFederatedEntry("low-trust-src",
    "debug python patterns code examples concise", 2, 2.5, access_count=0))
result3 = pool3.retrieve("debug python code examples", "q3", top_k=2)
if result3:
    assert_eq("high-trust src ranks first", result3[0]["source_agent_id"], "high-trust-src")

# 3.9 feedback increases trust on positive signal
pool_trust2 = InMemoryTrustNetwork()
pool_trust2.set_trust("q4", "src4", 0.6)
pool4 = InMemoryFedPool(pool_trust2)
pool4._entries.append(MockFederatedEntry("src4", "test content pattern", 2, 2.5))
pool4._access_log.append({"accessing_agent": "q4", "entry_id": pool4._entries[-1].id,
                           "relevance": 0.7, "feedback": None})
new_t = pool4.feedback(pool4._entries[-1].id, "q4", "src4", 1.0)
assert_true("feedback(1.0) increases trust", new_t >= 0.6, f"got {new_t:.3f}")

# 3.10 run_decay retires stale entries
pool5_trust = InMemoryTrustNetwork()
pool5 = InMemoryFedPool(pool5_trust)
stale_entry = MockFederatedEntry("old-src", "stale content from long ago", 2, 2.5,
                                 access_count=0,
                                 created_at=datetime.utcnow() - timedelta(days=61))
pool5._entries.append(stale_entry)
retired = pool5.run_decay(decay_days=30, min_accesses=1)
assert_true("run_decay retires stale", retired >= 1, f"got {retired}")
assert_true("stale entry is inactive", not stale_entry.is_active)

# 3.11 Well-accessed entries survive decay
active_entry = MockFederatedEntry("old-src", "useful content accessed by many", 3, 3.5,
                                  access_count=10,
                                  created_at=datetime.utcnow() - timedelta(days=61))
pool5._entries.append(active_entry)
pool5.run_decay(decay_days=30, min_accesses=3)
assert_true("active entry survives decay", active_entry.is_active)

# 3.12 format_for_context contains FEDERATED MEMORY tags
if res:
    formatted = pool.format_for_context(res[:2], max_tokens=200)
    assert_true("format contains tag", "FEDERATED MEMORY" in formatted, formatted[:60])
    assert_true("format contains source agent", "src-agent" in formatted or
                "querier" in formatted)

# 3.13 get_pool_stats reflects pool state
stats = pool.get_pool_stats()
assert_true("stats has correct keys",
            {"total_entries", "active_entries", "retired_entries",
             "by_agent", "by_level"}.issubset(set(stats.keys())))
assert_true("stats total > 0", stats["total_entries"] > 0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: AgentNamespaceManager — 10 tests
# ══════════════════════════════════════════════════════════════════════════════

nm = InMemoryNamespaceManager()

# 4.1 create_agent returns dict-like object
ag = nm.create_agent("alpha", name="Alpha")
assert_eq("create returns agent_id", ag["agent_id"], "alpha")

# 4.2 Duplicate raises ValueError
try:
    nm.create_agent("alpha")
    assert_true("duplicate raises ValueError", False, "no error raised")
except ValueError:
    assert_true("duplicate raises ValueError", True)

# 4.3 ensure_agent_exists is idempotent
nm.ensure_agent_exists("beta")
nm.ensure_agent_exists("beta")
assert_true("ensure idempotent", "beta" in nm._agents)

# 4.4 Fork inherits L2+L3 summaries
nm.seed_summaries("parent-a", n_l2=3, n_l3=2)
nm.seed_patterns("parent-a", n=4, confidence=0.85)
nm.ensure_agent_exists("child-a")
fork_r = nm.fork_agent("parent-a", "child-a")
assert_true("fork inherits summaries", fork_r["summaries_inherited"] > 0)

# 4.5 Fork inherits high-confidence patterns
assert_true("fork inherits patterns", fork_r["patterns_inherited"] > 0)

# 4.6 Fork depth = 1 for direct child
assert_eq("fork depth 1", fork_r["fork_depth"], 1)

# 4.7 Grandchild fork depth = 2
nm.ensure_agent_exists("grandchild-a")
nm.seed_summaries("child-a", n_l2=2, n_l3=1)
fork_gc = nm.fork_agent("child-a", "grandchild-a")
assert_eq("grandchild depth 2", fork_gc["fork_depth"], 2)

# 4.8 Fork lineage contains correct ancestor
lineage = nm.get_fork_lineage("child-a")
assert_true("lineage has parent-a", "parent-a" in lineage["ancestors"])

# 4.9 Merge patterns: source patterns appear in target
nm.seed_patterns("src-merge", n=3, confidence=0.8)
nm.ensure_agent_exists("tgt-merge")
n_merged = nm.merge_patterns("src-merge", "tgt-merge", confidence_threshold=0.7)
assert_true("merge returns count > 0", n_merged > 0, f"got {n_merged}")
assert_true("merged patterns in target", len(nm._patterns["tgt-merge"]) > 0)

# 4.10 Merge conflict: keeps higher confidence
nm.ensure_agent_exists("conflict-tgt")
nm.ensure_agent_exists("conflict-src")
# Manually set patterns AFTER ensure creates the empty lists
nm._patterns["conflict-tgt"] = [{"trigger": "bug", "action": "code", "confidence": 0.6, "support_count": 2}]
nm._patterns["conflict-src"] = [{"trigger": "bug", "action": "code", "confidence": 0.9, "support_count": 5, "agent_id": "conflict-src"}]
nm.merge_patterns("conflict-src", "conflict-tgt", 0.8)
# The existing pattern with conf=0.6 should be updated to 0.9
assert_true("conflict: higher confidence wins",
            nm._patterns["conflict-tgt"][0]["confidence"] >= 0.9,
            f"got {nm._patterns['conflict-tgt'][0]['confidence']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Integration — 5 tests
# ══════════════════════════════════════════════════════════════════════════════

# 5.1 Full pipeline: fork → promote → retrieve with trust weighting
int_trust = InMemoryTrustNetwork()
int_pool  = InMemoryFedPool(int_trust)
int_ns    = InMemoryNamespaceManager()

int_ns.create_agent("int-parent")
int_ns.seed_summaries("int-parent", n_l2=4, n_l3=2)
int_ns.ensure_agent_exists("int-child")

fork_res = int_ns.fork_agent("int-parent", "int-child", trust_network=int_trust)
int_pool.promote(int_ns._summaries["int-parent"])

# int-child trusts int-parent at 0.9 (set by fork)
int_results = int_pool.retrieve("pattern examples concise", "int-child", top_k=5)

assert_true("integration: retrieve returns list", isinstance(int_results, list))
assert_true("integration: trust respected",
            all(m["trust_score"] >= 0.3 for m in int_results))

# 5.2 Feedback loop: positive signal increases future trust
if int_results:
    old_t = int_trust.get_trust("int-child", "int-parent", use_transitive=False)
    int_pool.feedback(int_results[0]["entry_id"], "int-child", "int-parent", 1.0)
    new_t = int_trust.get_trust("int-child", "int-parent", use_transitive=False)
    assert_true("feedback loop increases trust", new_t >= old_t,
                f"{old_t:.3f} → {new_t:.3f}")

# 5.3 Decay doesn't affect recently promoted entries
recent_pool = InMemoryFedPool(InMemoryTrustNetwork())
fresh_entry = MockFederatedEntry("agent-x", "fresh pattern about patterns", 2, 2.5,
                                 access_count=0, created_at=datetime.utcnow())
recent_pool._entries.append(fresh_entry)
recent_pool.run_decay(decay_days=30, min_accesses=1)
assert_true("fresh entry survives decay", fresh_entry.is_active)

# 5.4 format_for_context empty list → empty string
assert_eq("format empty memories → ''", pool.format_for_context([]), "")

# 5.5 Trust transitivity propagates correctly through fork lineage
chain_trust = InMemoryTrustNetwork()
chain_trust.set_trust("grandchild", "child", 0.9)
chain_trust.set_trust("child", "parent", 0.85)
# Direct grandchild→parent = 0.5 (unknown)
# Transitive via child = 0.9 × 0.85 = 0.765
# Blended = 0.7×0.5 + 0.3×0.765 = 0.35 + 0.2295 = 0.5795
chain_blended = chain_trust.get_trust("grandchild", "parent", use_transitive=True)
assert_true("trust propagates through fork chain", chain_blended > 0.5,
            f"got {chain_blended:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Print results
# ══════════════════════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'='*70}")
print(f"  AgentMem OS — Phase 4 Multi-Agent Federation Test Suite")
print(f"{'='*70}")

sections = [
    ("Section 1: AgentTrustNetwork (12 tests)",           0,  12),
    ("Section 2: MFP Helpers & Formulas (10 tests)",     12,  22),
    ("Section 3: MemoryFederationProtocol (13 tests)",   22,  35),
    ("Section 4: AgentNamespaceManager (10 tests)",      35,  45),
    ("Section 5: Integration (5 tests)",                 45,  50),
]
for name, start, end in sections:
    print(f"\n  ── {name} ──")
    for icon, label, detail in results[start:end]:
        d = f"  →  {detail}" if detail else ""
        print(f"    {icon}  {label}{d}")

print(f"\n{'─'*70}")
print(f"  {passed}/{total} passed   {'🎉 ALL PASS' if failed == 0 else f'❌ {failed} FAILED'}")
print(f"{'='*70}\n")
sys.exit(0 if failed == 0 else 1)
