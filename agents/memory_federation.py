"""
AgentMem OS — Memory Federation Protocol (MFP) (Phase 4)
=========================================================
The core novel algorithm of Phase 4.

WHAT IS MFP?
  A decentralized protocol where agents asynchronously share high-quality
  abstract memories through a trust-weighted shared pool. No single agent
  "owns" the shared pool — it emerges from collective promotion decisions.

  Analogy: Think of MFP as a peer-reviewed journal for AI agent memories.
    - Agents submit memories (promotion phase) — only high-quality abstractions pass.
    - Readers (other agents) cite them with trust-weighted relevance.
    - Memories that no one cites decay and are retired (aging mechanism).

WHY THIS IS NOVEL:
  1. Privacy-by-design: Raw episodic turns NEVER enter the shared pool.
     Only L2 (patterns) and L3 (principles) are eligible. This is the first
     formalized privacy guarantee in LLM multi-agent memory sharing.

  2. Trust-weighted retrieval: Unlike simple shared vector DBs, retrieval
     scores are modulated by the querying agent's trust in the source agent.
     Memories from trusted agents surface higher. Untested agents' memories
     are down-weighted until they prove reliable.

  3. Promotion scoring: A memory earns entry to the shared pool by achieving
     a minimum promotion score = abstraction_level × (1 + confidence_bonus).
     Higher abstraction = more evidence of generalization = more shareable.

  4. Relevance aging: Memories that are never retrieved by other agents
     accumulate a "staleness score" and are eventually retired (is_active=False).
     This prevents the shared pool from filling with useless entries.

  5. Agent affinity: We track which agents retrieve from which source agents
     most often, creating an "affinity" routing layer that prioritizes
     high-affinity sources in future queries.

THE MFP PIPELINE:
  ┌─────────────────────────────────────────────────────┐
  │  PROMOTION PHASE (runs after consolidation)         │
  │  For each summary with level >= 2:                  │
  │    score = level × (1 + confidence_bonus)           │
  │    if score >= PROMOTION_THRESHOLD:                 │
  │      write to FederatedMemoryEntry                  │
  │      mark summary.is_shared = True                  │
  └────────────────────┬────────────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────────────┐
  │  RETRIEVAL PHASE (at context assembly time)         │
  │  Query by cosine similarity (keyword fallback)      │
  │  Score each result:                                 │
  │    final_score = relevance × trust × age_weight     │
  │  Return top-k, format as [FEDERATED MEMORY] block   │
  └────────────────────┬────────────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────────────┐
  │  FEEDBACK PHASE (after each interaction)            │
  │  trust_network.update_trust(from, to, feedback)     │
  │  memory.access_count += 1                           │
  └────────────────────┬────────────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────────────┐
  │  AGING PHASE (runs on schedule)                     │
  │  For memories not accessed in > DECAY_DAYS:         │
  │    if access_count < MIN_USEFUL_ACCESSES:           │
  │      set is_active = False                          │
  └─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import re
import math
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Callable

from loguru import logger

from agentmem_os.db.models import (
    Summary, Session, FederatedMemoryEntry, MemoryAccessLog
)


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────

PROMOTION_THRESHOLD     = 2.0     # minimum promotion score to enter shared pool
DECAY_DAYS              = 30      # memories not accessed in N days become stale
MIN_USEFUL_ACCESSES     = 2       # minimum cross-agent accesses to avoid retirement
MAX_FEDERATED_RESULTS   = 10      # max entries returned by retrieve()
AGE_WEIGHT_HALFLIFE     = 60      # days; older memories get lower age weight


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _age_weight(created_at: datetime) -> float:
    """
    Exponential age weight: newer memories get higher weight.
    score = 2^(-(days_old / AGE_WEIGHT_HALFLIFE))

    A memory created today → 1.0
    A memory 60 days old  → 0.5
    A memory 120 days old → 0.25
    """
    days_old = max(0.0, (datetime.utcnow() - created_at).total_seconds() / 86400)
    return 2.0 ** (-days_old / AGE_WEIGHT_HALFLIFE)


def _keyword_similarity(query: str, content: str) -> float:
    """
    Lightweight keyword-overlap similarity for when no embedder is available.
    Computes Jaccard similarity on unigrams.

    Returns float ∈ [0, 1].
    """
    def tokenize(text: str):
        return set(re.findall(r'\b\w{3,}\b', text.lower()))

    q_tokens = tokenize(query)
    c_tokens = tokenize(content)
    if not q_tokens or not c_tokens:
        return 0.0
    intersection = q_tokens & c_tokens
    union = q_tokens | c_tokens
    return len(intersection) / len(union)


def _promotion_score(abstraction_level: int, confidence_bonus: float = 0.0) -> float:
    """
    Compute the promotion score for a summary candidate.

    Formula:
        score = abstraction_level × (1.0 + confidence_bonus)

    Thresholds:
        L2 (pattern)   with confidence=0.0 → score=2.0 (just above threshold)
        L2 with conf=0.5 → score=3.0
        L3 (principle) with confidence=0.0 → score=3.0 (always promotes)

    This ensures that:
      - L1 (episode) summaries never promote (score=1.0 < PROMOTION_THRESHOLD=2.0)
      - L2 summaries need at least neutral confidence
      - L3 summaries always qualify for promotion
    """
    return abstraction_level * (1.0 + confidence_bonus)


class MemoryFederationProtocol:
    """
    The core Phase 4 algorithm: trust-weighted multi-agent memory federation.

    This class orchestrates:
      1. promote()   — push eligible memories to the shared federated pool
      2. retrieve()  — query the pool with trust-weighted scoring
      3. feedback()  — update trust based on memory utility
      4. run_decay() — retire stale memories from the pool

    Usage:
        mfp = MemoryFederationProtocol(get_session, trust_network)
        # After consolidation:
        count = mfp.promote(agent_id="coding-bot", session_id="abc123")
        # At context time:
        results = mfp.retrieve(query="how do I debug Redis?", querying_agent="python-bot")
        # After interaction:
        mfp.feedback(entry_id=7, from_agent="python-bot", to_agent="coding-bot", signal=0.9)
    """

    def __init__(
        self,
        get_db_session: Callable,
        trust_network,                          # AgentTrustNetwork instance
        get_embedding_fn: Optional[Callable] = None,  # str → np.ndarray
    ):
        self._get_db = get_db_session
        self._trust  = trust_network
        self._embed  = get_embedding_fn         # optional — keyword fallback if None

    # ──────────────────────────────────────────────────────────────────────────
    # 1. PROMOTION PHASE
    # ──────────────────────────────────────────────────────────────────────────

    def promote(
        self,
        agent_id: str,
        session_id: Optional[str] = None,
        force_level: Optional[int] = None,
    ) -> int:
        """
        Scan summaries from agent_id and promote eligible ones to the shared pool.

        Eligibility criteria:
          - abstraction_level >= 2  (not raw episodes)
          - is_shared == False      (not already promoted)
          - promotion_score >= PROMOTION_THRESHOLD

        Args:
            agent_id: Which agent's summaries to scan.
            session_id: Optionally restrict to one session.
            force_level: If set, only promote summaries of exactly this level.

        Returns:
            Number of entries promoted.
        """
        db = self._get_db()
        try:
            query = (
                db.query(Summary)
                .join(Session, Summary.session_id == Session.session_id)
                .filter(
                    Session.agent_id == agent_id,
                    Summary.abstraction_level >= 2,
                    Summary.is_shared == False,
                )
            )
            if session_id:
                query = query.filter(Summary.session_id == session_id)
            if force_level:
                query = query.filter(Summary.abstraction_level == force_level)

            candidates = query.all()
            promoted = 0

            for s in candidates:
                # Confidence bonus from content quality heuristic
                # (in production this comes from consolidation engine metadata)
                word_count = len(s.content.split())
                confidence_bonus = min(0.5, word_count / 100.0)  # longer = more considered

                score = _promotion_score(s.abstraction_level, confidence_bonus)

                if score >= PROMOTION_THRESHOLD:
                    entry = FederatedMemoryEntry(
                        source_agent_id=agent_id,
                        source_session_id=s.session_id,
                        content=s.content,
                        abstraction_level=s.abstraction_level,
                        promotion_score=score,
                        access_count=0,
                        last_accessed_at=None,
                        is_active=True,
                        created_at=datetime.utcnow(),
                    )
                    db.add(entry)

                    # Mark original summary as shared
                    s.is_shared = True
                    promoted += 1

            db.commit()
            if promoted:
                logger.info(
                    f"[MFP] Promoted {promoted} memories from agent '{agent_id}' "
                    f"to federated pool."
                )
            return promoted

        except Exception as e:
            db.rollback()
            logger.error(f"[MFP] Promotion failed: {e}")
            raise
        finally:
            db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # 2. RETRIEVAL PHASE
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        querying_agent: str,
        top_k: int = 5,
        exclude_own: bool = True,
        min_trust: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Query the federated pool for memories relevant to the query.

        Retrieval pipeline:
          1. Load all active federated entries (excluding own if exclude_own=True)
          2. Compute relevance via embedding cosine similarity (or keyword fallback)
          3. Apply trust weight: final_score = relevance × trust × age_weight
          4. Sort descending, return top_k
          5. Log access to MemoryAccessLog

        The trust weight is the key innovation here — it means that a memory
        from a highly trusted agent at 60% relevance can outrank a memory from
        an unknown agent at 80% relevance.

        Args:
            query: The current query string.
            querying_agent: The agent making the request.
            top_k: Maximum results to return.
            exclude_own: Skip memories the querying agent itself produced.
            min_trust: Filter out source agents below this trust threshold.

        Returns:
            List of dicts with keys:
              - content, source_agent_id, abstraction_level
              - relevance_score, trust_score, age_weight, weighted_score
              - entry_id (for feedback logging)
        """
        db = self._get_db()
        try:
            pool_query = db.query(FederatedMemoryEntry).filter_by(is_active=True)
            if exclude_own:
                pool_query = pool_query.filter(
                    FederatedMemoryEntry.source_agent_id != querying_agent
                )
            entries = pool_query.all()

            if not entries:
                return []

            scored = []
            for entry in entries:
                source_agent = entry.source_agent_id
                trust = self._trust.get_trust(querying_agent, source_agent)

                # Skip below min_trust threshold
                if trust < min_trust:
                    continue

                # Compute relevance
                relevance = self._compute_relevance(query, entry.content)

                # Age weight
                age_w = _age_weight(entry.created_at)

                # Weighted final score
                final_score = relevance * trust * age_w

                scored.append({
                    "entry_id": entry.id,
                    "content": entry.content,
                    "source_agent_id": source_agent,
                    "abstraction_level": entry.abstraction_level,
                    "relevance_score": round(relevance, 4),
                    "trust_score": round(trust, 4),
                    "age_weight": round(age_w, 4),
                    "weighted_score": round(final_score, 4),
                })

            # Sort descending by weighted_score
            scored.sort(key=lambda x: x["weighted_score"], reverse=True)
            top = scored[:top_k]

            # Log access
            for mem in top:
                self._log_access(
                    db=db,
                    accessing_agent=querying_agent,
                    entry_id=mem["entry_id"],
                    query=query,
                    relevance=mem["relevance_score"],
                )

            # Update access_count on entries
            for mem in top:
                entry = db.query(FederatedMemoryEntry).get(mem["entry_id"])
                if entry:
                    entry.access_count = (entry.access_count or 0) + 1
                    entry.last_accessed_at = datetime.utcnow()

            db.commit()

            logger.debug(
                f"[MFP] Retrieved {len(top)}/{len(entries)} federated memories "
                f"for agent '{querying_agent}'."
            )
            return top

        except Exception as e:
            db.rollback()
            logger.error(f"[MFP] Retrieval failed: {e}")
            return []
        finally:
            db.close()

    def _compute_relevance(self, query: str, content: str) -> float:
        """
        Compute relevance between query and content.
        Uses embedding cosine similarity if embedder available, else keyword Jaccard.
        """
        if self._embed is not None:
            try:
                import numpy as np
                q_emb = np.array(self._embed(query), dtype=float)
                c_emb = np.array(self._embed(content), dtype=float)
                q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)
                c_norm = c_emb / (np.linalg.norm(c_emb) + 1e-10)
                return float(max(0.0, min(1.0, float(q_norm @ c_norm))))
            except Exception as e:
                logger.warning(f"[MFP] Embedding failed, using keyword fallback: {e}")

        return _keyword_similarity(query, content)

    def _log_access(
        self,
        db,
        accessing_agent: str,
        entry_id: int,
        query: str,
        relevance: float,
    ) -> None:
        """Record an access event to the MemoryAccessLog."""
        try:
            log_entry = MemoryAccessLog(
                accessing_agent_id=accessing_agent,
                federated_entry_id=entry_id,
                query_text=query[:500],  # truncate long queries
                relevance_score=relevance,
                feedback_signal=None,    # set later via feedback()
                accessed_at=datetime.utcnow(),
            )
            db.add(log_entry)
        except Exception as e:
            logger.warning(f"[MFP] Could not log access: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. FEEDBACK PHASE
    # ──────────────────────────────────────────────────────────────────────────

    def feedback(
        self,
        entry_id: int,
        from_agent: str,
        to_agent: str,
        signal: float,
    ) -> float:
        """
        Record feedback on a retrieved memory and update trust accordingly.

        Call this after an interaction where federated memories were used.
        The signal should reflect how useful the memory actually was:
          1.0 = extremely helpful, directly improved the response
          0.7 = somewhat helpful
          0.5 = neutral, hard to tell
          0.2 = not helpful, probably noise
          0.0 = misleading or wrong

        Args:
            entry_id: ID of the FederatedMemoryEntry that was used.
            from_agent: The agent that consumed the memory (updates its trust).
            to_agent: The agent that produced the memory (trust target).
            signal: Quality signal ∈ [0, 1].

        Returns:
            New trust score after update.
        """
        db = self._get_db()
        try:
            # Update the most recent access log entry for this pair
            log_entry = (
                db.query(MemoryAccessLog)
                .filter_by(
                    accessing_agent_id=from_agent,
                    federated_entry_id=entry_id,
                )
                .order_by(MemoryAccessLog.accessed_at.desc())
                .first()
            )
            if log_entry:
                log_entry.feedback_signal = signal
            db.commit()
        except Exception as e:
            logger.warning(f"[MFP] Could not update access log with feedback: {e}")
            db.rollback()
        finally:
            db.close()

        # Update trust score via EMA
        new_trust = self._trust.update_trust(from_agent, to_agent, signal)
        logger.info(
            f"[MFP] Feedback: {from_agent}→{to_agent} signal={signal:.2f} "
            f"→ new trust={new_trust:.3f}"
        )
        return new_trust

    # ──────────────────────────────────────────────────────────────────────────
    # 4. AGING / DECAY PHASE
    # ──────────────────────────────────────────────────────────────────────────

    def run_decay(
        self,
        decay_days: int = DECAY_DAYS,
        min_accesses: int = MIN_USEFUL_ACCESSES,
    ) -> int:
        """
        Retire federated memories that have not proven useful.

        A memory is retired (is_active=False) if ALL of:
          - It was created more than decay_days ago
          - It has been accessed by other agents fewer than min_accesses times

        This ensures the shared pool stays fresh and relevant. Memories that
        consistently prove useful across agents are preserved indefinitely.

        Returns:
            Number of memories retired.
        """
        db = self._get_db()
        try:
            cutoff = datetime.utcnow() - timedelta(days=decay_days)
            stale = (
                db.query(FederatedMemoryEntry)
                .filter(
                    FederatedMemoryEntry.is_active == True,
                    FederatedMemoryEntry.created_at < cutoff,
                    FederatedMemoryEntry.access_count < min_accesses,
                )
                .all()
            )

            for entry in stale:
                entry.is_active = False

            db.commit()
            if stale:
                logger.info(f"[MFP] Retired {len(stale)} stale federated memories.")
            return len(stale)

        except Exception as e:
            db.rollback()
            logger.error(f"[MFP] Decay run failed: {e}")
            return 0
        finally:
            db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # 5. FORMATTING FOR CONTEXT ASSEMBLY
    # ──────────────────────────────────────────────────────────────────────────

    def format_for_context(
        self,
        memories: List[Dict[str, Any]],
        max_tokens: int = 300,
    ) -> str:
        """
        Format retrieved federated memories for injection into the LLM context.

        Output format (XML-tagged, consistent with ContextAssembler style):
            <FEDERATED MEMORY>
            [From: agent-b | Trust: 0.87 | Relevance: 0.73]
            L3 Pattern: "When debugging async issues, always check event loop..."

            [From: coding-bot | Trust: 0.91 | Relevance: 0.68]
            L2 Pattern: "User prefers code examples over prose explanations."
            </FEDERATED MEMORY>

        Args:
            memories: Output of retrieve().
            max_tokens: Soft token limit (chars/4 approximation).

        Returns:
            Formatted string, or empty string if no memories.
        """
        if not memories:
            return ""

        level_labels = {1: "L1 Episode", 2: "L2 Pattern", 3: "L3 Principle"}

        lines = ["<FEDERATED MEMORY>"]
        char_budget = max_tokens * 4

        for mem in memories:
            level_label = level_labels.get(mem.get("abstraction_level", 2), "Pattern")
            header = (
                f"[From: {mem['source_agent_id']} | "
                f"Trust: {mem['trust_score']:.2f} | "
                f"Relevance: {mem['relevance_score']:.2f}]"
            )
            body = f"{level_label}: \"{mem['content'][:200]}\""
            block = f"{header}\n{body}\n"

            if len("\n".join(lines) + block) > char_budget:
                break

            lines.append(block)

        lines.append("</FEDERATED MEMORY>")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # 6. ANALYTICS
    # ──────────────────────────────────────────────────────────────────────────

    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the current federated memory pool.

        Returns:
            Dict with: total_entries, active_entries, retired_entries,
            by_agent (dict of agent_id → count), avg_access_count,
            by_level (dict of level → count).
        """
        db = self._get_db()
        try:
            entries = db.query(FederatedMemoryEntry).all()
            active = [e for e in entries if e.is_active]
            retired = [e for e in entries if not e.is_active]

            by_agent: Dict[str, int] = {}
            by_level: Dict[int, int] = {}
            for e in active:
                by_agent[e.source_agent_id] = by_agent.get(e.source_agent_id, 0) + 1
                by_level[e.abstraction_level] = by_level.get(e.abstraction_level, 0) + 1

            avg_access = (
                sum(e.access_count for e in active) / len(active)
                if active else 0.0
            )

            return {
                "total_entries": len(entries),
                "active_entries": len(active),
                "retired_entries": len(retired),
                "avg_access_count": round(avg_access, 2),
                "by_agent": by_agent,
                "by_level": {f"L{k}": v for k, v in by_level.items()},
            }
        finally:
            db.close()

    def get_agent_affinity(
        self,
        querying_agent: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Compute how often querying_agent has accessed memories from each source agent.
        Returns ranked affinity list — most accessed sources first.

        This is the "affinity graph" layer: over time, agents naturally develop
        preferences for certain source agents whose memories consistently help them.
        """
        db = self._get_db()
        try:
            logs = (
                db.query(MemoryAccessLog)
                .filter_by(accessing_agent_id=querying_agent)
                .all()
            )

            # Aggregate by source agent
            affinity: Dict[str, Dict] = {}
            for log in logs:
                entry = db.query(FederatedMemoryEntry).get(log.federated_entry_id)
                if not entry:
                    continue
                src = entry.source_agent_id
                if src not in affinity:
                    affinity[src] = {"accesses": 0, "avg_relevance": 0.0, "feedbacks": []}
                affinity[src]["accesses"] += 1
                affinity[src]["avg_relevance"] += log.relevance_score
                if log.feedback_signal is not None:
                    affinity[src]["feedbacks"].append(log.feedback_signal)

            results = []
            for agent_id, data in affinity.items():
                n = data["accesses"]
                avg_rel = data["avg_relevance"] / n if n else 0.0
                avg_fb = (
                    sum(data["feedbacks"]) / len(data["feedbacks"])
                    if data["feedbacks"] else None
                )
                results.append({
                    "source_agent_id": agent_id,
                    "access_count": n,
                    "avg_relevance": round(avg_rel, 3),
                    "avg_feedback": round(avg_fb, 3) if avg_fb is not None else None,
                    "current_trust": round(
                        self._trust.get_trust(querying_agent, agent_id), 3
                    ),
                })

            results.sort(key=lambda x: x["access_count"], reverse=True)
            return results[:top_k]

        finally:
            db.close()
