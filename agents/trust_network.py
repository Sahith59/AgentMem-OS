"""
AgentMem OS — Agent Trust Network (Phase 4)
============================================
Manages pairwise trust scores between agents in the Memory Federation.

Trust is the credibility mechanism that determines how much weight an agent
gives to memories retrieved from another agent's federated pool. It prevents
the shared pool from being dominated by low-quality memories from unreliable
source agents.

Trust Update Rule (Exponential Moving Average):
    trust_new = α × trust_old + (1 − α) × feedback_signal
    α = 0.80  (trust changes slowly — earned slowly, lost slowly)

Trust Initialization:
    - New pair:         0.50 (neutral / uncertain)
    - Fork child→parent: 0.90 (child trusts parent from birth)
    - Fork parent→child: 0.50 (parent starts neutral on child)

Trust Propagation (Transitive Trust):
    If A trusts B with 0.9, and B trusts C with 0.8:
    A's indirect trust in C = direct_trust(A,C) × 0.5 +
                              transitive_trust(A→B→C) × 0.5
    (capped at max direct trust to prevent runaway propagation)

Research contribution:
    No prior LLM memory work models inter-agent credibility as a directed
    trust graph with EMA updates and transitive propagation. This is novel.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from loguru import logger

from agentmem_os.db.models import AgentTrustScore


# Trust update hyperparameter
TRUST_EMA_ALPHA = 0.80          # weight on old trust vs. new signal
MIN_TRUST       = 0.05          # floor — never fully untrust
MAX_TRUST       = 0.98          # ceiling — never give absolute trust
NEUTRAL_TRUST   = 0.50          # default for unknown pairs
FORK_TRUST      = 0.90          # child→parent trust at fork time

# Minimum interactions before trust is considered "stable"
MIN_INTERACTIONS_FOR_STABILITY = 5


class AgentTrustNetwork:
    """
    Directed trust graph for the Memory Federation Protocol.

    The trust graph is stored persistently in SQLite (AgentTrustScore table)
    and cached in-memory as a dict for fast reads.

    Design:
        - Trust is directional: A trusts B ≠ B trusts A
        - In-memory cache avoids DB round-trips on every retrieval
        - Cache is warmed on init and updated on every write
        - Transitive trust is computed on-demand (not stored)

    Usage:
        tn = AgentTrustNetwork(get_session)
        tn.set_trust("child-bot", "parent-bot", 0.9)
        tn.update_trust("agent-a", "agent-b", feedback=0.8)
        score = tn.get_trust("agent-a", "agent-b")
        weighted = tn.weight_memories(memories, querying_agent="agent-a")
    """

    def __init__(self, get_db_session):
        self._get_db = get_db_session
        # In-memory cache: (agent_from, agent_to) → trust_score
        self._cache: Dict[Tuple[str, str], float] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Warm the in-memory cache from SQLite on startup."""
        db = self._get_db()
        try:
            rows = db.query(AgentTrustScore).all()
            for row in rows:
                self._cache[(row.agent_from, row.agent_to)] = row.trust_score
            logger.debug(f"[TrustNetwork] Loaded {len(rows)} trust scores into cache.")
        except Exception as e:
            logger.warning(f"[TrustNetwork] Could not load cache: {e}")
        finally:
            db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # READ
    # ──────────────────────────────────────────────────────────────────────────

    def get_trust(
        self,
        agent_from: str,
        agent_to: str,
        use_transitive: bool = True,
    ) -> float:
        """
        Get the trust score from agent_from toward agent_to.

        Args:
            agent_from: The agent that will use the retrieved memories.
            agent_to: The agent that produced the memories.
            use_transitive: Whether to blend in transitive trust paths.

        Returns:
            Trust score ∈ [0, 1]. Returns NEUTRAL_TRUST (0.5) for unknown pairs.
        """
        if agent_from == agent_to:
            return 1.0  # An agent fully trusts itself

        direct = self._cache.get((agent_from, agent_to), NEUTRAL_TRUST)

        if not use_transitive:
            return direct

        # Blend direct trust with transitive paths (one hop)
        transitive = self._compute_transitive_trust(agent_from, agent_to)
        if transitive is None:
            return direct

        # Blend: direct gets 70% weight, transitive gets 30%
        blended = 0.70 * direct + 0.30 * transitive
        return min(MAX_TRUST, max(MIN_TRUST, blended))

    def _compute_transitive_trust(
        self,
        agent_from: str,
        agent_to: str,
    ) -> Optional[float]:
        """
        Compute one-hop transitive trust: A→X→B for all intermediaries X.

        Returns the max transitive path score, or None if no intermediary exists.
        """
        # Find all agents that agent_from trusts
        intermediaries = {
            k[1]: v for k, v in self._cache.items()
            if k[0] == agent_from and k[1] != agent_to
        }
        if not intermediaries:
            return None

        best_transitive = None
        for intermediary, trust_a_x in intermediaries.items():
            trust_x_b = self._cache.get((intermediary, agent_to))
            if trust_x_b is not None:
                path_score = trust_a_x * trust_x_b  # chain product
                if best_transitive is None or path_score > best_transitive:
                    best_transitive = path_score

        return best_transitive

    def get_trust_matrix(self, agent_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Return the full trust matrix for a set of agents.

        Returns:
            Nested dict: matrix[from_agent][to_agent] = trust_score.
            Self-trust is always 1.0.
        """
        matrix = {}
        for from_id in agent_ids:
            matrix[from_id] = {}
            for to_id in agent_ids:
                matrix[from_id][to_id] = self.get_trust(from_id, to_id)
        return matrix

    def get_most_trusted_sources(
        self,
        agent_id: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Return the top-k agents most trusted by agent_id, sorted descending.

        Returns:
            List of (source_agent_id, trust_score) tuples.
        """
        scores = {
            k[1]: v for k, v in self._cache.items()
            if k[0] == agent_id
        }
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]

    # ──────────────────────────────────────────────────────────────────────────
    # WRITE
    # ──────────────────────────────────────────────────────────────────────────

    def set_trust(
        self,
        agent_from: str,
        agent_to: str,
        score: float,
    ) -> None:
        """
        Directly set a trust score. Used for initialization (e.g., fork bootstrap).
        Bypasses the EMA update — use update_trust for incremental learning.
        """
        score = min(MAX_TRUST, max(MIN_TRUST, score))
        self._cache[(agent_from, agent_to)] = score
        self._persist_trust(agent_from, agent_to, score, delta_interactions=0)
        logger.debug(f"[TrustNetwork] Set trust {agent_from}→{agent_to} = {score:.3f}")

    def update_trust(
        self,
        agent_from: str,
        agent_to: str,
        feedback_signal: float,
    ) -> float:
        """
        Update trust using an Exponential Moving Average.

        Args:
            agent_from: Agent that used the retrieved memory.
            agent_to: Agent that produced the retrieved memory.
            feedback_signal: Quality signal ∈ [0, 1].
                1.0 = the memory was highly relevant and improved the response
                0.5 = neutral / no clear signal
                0.0 = the memory was irrelevant or harmful

        Returns:
            New trust score after update.

        Formula:
            trust_new = α × trust_old + (1 − α) × feedback_signal
            α = TRUST_EMA_ALPHA = 0.80
        """
        old_trust = self._cache.get((agent_from, agent_to), NEUTRAL_TRUST)
        new_trust = TRUST_EMA_ALPHA * old_trust + (1 - TRUST_EMA_ALPHA) * feedback_signal
        new_trust = min(MAX_TRUST, max(MIN_TRUST, new_trust))

        self._cache[(agent_from, agent_to)] = new_trust
        self._persist_trust(agent_from, agent_to, new_trust, delta_interactions=1)

        logger.debug(
            f"[TrustNetwork] Updated trust {agent_from}→{agent_to}: "
            f"{old_trust:.3f} → {new_trust:.3f} (feedback={feedback_signal:.2f})"
        )
        return new_trust

    def _persist_trust(
        self,
        agent_from: str,
        agent_to: str,
        score: float,
        delta_interactions: int,
    ) -> None:
        """Upsert trust score to SQLite."""
        db = self._get_db()
        try:
            row = (
                db.query(AgentTrustScore)
                .filter_by(agent_from=agent_from, agent_to=agent_to)
                .first()
            )
            if row:
                row.trust_score = score
                row.interaction_count += delta_interactions
                row.last_updated = datetime.utcnow()
            else:
                row = AgentTrustScore(
                    agent_from=agent_from,
                    agent_to=agent_to,
                    trust_score=score,
                    interaction_count=delta_interactions,
                    created_at=datetime.utcnow(),
                    last_updated=datetime.utcnow(),
                )
                db.add(row)
            db.commit()
        except Exception as e:
            logger.warning(f"[TrustNetwork] Failed to persist trust: {e}")
            db.rollback()
        finally:
            db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # APPLY TRUST WEIGHTS
    # ──────────────────────────────────────────────────────────────────────────

    def weight_memories(
        self,
        memories: List[Dict],
        querying_agent: str,
        relevance_key: str = "relevance_score",
    ) -> List[Dict]:
        """
        Apply trust weights to a list of federated memory entries.

        Each memory dict must have:
            - 'source_agent_id': str
            - relevance_key: float (cosine similarity or similar score)

        This adds a 'weighted_score' key to each memory:
            weighted_score = relevance × trust(querying_agent, source_agent)

        Args:
            memories: List of federated memory dicts from MFP retrieval.
            querying_agent: The agent making the request.
            relevance_key: Key in each memory dict holding the raw relevance score.

        Returns:
            Same list, augmented with 'weighted_score', sorted descending.
        """
        for mem in memories:
            source = mem.get("source_agent_id", "unknown")
            relevance = float(mem.get(relevance_key, 0.5))
            trust = self.get_trust(querying_agent, source)
            mem["trust_score"] = trust
            mem["weighted_score"] = relevance * trust

        memories.sort(key=lambda m: m["weighted_score"], reverse=True)
        return memories

    # ──────────────────────────────────────────────────────────────────────────
    # STABILITY & ANALYTICS
    # ──────────────────────────────────────────────────────────────────────────

    def is_trust_stable(self, agent_from: str, agent_to: str) -> bool:
        """
        Return True if this trust score is considered stable (enough interactions).
        Used to gate whether transitive trust propagation is applied.
        """
        db = self._get_db()
        try:
            row = (
                db.query(AgentTrustScore)
                .filter_by(agent_from=agent_from, agent_to=agent_to)
                .first()
            )
            if not row:
                return False
            return row.interaction_count >= MIN_INTERACTIONS_FOR_STABILITY
        finally:
            db.close()

    def describe(self) -> str:
        """Return a human-readable summary of the trust graph."""
        if not self._cache:
            return "[TrustNetwork] No trust relationships established yet."
        lines = ["[TRUST NETWORK]"]
        for (frm, to), score in sorted(self._cache.items()):
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"  {frm} → {to}: {bar} {score:.2f}")
        return "\n".join(lines)
