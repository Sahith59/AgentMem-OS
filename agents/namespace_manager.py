"""
AgentMem OS — Agent Namespace Manager (Phase 4)
================================================
Manages the full lifecycle of agent namespaces: create, list, fork, merge, archive.

The key concept here is MEMORY FORKING: when you fork an agent, the new child
agent inherits the parent's semantic knowledge (L2/L3 summaries and procedural
patterns) but starts with a clean episodic slate. Like a git branch — the child
shares history up to the fork point, then diverges.

Why this matters for research:
  - Enables specialization: fork a "general coding assistant" into a
    "Python expert" that inherits all Python knowledge but builds its own patterns.
  - Enables A/B testing: fork an agent, modify one, compare memory evolution.
  - Enables continuity: when a domain expert agent is retired, fork its knowledge
    into a new agent rather than losing years of accumulated memory.

Novel aspects vs. prior work:
  - MemGPT, LangChain, ChatGPT Memory: none support agent memory forking.
  - This is the first formalization of git-style memory branching for LLM agents.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from loguru import logger

from memnai.db.models import (
    AgentNamespace, Session, Summary, ProceduralPattern,
    AgentTrustScore, AgentForkRecord
)


class AgentNamespaceManager:
    """
    Full lifecycle management for agent namespaces.

    Design:
      - get_db_session injected (dependency injection pattern from Phase 3)
      - All operations are transactional — partial failures roll back
      - Fork is atomic: all inherited memories copied or none

    Usage:
        manager = AgentNamespaceManager(get_session)
        manager.create_agent("python-bot", name="Python Expert")
        manager.fork_agent(parent_id="coding-bot", child_id="python-bot")
        manager.list_agents()
    """

    def __init__(self, get_db_session):
        self._get_db = get_db_session

    # ──────────────────────────────────────────────────────────────────────────
    # CREATE
    # ──────────────────────────────────────────────────────────────────────────

    def create_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> AgentNamespace:
        """
        Create a new agent namespace with a clean memory state.

        Args:
            agent_id: Unique slug identifier (e.g. "python-bot", "customer-support-v2").
            name: Human-readable display name.
            system_prompt: The agent's base system prompt (stored in metadata).
            metadata: Arbitrary config dict (model preferences, temperature, etc.).

        Returns:
            The created AgentNamespace ORM object.

        Raises:
            ValueError: If agent_id already exists.
        """
        db = self._get_db()
        try:
            existing = db.query(AgentNamespace).filter_by(agent_id=agent_id).first()
            if existing:
                raise ValueError(f"Agent '{agent_id}' already exists.")

            meta = metadata or {}
            if system_prompt:
                meta["system_prompt"] = system_prompt

            agent = AgentNamespace(
                agent_id=agent_id,
                name=name or agent_id,
                metadata_=meta,
                created_at=datetime.utcnow(),
            )
            db.add(agent)
            db.commit()
            logger.info(f"[NamespaceManager] Created agent '{agent_id}'.")
            return agent

        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # FORK (the novel operation)
    # ──────────────────────────────────────────────────────────────────────────

    def fork_agent(
        self,
        parent_agent_id: str,
        child_agent_id: str,
        child_name: Optional[str] = None,
        inherit_levels: List[int] = None,
        trust_network=None,
    ) -> Dict[str, Any]:
        """
        Fork an existing agent: the child inherits the parent's semantic knowledge.

        WHAT GETS INHERITED:
          - summaries with abstraction_level in inherit_levels (default: [2, 3])
            These are patterns and principles — not raw episodes.
          - ProceduralPatterns with confidence >= 0.7 from the parent agent.

        WHAT DOES NOT GET INHERITED (by design):
          - Raw Turn records — episodic memory stays private.
          - Sessions — child starts with no conversation history.
          - Trust scores — child builds its own reputation from scratch
            (except: child automatically trusts parent at 0.9).

        WHY THIS IS NOVEL:
          This is the first formalization of memory branching for LLM agents.
          The concept mirrors git's branch-from-commit model:
            - Parent's semantic knowledge = the commit history
            - Fork = new branch from that commit
            - Child's episodes = new commits after the branch

        Args:
            parent_agent_id: ID of the agent to fork from.
            child_agent_id: ID for the new child agent.
            child_name: Optional display name.
            inherit_levels: Which abstraction levels to copy (default [2, 3]).
            trust_network: Optional AgentTrustNetwork to initialize trust scores.

        Returns:
            Dict with fork_record and counts of inherited memories.
        """
        if inherit_levels is None:
            inherit_levels = [2, 3]

        db = self._get_db()
        try:
            # Validate parent exists
            parent = db.query(AgentNamespace).filter_by(agent_id=parent_agent_id).first()
            if not parent:
                raise ValueError(f"Parent agent '{parent_agent_id}' not found.")

            # Create child namespace (idempotent — use existing if already created)
            child = db.query(AgentNamespace).filter_by(agent_id=child_agent_id).first()
            if not child:
                child = AgentNamespace(
                    agent_id=child_agent_id,
                    name=child_name or f"{child_agent_id} (fork of {parent_agent_id})",
                    metadata_={
                        "forked_from": parent_agent_id,
                        "fork_timestamp": datetime.utcnow().isoformat(),
                    },
                    created_at=datetime.utcnow(),
                )
                db.add(child)
                db.flush()

            # Inherit summaries (L2/L3 only — privacy preserving)
            parent_summaries = (
                db.query(Summary)
                .join(Session, Summary.session_id == Session.session_id)
                .filter(
                    Session.agent_id == parent_agent_id,
                    Summary.abstraction_level.in_(inherit_levels),
                )
                .all()
            )

            inherited_summaries = 0
            for s in parent_summaries:
                # Create a copy tagged with the child's context
                child_summary = Summary(
                    session_id=s.session_id,        # preserves provenance
                    agent_id=child_agent_id,        # re-tagged for child
                    turn_range=s.turn_range,
                    content=f"[INHERITED from {parent_agent_id}] {s.content}",
                    entities=s.entities,
                    cluster_id=s.cluster_id,
                    abstraction_level=s.abstraction_level,
                    is_shared=False,
                    created_at=datetime.utcnow(),
                )
                db.add(child_summary)
                inherited_summaries += 1

            # Inherit high-confidence procedural patterns
            parent_patterns = (
                db.query(ProceduralPattern)
                .filter(
                    ProceduralPattern.agent_id == parent_agent_id,
                    ProceduralPattern.confidence >= 0.7,
                )
                .all()
            )

            inherited_patterns = 0
            for p in parent_patterns:
                child_pattern = ProceduralPattern(
                    agent_id=child_agent_id,
                    trigger=p.trigger,
                    action=p.action,
                    full_pattern=f"[INHERITED] {p.full_pattern}",
                    confidence=p.confidence * 0.85,  # slight confidence decay on inheritance
                    support_count=max(1, p.support_count // 2),
                    source_sessions=p.source_sessions,
                    created_at=datetime.utcnow(),
                    is_global=False,
                )
                db.add(child_pattern)
                inherited_patterns += 1

            # Compute fork depth
            parent_fork = (
                db.query(AgentForkRecord)
                .filter_by(child_agent_id=parent_agent_id)
                .first()
            )
            fork_depth = (parent_fork.fork_depth + 1) if parent_fork else 1

            # Record the fork relationship
            fork_record = AgentForkRecord(
                parent_agent_id=parent_agent_id,
                child_agent_id=child_agent_id,
                fork_depth=fork_depth,
                summaries_inherited=inherited_summaries,
                patterns_inherited=inherited_patterns,
                forked_at=datetime.utcnow(),
            )
            db.add(fork_record)
            db.commit()

            logger.info(
                f"[NamespaceManager] Forked '{parent_agent_id}' → '{child_agent_id}' | "
                f"summaries={inherited_summaries}, patterns={inherited_patterns}, depth={fork_depth}"
            )

            # Bootstrap trust: child inherits trust in parent at 0.9
            if trust_network:
                trust_network.set_trust(child_agent_id, parent_agent_id, 0.9)
                trust_network.set_trust(parent_agent_id, child_agent_id, 0.5)

            return {
                "parent_agent_id": parent_agent_id,
                "child_agent_id": child_agent_id,
                "fork_depth": fork_depth,
                "summaries_inherited": inherited_summaries,
                "patterns_inherited": inherited_patterns,
            }

        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # READ
    # ──────────────────────────────────────────────────────────────────────────

    def get_agent(self, agent_id: str) -> Optional[AgentNamespace]:
        """Retrieve an agent namespace by ID."""
        db = self._get_db()
        try:
            return db.query(AgentNamespace).filter_by(agent_id=agent_id).first()
        finally:
            db.close()

    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all agents with summary statistics.

        Returns:
            List of dicts with keys: agent_id, name, session_count,
            summary_count, pattern_count, created_at.
        """
        db = self._get_db()
        try:
            agents = db.query(AgentNamespace).all()
            result = []
            for a in agents:
                session_count = (
                    db.query(Session)
                    .filter_by(agent_id=a.agent_id)
                    .count()
                )
                summary_count = (
                    db.query(Summary)
                    .filter_by(agent_id=a.agent_id)
                    .count()
                )
                pattern_count = (
                    db.query(ProceduralPattern)
                    .filter_by(agent_id=a.agent_id)
                    .count()
                )
                result.append({
                    "agent_id": a.agent_id,
                    "name": a.name,
                    "session_count": session_count,
                    "summary_count": summary_count,
                    "pattern_count": pattern_count,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                })
            return result
        finally:
            db.close()

    def get_fork_lineage(self, agent_id: str) -> Dict[str, Any]:
        """
        Return the full fork lineage for an agent: ancestors and descendants.

        Returns:
            Dict with 'ancestors' (list of agent_ids) and 'descendants'.
        """
        db = self._get_db()
        try:
            ancestors = []
            current = agent_id
            seen = set()
            while True:
                record = (
                    db.query(AgentForkRecord)
                    .filter_by(child_agent_id=current)
                    .first()
                )
                if not record or record.parent_agent_id in seen:
                    break
                ancestors.append(record.parent_agent_id)
                seen.add(record.parent_agent_id)
                current = record.parent_agent_id

            descendants_records = (
                db.query(AgentForkRecord)
                .filter_by(parent_agent_id=agent_id)
                .all()
            )
            descendants = [r.child_agent_id for r in descendants_records]

            return {
                "agent_id": agent_id,
                "ancestors": ancestors,
                "descendants": descendants,
                "fork_depth": len(ancestors),
            }
        finally:
            db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # MERGE (semantic merge of two agents' pattern pools)
    # ──────────────────────────────────────────────────────────────────────────

    def merge_patterns(
        self,
        source_agent_id: str,
        target_agent_id: str,
        confidence_threshold: float = 0.75,
    ) -> int:
        """
        Merge high-confidence patterns from source into target agent.

        Unlike fork (which creates a new agent), merge augments an existing
        agent's pattern library with another agent's learnings. Useful for:
          - Merging a specialized agent's patterns into the general agent
          - Aggregating learnings from multiple domain agents

        Conflict resolution: if target already has the same (trigger, action)
        pair, we keep the higher confidence value.

        Returns:
            Number of patterns merged.
        """
        db = self._get_db()
        try:
            source_patterns = (
                db.query(ProceduralPattern)
                .filter(
                    ProceduralPattern.agent_id == source_agent_id,
                    ProceduralPattern.confidence >= confidence_threshold,
                )
                .all()
            )

            merged = 0
            for sp in source_patterns:
                # Check for conflict
                existing = (
                    db.query(ProceduralPattern)
                    .filter_by(
                        agent_id=target_agent_id,
                        trigger=sp.trigger,
                        action=sp.action,
                    )
                    .first()
                )

                if existing:
                    # Keep higher confidence
                    if sp.confidence > existing.confidence:
                        existing.confidence = sp.confidence
                        existing.support_count += sp.support_count
                    merged += 1
                else:
                    new_pattern = ProceduralPattern(
                        agent_id=target_agent_id,
                        trigger=sp.trigger,
                        action=sp.action,
                        full_pattern=f"[MERGED from {source_agent_id}] {sp.full_pattern}",
                        confidence=sp.confidence * 0.9,  # slight decay on merge
                        support_count=sp.support_count,
                        created_at=datetime.utcnow(),
                        is_global=False,
                    )
                    db.add(new_pattern)
                    merged += 1

            db.commit()
            logger.info(
                f"[NamespaceManager] Merged {merged} patterns: "
                f"'{source_agent_id}' → '{target_agent_id}'"
            )
            return merged

        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # DELETE / ARCHIVE
    # ──────────────────────────────────────────────────────────────────────────

    def ensure_agent_exists(self, agent_id: str) -> AgentNamespace:
        """
        Get or create an agent namespace. Idempotent — safe to call on every request.
        Used by store.py and context_assembler.py to ensure the default agent exists.
        """
        db = self._get_db()
        try:
            agent = db.query(AgentNamespace).filter_by(agent_id=agent_id).first()
            if not agent:
                agent = AgentNamespace(
                    agent_id=agent_id,
                    name=agent_id,
                    metadata_={},
                    created_at=datetime.utcnow(),
                )
                db.add(agent)
                db.commit()
                logger.debug(f"[NamespaceManager] Auto-created agent '{agent_id}'.")
            return agent
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
