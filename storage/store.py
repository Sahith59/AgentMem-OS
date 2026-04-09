"""
AgentMem OS — ConversationStore (v2)
======================================
The central orchestrator of all 4 memory tiers.

On every save_turn():
  → Tier 2: Persists to SQLite (episodic)
  → Tier 2: Updates Redis cache (working memory L1)
  → KG:     Ingests entities into Knowledge Graph
  → Trigger: Background importance scoring + threshold check

On compression (background, sleep engine):
  → Importance Scorer selects lowest-value turns
  → DBSCAN clusters them semantically
  → SleepConsolidationEngine generates one summary per cluster
  → Summaries written to Tier 3 (SQLite + ChromaDB)
  → Compressed turns deleted from Tier 2

On get_history():
  → Redis first (L1 cache), fallback to SQLite

On create_branch():
  → Compresses all parent turns → snapshot
  → Child session inherits snapshot as Tier 3 seed
"""

import threading
from typing import List, Optional

from loguru import logger

from memnai.db.models import Session, Turn
from memnai.db.engine import get_session
from memnai.llm.token_counter import TokenCounter


class ConversationStore:
    """
    Unified memory store across all tiers.

    Initialization is lazy — heavy deps (spaCy, ChromaDB) load on first use.
    """

    CONTEXT_WINDOW   = 128_000      # tokens
    COMPRESS_THRESH  = 0.70         # compress when > 70% full
    COMPRESS_FRAC    = 0.30         # compress bottom 30% by importance

    def __init__(self):
        self.db = get_session()
        self.token_counter = TokenCounter()
        self._redis   = None    # lazy
        self._summarizer  = None    # lazy
        self._chroma  = None    # lazy
        self._scorer  = None    # lazy
        self._engine  = None    # lazy (SleepConsolidationEngine)
        self._kg      = None    # lazy (EntityKnowledgeGraph)
        self._proc    = None    # lazy (ProceduralMemory)

    # ──────────────────────────────────────────────────────────────────────────
    # Session Management
    # ──────────────────────────────────────────────────────────────────────────

    def get_or_create_session(
        self,
        session_id: str,
        name: str = None,
        model: str = "ollama/llama3.2:3b",
        agent_id: str = None,
    ) -> Session:
        session = self.db.query(Session).filter(
            Session.session_id == session_id
        ).first()

        if not session:
            session = Session(
                session_id=session_id,
                name=name or session_id,
                model=model,
                branch_type="root",
                agent_id=agent_id,
            )
            self.db.add(session)
            self.db.commit()

        return session

    # ──────────────────────────────────────────────────────────────────────────
    # Turn Storage  (Tier 2: Episodic)
    # ──────────────────────────────────────────────────────────────────────────

    def save_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        agent_id: str = None,
    ):
        """
        Persist a conversation turn across all relevant tiers.

        1. Count tokens
        2. Save to SQLite (Tier 2)
        3. Cache in Redis (Tier 1)
        4. Update Entity Knowledge Graph (asynchronously)
        5. Trigger background compression check
        """
        session = self.get_or_create_session(session_id, agent_id=agent_id)
        tokens = self.token_counter.count(content)

        turn = Turn(
            session_id=session_id,
            role=role,
            content=content,
            token_count=tokens,
        )
        self.db.add(turn)
        session.total_tokens += tokens
        self.db.commit()

        # Redis L1 cache
        try:
            redis = self._get_redis()
            redis.push_turn(session_id, {
                "role": role,
                "content": content,
                "token_count": tokens,
            })
        except Exception as e:
            logger.debug(f"[Store] Redis cache push skipped: {e}")

        # Knowledge Graph ingestion (background — non-blocking)
        threading.Thread(
            target=self._ingest_kg,
            args=(session_id, agent_id, content),
            daemon=True,
        ).start()

        # Compression check (background — non-blocking)
        threading.Thread(
            target=self._check_and_compress,
            args=(session_id,),
            daemon=True,
        ).start()

    # ──────────────────────────────────────────────────────────────────────────
    # History Retrieval  (Tier 2: Episodic, via Redis cache)
    # ──────────────────────────────────────────────────────────────────────────

    def get_history(self, session_id: str, last_n: int = 20) -> List[dict]:
        """
        Return the most recent `last_n` turns.
        Checks Redis (L1) first; falls back to SQLite.
        """
        try:
            redis = self._get_redis()
            cached = redis.get_history(session_id)
            if cached:
                return cached[-last_n:]
        except Exception:
            pass

        turns = (
            self.db.query(Turn)
            .filter(Turn.session_id == session_id)
            .order_by(Turn.id.desc())
            .limit(last_n)
            .all()
        )

        if not turns:
            return []

        result = [
            {"role": t.role, "content": t.content, "token_count": t.token_count}
            for t in reversed(turns)
        ]

        # Repopulate Redis
        try:
            redis = self._get_redis()
            for t in result:
                redis.push_turn(session_id, t)
        except Exception:
            pass

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Branching
    # ──────────────────────────────────────────────────────────────────────────

    def create_branch(
        self,
        parent_id: str,
        new_name: str,
        branch_type: str = "hard",
    ) -> Session:
        """
        Create a child session branching from parent_id.
        Snapshot of parent's full conversation is compressed and inherited.
        """
        parent = self.db.query(Session).filter(
            Session.session_id == parent_id
        ).first()
        if not parent:
            raise ValueError(f"Parent session '{parent_id}' not found.")

        latest_turn = (
            self.db.query(Turn)
            .filter(Turn.session_id == parent_id)
            .order_by(Turn.id.desc())
            .first()
        )
        branch_point = latest_turn.id if latest_turn else 0

        # Compress all parent turns into an inherited snapshot
        parent_turns = (
            self.db.query(Turn)
            .filter(Turn.session_id == parent_id)
            .order_by(Turn.id.asc())
            .all()
        )

        if parent_turns:
            raw = [{"role": t.role, "content": t.content} for t in parent_turns]
            try:
                summarizer = self._get_summarizer()
                inherited_state, _ = summarizer.compress(raw)
            except Exception:
                inherited_state = (
                    f"[Snapshot of {len(parent_turns)} turns from session '{parent_id}']\n"
                    + " | ".join(
                        f"{t.role}: {t.content[:80]}" for t in parent_turns[-5:]
                    )
                )
        else:
            inherited_state = f"[Empty parent session: {parent_id}]"

        child_id = f"{parent_id}/{new_name}"
        child = Session(
            session_id=child_id,
            parent_session_id=parent_id,
            branch_point_turn=branch_point,
            name=new_name,
            model=parent.model,
            branch_type=branch_type,
            inherited_context=inherited_state,
            agent_id=parent.agent_id,
        )
        self.db.add(child)
        self.db.commit()

        logger.info(
            f"[Store] Branch created: {parent_id} → {child_id} "
            f"at turn {branch_point}"
        )
        return child

    def list_branches(self, root_id: str) -> List[Session]:
        return (
            self.db.query(Session)
            .filter(
                (Session.session_id == root_id) |
                (Session.parent_session_id == root_id)
            )
            .all()
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Background: Compression Check
    # ──────────────────────────────────────────────────────────────────────────

    def _check_and_compress(self, session_id: str):
        """
        Background thread: check if compression is needed and run if so.
        Uses SleepConsolidationEngine (DBSCAN + importance scoring).
        """
        db = get_session()
        try:
            from memnai.db.models import Session  # noqa: F811
            session = db.query(Session).filter(
                Session.session_id == session_id
            ).first()
            if not session:
                return

            threshold = self.CONTEXT_WINDOW * self.COMPRESS_THRESH
            if session.total_tokens > threshold:
                engine = self._get_engine()
                report = engine.consolidate(session_id, compress_fraction=self.COMPRESS_FRAC)
                if not report.get("skipped"):
                    logger.info(
                        f"[Store] Auto-compression complete for {session_id}: "
                        f"{report.get('tokens_freed', 0)} tokens freed, "
                        f"{report.get('clusters', 0)} clusters."
                    )
        except Exception as e:
            logger.warning(f"[Store] _check_and_compress error: {e}")
        finally:
            db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Background: Knowledge Graph Ingestion
    # ──────────────────────────────────────────────────────────────────────────

    def _ingest_kg(self, session_id: str, agent_id: Optional[str], content: str):
        """Background thread: extract entities and update the knowledge graph."""
        try:
            kg = self._get_kg()
            count = kg.ingest_turn(session_id, agent_id, content)
            if count:
                logger.debug(
                    f"[Store] KG: +{count} entities from session={session_id}"
                )
        except Exception as e:
            logger.debug(f"[Store] KG ingestion skipped: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Lazy Dependency Getters
    # ──────────────────────────────────────────────────────────────────────────

    def _get_redis(self):
        if self._redis is None:
            from memnai.cache.redis_client import RedisCache
            self._redis = RedisCache()
        return self._redis

    def _get_summarizer(self):
        if self._summarizer is None:
            from memnai.llm.summarizer import SummarizationEngine
            self._summarizer = SummarizationEngine()
        return self._summarizer

    def _get_chroma(self):
        if self._chroma is None:
            from memnai.db.chroma_client import ChromaManager
            self._chroma = ChromaManager()
        return self._chroma

    def _get_scorer(self):
        if self._scorer is None:
            from memnai.llm.importance_scorer import MemoryImportanceScorer
            self._scorer = MemoryImportanceScorer()
        return self._scorer

    def _get_engine(self):
        if self._engine is None:
            from memnai.llm.consolidation_engine import SleepConsolidationEngine
            self._engine = SleepConsolidationEngine(
                get_db_session=get_session,
                summarizer=self._get_summarizer(),
                chroma=self._get_chroma(),
                importance_scorer=self._get_scorer(),
            )
        return self._engine

    def _get_kg(self):
        if self._kg is None:
            from memnai.db.knowledge_graph import EntityKnowledgeGraph
            self._kg = EntityKnowledgeGraph(get_session)
        return self._kg

    def _get_proc(self):
        if self._proc is None:
            from memnai.llm.procedural_memory import ProceduralMemory
            self._proc = ProceduralMemory(get_session, self._get_summarizer())
        return self._proc

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def close(self):
        self.db.close()
        try:
            if self._redis:
                self._redis.persist()
        except Exception:
            pass
