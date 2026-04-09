"""
AgentMem OS — Sleep Consolidation Engine
==========================================
Novel Algorithm #2: Background memory consolidation inspired by biological
sleep-based memory consolidation in the hippocampus-neocortex system.

During biological sleep, the hippocampus replays episodic memories and
selectively transfers them to neocortical long-term storage as semantic
abstractions. Individual episodes merge into generalized knowledge.

This engine replicates that process for LLM agents:

  1. DBSCAN Clustering  — groups semantically similar episodic turns
  2. Cluster Abstraction — generates a higher-order summary per cluster
  3. Abstraction Levels  — tracks whether a summary is Episode (L1),
                           Pattern (L2), or Principle (L3)
  4. Promotion           — mature patterns can be promoted to shared pool

Key difference from naive compression:
  - Naive: compresses oldest N% of turns (loses semantic structure)
  - This:  clusters by meaning, then abstracts each cluster (preserves structure)

Research contribution:
  First implementation of biologically-inspired sleep consolidation for
  LLM agent memory. Introduces abstraction_level to track memory maturity.
"""

import time
import threading
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
from loguru import logger


class SleepConsolidationEngine:
    """
    Background consolidation engine that runs on a schedule or on-demand.

    Lifecycle:
        engine = SleepConsolidationEngine(db_session, summarizer, chroma, scorer)
        engine.start_background_scheduler(interval_seconds=300)  # every 5 minutes
        # Or manually:
        report = engine.consolidate(session_id)
    """

    def __init__(
        self,
        get_db_session,          # callable: () -> SQLAlchemy session
        summarizer,              # SummarizationEngine instance
        chroma,                  # ChromaManager instance
        importance_scorer,       # MemoryImportanceScorer instance
        get_embedding_fn=None,   # callable: str -> np.ndarray (for clustering)
    ):
        self.get_db = get_db_session
        self.summarizer = summarizer
        self.chroma = chroma
        self.scorer = importance_scorer
        self.get_embedding_fn = get_embedding_fn
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def consolidate(
        self,
        session_id: str,
        compress_fraction: float = 0.30,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Run one full consolidation cycle for a session.

        Steps:
          1. Score all turns for importance
          2. Select bottom `compress_fraction` for compression
          3. Cluster selected turns by semantic similarity (DBSCAN)
          4. Generate one summary per cluster (not one for all)
          5. Store summaries in DB + ChromaDB
          6. Delete compressed turns
          7. Log consolidation metrics

        Returns a report dict with metrics (for benchmarking).
        """
        start_time = time.time()
        db = self.get_db()

        try:
            from memnai.db.models import Turn, Summary, Session, ConsolidationLog

            session = db.query(Session).filter(Session.session_id == session_id).first()
            if not session:
                return {"error": f"Session {session_id} not found"}

            # Check threshold unless forced
            if not force:
                threshold = 128_000 * 0.70   # 70% of 128k context window
                if session.total_tokens < threshold:
                    return {
                        "skipped": True,
                        "reason": f"tokens ({session.total_tokens}) below threshold ({threshold:.0f})"
                    }

            # Load all non-compressed turns ordered by id
            turns_orm = (
                db.query(Turn)
                .filter(Turn.session_id == session_id, Turn.is_compressed == False)
                .order_by(Turn.id.asc())
                .all()
            )

            if len(turns_orm) < 4:
                return {"skipped": True, "reason": "too few turns to consolidate"}

            turns_dicts = [
                {"id": t.id, "role": t.role, "content": t.content,
                 "token_count": t.token_count, "importance_score": t.importance_score}
                for t in turns_orm
            ]

            tokens_before = session.total_tokens

            # ── Step 1-2: Importance scoring → select candidates ──────────────
            to_compress, _ = self.scorer.get_compression_candidates(
                turns_dicts,
                compress_fraction=compress_fraction,
                get_embedding_fn=self.get_embedding_fn,
            )

            if not to_compress:
                return {"skipped": True, "reason": "no compression candidates found"}

            # ── Step 3: Cluster candidates by semantic similarity ─────────────
            clusters = self._cluster_turns(to_compress)
            logger.info(
                f"[ConsolidationEngine] Session={session_id} | "
                f"{len(to_compress)} turns → {len(clusters)} clusters"
            )

            # ── Step 4-5: Generate one summary per cluster ────────────────────
            summaries_created = 0
            total_tokens_removed = 0
            turn_ids_to_delete = set()

            for cluster_id, cluster_turns in clusters.items():
                summary_text, entities = self._generate_cluster_summary(
                    cluster_turns, cluster_id
                )

                # Determine abstraction level
                # L1 = fresh episode summary (default)
                # We'll upgrade to L2/L3 during later consolidation cycles
                abstraction_level = 1

                # Build turn_range string from IDs
                ids = sorted([t["id"] for t in cluster_turns])
                turn_range = f"{ids[0]}-{ids[-1]}"

                # Store in SQLite
                summary = Summary(
                    session_id=session_id,
                    turn_range=turn_range,
                    content=summary_text,
                    entities=",".join(entities) if entities else "",
                    cluster_id=cluster_id,
                    abstraction_level=abstraction_level,
                    is_shared=False,
                )
                db.add(summary)

                # Store in ChromaDB for semantic retrieval
                try:
                    self.chroma.add_summary_chunk(
                        session_id=session_id,
                        chunk_id=f"{session_id}_{turn_range}_{cluster_id}",
                        content=summary_text,
                        metadata={
                            "turn_range": turn_range,
                            "entities": ",".join(entities) if entities else "",
                            "cluster_id": str(cluster_id),
                            "abstraction_level": str(abstraction_level),
                        }
                    )
                except Exception as e:
                    logger.warning(f"[ConsolidationEngine] ChromaDB push failed: {e}")

                # Track deletions
                turn_ids_to_delete.update(ids)
                total_tokens_removed += sum(t.get("token_count", 0) for t in cluster_turns)
                summaries_created += 1

            # ── Step 6: Delete compressed turns, mark them ───────────────────
            for turn_id in turn_ids_to_delete:
                t = db.query(Turn).filter(Turn.id == turn_id).first()
                if t:
                    db.delete(t)

            session.total_tokens = max(0, session.total_tokens - total_tokens_removed)
            db.commit()

            tokens_after = session.total_tokens
            duration = time.time() - start_time
            compression_ratio = tokens_after / max(1, tokens_before)

            # ── Step 7: Log consolidation metrics ────────────────────────────
            log_entry = ConsolidationLog(
                session_id=session_id,
                turns_processed=len(to_compress),
                clusters_found=len(clusters),
                summaries_generated=summaries_created,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                compression_ratio=compression_ratio,
                duration_seconds=duration,
                triggered_by="threshold" if not force else "manual",
            )
            db.add(log_entry)
            db.commit()

            report = {
                "session_id": session_id,
                "turns_compressed": len(to_compress),
                "clusters": len(clusters),
                "summaries_created": summaries_created,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "compression_ratio": round(compression_ratio, 3),
                "tokens_freed": tokens_before - tokens_after,
                "duration_seconds": round(duration, 3),
            }
            logger.info(f"[ConsolidationEngine] Complete: {report}")
            return report

        except Exception as e:
            logger.error(f"[ConsolidationEngine] Error consolidating {session_id}: {e}")
            db.rollback()
            return {"error": str(e)}
        finally:
            db.close()

    def consolidate_all_sessions(self) -> List[Dict]:
        """Run consolidation on all active sessions above threshold."""
        db = self.get_db()
        try:
            from memnai.db.models import Session
            sessions = db.query(Session).filter(
                Session.is_archived == False,
                Session.total_tokens > 128_000 * 0.70
            ).all()
            session_ids = [s.session_id for s in sessions]
        finally:
            db.close()

        reports = []
        for sid in session_ids:
            report = self.consolidate(sid)
            reports.append(report)
        return reports

    # ──────────────────────────────────────────────────────────────────────────
    # Background Scheduler
    # ──────────────────────────────────────────────────────────────────────────

    def start_background_scheduler(self, interval_seconds: int = 300):
        """
        Start a daemon thread that runs consolidation every `interval_seconds`.
        This is the "sleep" — it runs while the agent is between conversations.
        """
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("[ConsolidationEngine] Scheduler already running.")
            return

        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            args=(interval_seconds,),
            daemon=True,
            name="agentmem-consolidation"
        )
        self._scheduler_thread.start()
        logger.info(
            f"[ConsolidationEngine] Background scheduler started "
            f"(interval={interval_seconds}s)."
        )

    def stop_background_scheduler(self):
        """Stop the background consolidation scheduler."""
        self._stop_event.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        logger.info("[ConsolidationEngine] Scheduler stopped.")

    def _scheduler_loop(self, interval_seconds: int):
        while not self._stop_event.wait(timeout=interval_seconds):
            logger.info("[ConsolidationEngine] Running scheduled consolidation...")
            try:
                reports = self.consolidate_all_sessions()
                active = [r for r in reports if not r.get("skipped") and not r.get("error")]
                if active:
                    logger.info(
                        f"[ConsolidationEngine] Consolidated {len(active)} sessions "
                        f"in this cycle."
                    )
            except Exception as e:
                logger.error(f"[ConsolidationEngine] Scheduler error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Clustering: DBSCAN on turn embeddings
    # ──────────────────────────────────────────────────────────────────────────

    def _cluster_turns(self, turns: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Cluster turns by semantic similarity using DBSCAN.

        DBSCAN is chosen over k-means because:
        - No need to specify k in advance
        - Naturally handles noise (turns that don't belong to any cluster)
        - Finds arbitrarily shaped clusters

        Returns dict: {cluster_id: [turn_dicts]}
        Noise points (cluster_id=-1) are each treated as their own cluster.
        """
        if len(turns) < 2:
            return {0: turns}

        if self.get_embedding_fn is None:
            # No embedder available: treat all turns as one cluster
            logger.warning(
                "[ConsolidationEngine] No embedding function. "
                "Clustering as single group."
            )
            return {0: turns}

        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import normalize

            contents = [t["content"] for t in turns]
            embeddings = np.array([self.get_embedding_fn(c) for c in contents])
            embeddings_normed = normalize(embeddings, norm="l2")

            # DBSCAN with cosine metric
            # eps=0.25: turns within 0.25 cosine distance are neighbors
            # min_samples=2: minimum cluster size (noise points handled separately)
            clustering = DBSCAN(
                eps=0.25,
                min_samples=2,
                metric="cosine",
                algorithm="brute",
            ).fit(embeddings_normed)

            labels = clustering.labels_  # -1 = noise

            # Group turns by cluster label
            clusters: Dict[int, List[Dict]] = {}
            noise_counter = max(labels) + 1 if len(labels) > 0 else 0

            for i, label in enumerate(labels):
                if label == -1:
                    # Each noise point becomes its own cluster
                    clusters[noise_counter] = [turns[i]]
                    noise_counter += 1
                else:
                    clusters.setdefault(label, []).append(turns[i])

            logger.debug(
                f"[ConsolidationEngine] DBSCAN: {len(set(labels))} raw labels → "
                f"{len(clusters)} clusters from {len(turns)} turns."
            )
            return clusters

        except Exception as e:
            logger.warning(f"[ConsolidationEngine] Clustering failed ({e}). Single group.")
            return {0: turns}

    # ──────────────────────────────────────────────────────────────────────────
    # Summary Generation per Cluster
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_cluster_summary(
        self,
        cluster_turns: List[Dict],
        cluster_id: int,
    ) -> Tuple[str, List[str]]:
        """
        Generate a semantic summary for a cluster of related turns.
        Falls back to a structured concatenation if LLM is unavailable.
        """
        try:
            summary_text, entities = self.summarizer.compress(cluster_turns)
            return summary_text, entities
        except Exception as e:
            logger.warning(
                f"[ConsolidationEngine] LLM summarization failed for "
                f"cluster {cluster_id}: {e}. Using fallback."
            )
            return self._fallback_summary(cluster_turns, cluster_id)

    def _fallback_summary(
        self,
        turns: List[Dict],
        cluster_id: int,
    ) -> Tuple[str, List[str]]:
        """
        Deterministic fallback when LLM is unavailable.
        Produces a structured concatenation with metadata.
        """
        roles_content = [
            f"{t['role'].upper()}: {t['content'][:200]}" for t in turns
        ]
        summary = (
            f"[Cluster {cluster_id} | {len(turns)} turns]\n"
            + "\n".join(roles_content)
        )
        # Extract capitalized words as pseudo-entities
        import re
        entities = list(set(
            re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', " ".join(t["content"] for t in turns))
        ))[:10]
        return summary, entities
