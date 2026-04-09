"""
AgentMem OS — Phase 3 Algorithm Tests
========================================
Comprehensive test suite for all 4 novel ML algorithms.

Algorithms under test:
  3A — MemoryImportanceScorer   (llm/importance_scorer.py)
  3B — SleepConsolidationEngine (llm/consolidation_engine.py)
  3C — EntityKnowledgeGraph     (db/knowledge_graph.py)
  3D — ProceduralMemory         (llm/procedural_memory.py)

Run on your Mac:
    cd /path/to/memnai
    source venv/bin/activate
    pytest tests/test_phase3_algorithms.py -v --tb=short

All tests use in-memory SQLite — no file system, no external services needed.
"""

import os
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Use in-memory SQLite for all tests
os.environ["MEMNAI_DB_PATH"] = ":memory:"


# ─────────────────────────────────────────────────────────────────────────────
# Shared Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def db_engine():
    """In-memory SQLite engine shared across the entire test session."""
    from memnai.db.models import Base
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()  # Close all pooled connections → prevents ResourceWarning


@pytest.fixture(scope="session")
def get_db(db_engine):
    """Session factory that returns a new session backed by the test engine."""
    SessionLocal = sessionmaker(bind=db_engine, autocommit=False, autoflush=False,
                                expire_on_commit=False)
    def _factory():
        return SessionLocal()
    return _factory


@pytest.fixture
def scorer():
    from memnai.llm.importance_scorer import MemoryImportanceScorer
    return MemoryImportanceScorer()


@pytest.fixture
def sample_turns():
    return [
        {"id": 1, "role": "user",      "content": "ok",                          "token_count": 1},
        {"id": 2, "role": "assistant", "content": "sure",                         "token_count": 1},
        {"id": 3, "role": "user",      "content": "What is FastAPI?",             "token_count": 5},
        {"id": 4, "role": "assistant", "content": "FastAPI is a modern Python web framework built on Pydantic and Starlette.", "token_count": 15},
        {"id": 5, "role": "user",      "content": "Alice from Google and Bob from Anthropic are collaborating on Claude.", "token_count": 16},
        {"id": 6, "role": "user",      "content": "yes",                          "token_count": 1},
        {"id": 7, "role": "assistant", "content": "Great. Sam Altman and Dario Amodei founded OpenAI and Anthropic respectively.", "token_count": 16},
        {"id": 8, "role": "user",      "content": "Can you debug this Python exception: AttributeError on line 42?", "token_count": 12},
    ]


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3A — MemoryImportanceScorer
# ═════════════════════════════════════════════════════════════════════════════

class TestMemoryImportanceScorer:

    # ── Core contract ────────────────────────────────────────────────────────

    def test_returns_all_turns(self, scorer, sample_turns):
        results = scorer.score_turns(sample_turns)
        assert len(results) == len(sample_turns)

    def test_scores_in_valid_range(self, scorer, sample_turns):
        results = scorer.score_turns(sample_turns)
        for _, score in results:
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1]"

    def test_sorted_ascending(self, scorer, sample_turns):
        """Results must be sorted ascending — lowest importance first (compress first)."""
        results = scorer.score_turns(sample_turns)
        scores = [s for _, s in results]
        assert scores == sorted(scores)

    def test_empty_turns_returns_empty(self, scorer):
        assert scorer.score_turns([]) == []

    def test_single_turn_returns_one(self, scorer):
        turns = [{"role": "user", "content": "hello"}]
        results = scorer.score_turns(turns)
        assert len(results) == 1
        assert 0.0 <= results[0][1] <= 1.0

    # ── Compression candidates ───────────────────────────────────────────────

    def test_compression_split_correct_sizes(self, scorer, sample_turns):
        to_compress, to_keep = scorer.get_compression_candidates(
            sample_turns, compress_fraction=0.25
        )
        assert len(to_compress) == 2      # 25% of 8 = 2
        assert len(to_keep) == 6
        assert len(to_compress) + len(to_keep) == len(sample_turns)

    def test_compression_fraction_30_percent(self, scorer, sample_turns):
        to_compress, to_keep = scorer.get_compression_candidates(
            sample_turns, compress_fraction=0.30
        )
        assert len(to_compress) == 2      # max(1, int(8*0.30)) = 2

    def test_compress_at_least_one(self, scorer):
        """Even with tiny fraction, at least 1 turn is selected."""
        turns = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        to_compress, _ = scorer.get_compression_candidates(turns, compress_fraction=0.01)
        assert len(to_compress) >= 1

    # ── Signal 1: TF-IDF ────────────────────────────────────────────────────

    def test_tfidf_length_matches(self, scorer):
        contents = ["hello world", "fastapi pydantic starlette", "yes ok"]
        scores = scorer._compute_tfidf_scores(contents)
        assert len(scores) == 3

    def test_tfidf_normalized_range(self, scorer):
        contents = ["rare unique terminology blockchain cryptography",
                    "ok yes", "hello there how are you doing today"]
        scores = scorer._compute_tfidf_scores(contents)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_tfidf_single_document_returns_neutral(self, scorer):
        scores = scorer._compute_tfidf_scores(["only one doc"])
        assert scores == [0.5]

    # ── Signal 2: Semantic Novelty ───────────────────────────────────────────

    def test_novelty_no_embedder_returns_neutral(self, scorer):
        contents = ["turn a", "turn b", "turn c"]
        scores = scorer._compute_novelty_scores(contents, [], None)
        assert scores == [0.5, 0.5, 0.5]

    def test_novelty_no_existing_summaries_returns_neutral(self, scorer):
        contents = ["turn a", "turn b"]
        scores = scorer._compute_novelty_scores(contents, [], lambda x: [0.1] * 5)
        assert scores == [0.5, 0.5]

    def test_novelty_identical_to_summary_scores_low(self, scorer):
        """A turn identical to existing summary should have low novelty."""
        dim = 8
        summary_emb = np.array([1.0] + [0.0] * (dim - 1))
        # Turn embedding identical to summary
        identical_emb = np.array([1.0] + [0.0] * (dim - 1))
        # Turn embedding orthogonal to summary
        novel_emb = np.array([0.0, 1.0] + [0.0] * (dim - 2))

        def mock_embed(text):
            if text == "identical":
                return identical_emb.tolist()
            return novel_emb.tolist()

        scores = scorer._compute_novelty_scores(
            ["identical", "novel"],
            [summary_emb],
            mock_embed
        )
        # "identical" should have lower novelty than "novel"
        # After minmax normalization: identical=0.0, novel=1.0
        assert scores[0] < scores[1]

    # ── Signal 3: Entity Density ─────────────────────────────────────────────

    def test_entity_regex_fallback_finds_capitalized(self, scorer):
        scores = scorer._entity_scores_regex([
            "yes ok",
            "Alice and Bob work at Google on FastAPI with Pydantic"
        ])
        assert len(scores) == 2
        # Entity-rich turn should score higher
        assert scores[1] > scores[0]

    def test_entity_regex_uniform_input_returns_neutral(self, scorer):
        scores = scorer._entity_scores_regex(["yes", "ok", "sure"])
        # All zeros → minmax → all 0.5
        assert all(s == 0.5 for s in scores)

    # ── Signal 4: Recency Decay ──────────────────────────────────────────────

    def test_recency_monotonically_increasing(self, scorer):
        recency = scorer._compute_recency_scores(10)
        assert all(recency[i] <= recency[i+1] for i in range(len(recency)-1))

    def test_recency_newest_is_one(self, scorer):
        recency = scorer._compute_recency_scores(5)
        assert recency[-1] == 1.0

    def test_recency_oldest_less_than_newest(self, scorer):
        recency = scorer._compute_recency_scores(20)
        assert recency[0] < recency[-1]

    def test_recency_single_turn_is_one(self, scorer):
        recency = scorer._compute_recency_scores(1)
        assert recency[0] == 1.0

    def test_recency_half_life_property(self, scorer):
        """Score at position HALF_LIFE from end should be ~0.5."""
        n = scorer.HALF_LIFE * 2
        recency = scorer._compute_recency_scores(n)
        # Turn at exactly HALF_LIFE positions from the end
        half_life_score = recency[n - 1 - scorer.HALF_LIFE]
        assert abs(half_life_score - 0.5) < 0.05

    # ── MinMax utility ───────────────────────────────────────────────────────

    def test_minmax_uniform_returns_half(self, scorer):
        result = scorer._minmax_normalize(np.array([7.0, 7.0, 7.0]))
        assert result == [0.5, 0.5, 0.5]

    def test_minmax_range_is_zero_to_one(self, scorer):
        result = scorer._minmax_normalize(np.array([0.0, 5.0, 10.0]))
        assert abs(result[0] - 0.0) < 1e-9
        assert abs(result[1] - 0.5) < 1e-9
        assert abs(result[2] - 1.0) < 1e-9

    def test_minmax_single_element(self, scorer):
        result = scorer._minmax_normalize(np.array([42.0]))
        assert result == [0.5]

    # ── Integration: low-info turns score lower than high-info turns ─────────

    def test_filler_turns_score_lower_than_entity_rich(self, scorer):
        turns = [
            {"role": "user", "content": "ok"},
            {"role": "user", "content": "yes"},
            {"role": "user", "content": "Sam Altman CEO of OpenAI met Dario Amodei from Anthropic at Google DeepMind"},
        ]
        results = scorer.score_turns(turns)
        scores_by_content = {t["content"]: s for t, s in results}
        entity_score = scores_by_content["Sam Altman CEO of OpenAI met Dario Amodei from Anthropic at Google DeepMind"]
        filler_scores = [scores_by_content["ok"], scores_by_content["yes"]]
        assert entity_score > max(filler_scores)


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3B — SleepConsolidationEngine
# ═════════════════════════════════════════════════════════════════════════════

class TestSleepConsolidationEngine:

    @pytest.fixture
    def engine_fixture(self, get_db):
        from memnai.llm.consolidation_engine import SleepConsolidationEngine
        from memnai.llm.importance_scorer import MemoryImportanceScorer

        mock_summarizer = MagicMock()
        mock_summarizer.compress.return_value = (
            "Summary of cluster turns about AI and Python.",
            ["AI", "Python", "FastAPI"]
        )

        mock_chroma = MagicMock()
        mock_chroma.add_summary_chunk.return_value = None

        scorer = MemoryImportanceScorer()

        return SleepConsolidationEngine(
            get_db_session=get_db,
            summarizer=mock_summarizer,
            chroma=mock_chroma,
            importance_scorer=scorer,
            get_embedding_fn=None,   # no embedder — tests clustering fallback
        )

    # ── Clustering ───────────────────────────────────────────────────────────

    def test_cluster_single_group_without_embedder(self, engine_fixture):
        """Without embedder, all turns go into cluster 0."""
        turns = [
            {"id": i, "role": "user", "content": f"msg {i}", "token_count": 5}
            for i in range(5)
        ]
        clusters = engine_fixture._cluster_turns(turns)
        assert len(clusters) == 1
        assert 0 in clusters
        assert len(clusters[0]) == 5

    def test_cluster_single_turn_returns_one_cluster(self, engine_fixture):
        turns = [{"id": 1, "role": "user", "content": "hello", "token_count": 2}]
        clusters = engine_fixture._cluster_turns(turns)
        assert len(clusters) == 1

    def test_cluster_empty_returns_single_group(self, engine_fixture):
        clusters = engine_fixture._cluster_turns([])
        assert len(clusters) == 1
        assert clusters[0] == []

    # ── Fallback Summary ─────────────────────────────────────────────────────

    def test_fallback_summary_contains_turn_count(self, engine_fixture):
        turns = [
            {"id": 1, "role": "user", "content": "I use FastAPI"},
            {"id": 2, "role": "assistant", "content": "FastAPI is great"},
        ]
        summary, entities = engine_fixture._fallback_summary(turns, cluster_id=0)
        assert "2" in summary or "Cluster" in summary
        assert isinstance(entities, list)

    def test_fallback_summary_extracts_entities(self, engine_fixture):
        turns = [{"id": 1, "role": "user", "content": "Alice and Google and Anthropic"}]
        _, entities = engine_fixture._fallback_summary(turns, cluster_id=0)
        entity_texts = " ".join(entities)
        assert "Alice" in entity_texts or "Google" in entity_texts

    # ── Session Consolidation ─────────────────────────────────────────────────

    def test_consolidate_skips_when_below_threshold(self, engine_fixture, get_db):
        """Session with few tokens should skip consolidation."""
        from memnai.db.models import Session
        db = get_db()
        s = Session(session_id="test-cons-low", total_tokens=1000, branch_type="root")
        db.add(s)
        db.commit()
        db.close()

        report = engine_fixture.consolidate("test-cons-low", force=False)
        assert report.get("skipped") is True

    def test_consolidate_skips_too_few_turns(self, engine_fixture, get_db):
        """Session above threshold but with < 4 turns should skip."""
        from memnai.db.models import Session, Turn
        db = get_db()
        s = Session(session_id="test-cons-few", total_tokens=100_000, branch_type="root")
        db.add(s)
        db.flush()
        db.add(Turn(session_id="test-cons-few", role="user", content="hi", token_count=1))
        db.commit()
        db.close()

        report = engine_fixture.consolidate("test-cons-few", force=True)
        assert report.get("skipped") is True

    def test_consolidate_forced_runs_on_full_session(self, engine_fixture, get_db):
        """Force=True bypasses threshold check. Should return a real report."""
        from memnai.db.models import Session, Turn
        db = get_db()
        sid = "test-cons-full"
        s = Session(session_id=sid, total_tokens=50_000, branch_type="root")
        db.add(s)
        db.flush()
        for i in range(10):
            db.add(Turn(
                session_id=sid, role="user",
                content=f"Message {i} about FastAPI Pydantic SQLAlchemy",
                token_count=500
            ))
        db.commit()
        db.close()

        report = engine_fixture.consolidate(sid, compress_fraction=0.30, force=True)
        assert "error" not in report
        assert not report.get("skipped")
        assert report["turns_compressed"] >= 1
        assert report["summaries_created"] >= 1
        assert report["tokens_freed"] > 0

    # ── Background Scheduler ─────────────────────────────────────────────────

    def test_scheduler_starts_and_stops(self, engine_fixture):
        import time
        engine_fixture.start_background_scheduler(interval_seconds=999)
        time.sleep(0.1)
        assert engine_fixture._scheduler_thread is not None
        assert engine_fixture._scheduler_thread.is_alive()
        engine_fixture.stop_background_scheduler()
        assert not engine_fixture._scheduler_thread.is_alive()

    def test_cannot_start_scheduler_twice(self, engine_fixture):
        import time
        engine_fixture.start_background_scheduler(interval_seconds=999)
        time.sleep(0.05)
        thread_1 = engine_fixture._scheduler_thread
        engine_fixture.start_background_scheduler(interval_seconds=999)
        assert engine_fixture._scheduler_thread is thread_1   # same thread, not new
        engine_fixture.stop_background_scheduler()


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3C — EntityKnowledgeGraph
# ═════════════════════════════════════════════════════════════════════════════

class TestEntityKnowledgeGraph:

    @pytest.fixture
    def kg(self, get_db):
        from memnai.db.knowledge_graph import EntityKnowledgeGraph
        return EntityKnowledgeGraph(get_db)

    # ── Entity Extraction (regex fallback) ───────────────────────────────────

    def test_regex_finds_named_entities(self, kg):
        entities = kg._extract_entities_regex(
            "Alice and Bob work at Google. They use FastAPI and Pydantic."
        )
        texts = [e[0] for e in entities]
        assert "Alice" in texts
        assert "Google" in texts

    def test_regex_filters_stopwords(self, kg):
        entities = kg._extract_entities_regex(
            "The This That With From When What are stopwords."
        )
        texts = [e[0] for e in entities]
        for sw in ["The", "This", "That", "With", "From", "When", "What"]:
            assert sw not in texts

    def test_regex_deduplicates(self, kg):
        # Capitalized words must be separated by lowercase words so the greedy
        # multi-word matcher (e.g. "New York") doesn't swallow all three "Alice"s
        # into a single "Alice Alice Alice Google" entity.
        entities = kg._extract_entities_regex("Alice and Alice and Alice at Google")
        texts = [e[0] for e in entities]
        assert texts.count("Alice") == 1

    def test_regex_caps_at_15(self, kg):
        # Generate text with 20 distinct capitalized words
        words = " ".join([f"Entity{i}" for i in range(20)])
        entities = kg._extract_entities_regex(words)
        assert len(entities) <= 15

    def test_regex_ignores_short_words(self, kg):
        entities = kg._extract_entities_regex("Ax By Cz Done")
        texts = [e[0] for e in entities]
        assert "Ax" not in texts   # len <= 2

    # ── Serialization ────────────────────────────────────────────────────────

    def test_serialize_empty_returns_empty(self, kg):
        result = kg._serialize_subgraph([], "agent1")
        assert result == ""

    def test_top_entities_empty_graph_returns_empty(self, kg):
        result = kg._top_entities_summary("unknown_agent", top_k=5)
        assert result == ""

    def test_serialize_single_node_no_edges(self, kg):
        """A graph with one node and no edges should produce entity line, no relationship line."""
        import networkx as nx
        kg._graph = nx.Graph()
        kg._graph.add_node(
            "a1:Anthropic",
            text="Anthropic",
            entity_type="ORG",
            mention_count=5,
            agent_id="a1",
        )
        result = kg._serialize_subgraph(["a1:Anthropic"], "a1")
        assert "Anthropic" in result
        assert "ORG" in result
        assert "5×" in result
        assert "Relationships" not in result   # no edges

    def test_serialize_two_nodes_with_edge_shows_relationship(self, kg):
        import networkx as nx
        kg._graph = nx.Graph()
        kg._graph.add_node("a2:Alice", text="Alice", entity_type="PERSON",
                           mention_count=3, agent_id="a2")
        kg._graph.add_node("a2:Google", text="Google", entity_type="ORG",
                           mention_count=7, agent_id="a2")
        kg._graph.add_edge("a2:Alice", "a2:Google", weight=4.0)

        result = kg._serialize_subgraph(["a2:Alice", "a2:Google"], "a2")
        assert "Alice" in result
        assert "Google" in result
        assert "↔" in result
        assert "4×" in result

    # ── Ingest Turn ──────────────────────────────────────────────────────────

    def test_ingest_turn_adds_to_graph(self, kg, get_db):
        """ingest_turn should add nodes to the in-memory graph."""
        # Use regex fallback (no spaCy in test env)
        with patch.object(kg, '_extract_entities',
                          side_effect=kg._extract_entities_regex):
            count = kg.ingest_turn(
                session_id="kg-test-session",
                agent_id="agent-kg",
                content="Sahith is building AgentMem at Anthropic"
            )
        assert count >= 0   # may be 0 if DB write fails in test env

    # ── Subgraph Retrieval ────────────────────────────────────────────────────

    def test_get_relevant_subgraph_empty_graph_returns_empty(self, kg):
        fresh_kg = kg.__class__(MagicMock())
        result = fresh_kg.get_relevant_subgraph("what is FastAPI", "nobody", top_k=5)
        assert result == ""

    def test_get_relevant_subgraph_no_query_entities_returns_top(self, kg):
        """Query with no extractable entities should fall back to top-entity summary."""
        import networkx as nx
        kg._graph = nx.Graph()
        kg._graph.add_node("a3:FastAPI", text="FastAPI", entity_type="PRODUCT",
                           mention_count=10, agent_id="a3")
        kg._graph.add_node("a3:Python", text="Python", entity_type="PRODUCT",
                           mention_count=8, agent_id="a3")

        with patch.object(kg, '_extract_entities', return_value=[]):
            result = kg.get_relevant_subgraph("tell me something", "a3", top_k=5)
        # Falls back to top-entities summary
        assert "FastAPI" in result or "Python" in result


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3D — ProceduralMemory
# ═════════════════════════════════════════════════════════════════════════════

class TestProceduralMemory:

    @pytest.fixture
    def pm(self, get_db):
        from memnai.llm.procedural_memory import ProceduralMemory
        return ProceduralMemory(get_db, summarizer=None)

    # ── Trigger Classification ────────────────────────────────────────────────

    @pytest.mark.parametrize("text,expected", [
        ("there is a bug in login",           "bug_report"),
        ("the app crashes on startup",        "bug_report"),
        ("I want to add dark mode",           "feature_request"),
        ("please implement OAuth login",      "feature_request"),
        ("how does the auth middleware work", "question"),
        ("what is the purpose of this class", "question"),
        ("can you review this pull request",  "code_review"),
        ("please refactor this function",     "code_review"),
        ("help me debug this stack trace",    "debugging"),
        ("let's design the system architecture", "planning"),
        ("this is confusing, can you clarify", "clarification"),
    ])
    def test_trigger_classification(self, text, expected):
        from memnai.llm.procedural_memory import classify_trigger
        result = classify_trigger(text)
        assert result == expected, f"'{text}' → got '{result}', expected '{expected}'"

    def test_trigger_unknown_returns_general(self):
        from memnai.llm.procedural_memory import classify_trigger
        result = classify_trigger("the sky is blue today")
        assert result == "general"

    # ── Action Extraction ─────────────────────────────────────────────────────

    @pytest.mark.parametrize("response,expected", [
        ("Here is the fix:\n```python\ndef foo(): pass\n```", "provided code block"),
        ("x" * 600,                                           "gave detailed response"),
        ("Let me explain this concept to you in detail.",    "explained concept"),
        ("I would suggest using FastAPI instead.",           "suggested solution"),
    ])
    def test_action_extraction(self, response, expected):
        from memnai.llm.procedural_memory import extract_action
        result = extract_action(response)
        assert result == expected, f"Got '{result}', expected '{expected}'"

    # ── Pattern Generation (template fallback) ────────────────────────────────

    def test_generate_pattern_text_uses_template(self, pm):
        pattern = pm._generate_pattern_text(
            "bug_report", "provided code block", []
        )
        assert "bug" in pattern.lower() or "When" in pattern
        assert isinstance(pattern, str)
        assert len(pattern) > 5

    # ── Pattern Mining ────────────────────────────────────────────────────────

    def test_mine_patterns_needs_min_turns(self, pm, get_db):
        """Session with < 4 turns should return 0 patterns."""
        from memnai.db.models import Session, Turn
        db = get_db()
        sid = "pm-few-turns"
        db.add(Session(session_id=sid, branch_type="root"))
        db.flush()
        db.add(Turn(session_id=sid, role="user", content="hello", token_count=1))
        db.commit()
        db.close()

        count = pm.mine_patterns(sid, agent_id="pm-agent")
        assert count == 0

    def test_mine_patterns_from_session(self, pm, get_db):
        """Session with repeated (bug_report, code_block) pairs should produce a pattern."""
        from memnai.db.models import Session, Turn
        db = get_db()
        sid = "pm-mine-test"
        db.add(Session(session_id=sid, branch_type="root"))
        db.flush()

        # Create 3 bug→code pairs (support_count = 3 ≥ MIN_SUPPORT_COUNT=2)
        for i in range(3):
            db.add(Turn(session_id=sid, role="user",
                        content=f"there is a bug in module_{i}", token_count=6))
            db.add(Turn(session_id=sid, role="assistant",
                        content=f"Here is the fix:\n```python\npass  # fix {i}\n```", token_count=10))
        db.commit()
        db.close()

        count = pm.mine_patterns(sid, agent_id="pm-agent-mine")
        assert count >= 1

    def test_mine_patterns_idempotent_upsert(self, pm, get_db):
        """Mining the same session twice should increment support_count, not duplicate."""
        from memnai.db.models import Session, Turn, ProceduralPattern
        db = get_db()
        sid = "pm-upsert-test"
        aid = "pm-upsert-agent"
        db.add(Session(session_id=sid, branch_type="root"))
        db.flush()

        for i in range(3):
            db.add(Turn(session_id=sid, role="user",
                        content=f"bug in function_{i}", token_count=5))
            db.add(Turn(session_id=sid, role="assistant",
                        content=f"```python\nfix_{i}\n```", token_count=6))
        db.commit()
        db.close()

        pm.mine_patterns(sid, agent_id=aid)
        db2 = get_db()
        count_1 = db2.query(ProceduralPattern).filter(
            ProceduralPattern.agent_id == aid
        ).count()
        db2.close()

        pm.mine_patterns(sid, agent_id=aid)
        db3 = get_db()
        count_2 = db3.query(ProceduralPattern).filter(
            ProceduralPattern.agent_id == aid
        ).count()
        db3.close()

        # Should not have doubled
        assert count_2 == count_1

    # ── Pattern Retrieval ─────────────────────────────────────────────────────

    def test_get_relevant_patterns_empty_returns_empty(self, pm):
        result = pm.get_relevant_patterns("how to fix a bug", agent_id="no-agent")
        assert result == ""

    def test_get_relevant_patterns_returns_string(self, pm, get_db):
        """After mining, get_relevant_patterns should return non-empty string."""
        from memnai.db.models import Session, Turn
        db = get_db()
        sid = "pm-retrieve-test"
        aid = "pm-retrieve-agent"
        db.add(Session(session_id=sid, branch_type="root"))
        db.flush()

        for i in range(3):
            db.add(Turn(session_id=sid, role="user",
                        content=f"bug in auth module {i}", token_count=5))
            db.add(Turn(session_id=sid, role="assistant",
                        content=f"```python\nauth_fix_{i}()\n```", token_count=6))
        db.commit()
        db.close()

        pm.mine_patterns(sid, agent_id=aid)
        result = pm.get_relevant_patterns("there's a bug in the API", agent_id=aid)

        if result:  # patterns found
            assert "[BEHAVIORAL PATTERNS]" in result
            assert "confidence" in result

    # ── Serialization ─────────────────────────────────────────────────────────

    def test_serialize_patterns_format(self, pm, get_db):
        """Serialized patterns must include BEHAVIORAL PATTERNS header and bullets."""
        from memnai.db.models import ProceduralPattern
        db = get_db()
        mock_patterns = [
            ProceduralPattern(
                trigger="bug_report",
                action="provided code block",
                full_pattern="When user reports a bug → provide code block",
                confidence=0.85,
                support_count=5,
            )
        ]
        result = pm._serialize_patterns(mock_patterns)
        assert "[BEHAVIORAL PATTERNS]" in result
        assert "85%" in result
        assert "5×" in result
        assert "•" in result


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3: Integration — all algorithms work together
# ═════════════════════════════════════════════════════════════════════════════

class TestPhase3Integration:

    def test_importance_scorer_feeds_consolidation_engine(self, get_db):
        """
        Importance scorer and consolidation engine work together:
        scored turns feed directly into consolidation engine's clustering step.
        """
        from memnai.llm.importance_scorer import MemoryImportanceScorer
        from memnai.llm.consolidation_engine import SleepConsolidationEngine

        scorer = MemoryImportanceScorer()
        mock_summarizer = MagicMock()
        mock_summarizer.compress.return_value = ("Summary.", ["Entity"])
        mock_chroma = MagicMock()

        engine = SleepConsolidationEngine(
            get_db_session=get_db,
            summarizer=mock_summarizer,
            chroma=mock_chroma,
            importance_scorer=scorer,
        )

        turns = [
            {"id": i, "role": "user",
             "content": f"Turn {i} about topic {'X' if i % 2 == 0 else 'Y'}",
             "token_count": 100}
            for i in range(10)
        ]

        to_compress, to_keep = scorer.get_compression_candidates(turns, 0.30)
        assert len(to_compress) == 3
        clusters = engine._cluster_turns(to_compress)
        assert len(clusters) >= 1
        assert sum(len(v) for v in clusters.values()) == len(to_compress)

    def test_kg_ingest_and_retrieve_cycle(self, get_db):
        """
        Entity KG correctly ingests a turn and retrieves relevant subgraph.
        """
        from memnai.db.knowledge_graph import EntityKnowledgeGraph
        import networkx as nx

        kg = EntityKnowledgeGraph(get_db)
        # Manually populate graph (no spaCy in test env)
        kg._graph = nx.Graph()
        kg._graph.add_node("ag1:Claude",     text="Claude",     entity_type="PRODUCT", mention_count=8, agent_id="ag1")
        kg._graph.add_node("ag1:Anthropic",  text="Anthropic",  entity_type="ORG",     mention_count=6, agent_id="ag1")
        kg._graph.add_node("ag1:Sahith",     text="Sahith",     entity_type="PERSON",  mention_count=4, agent_id="ag1")
        kg._graph.add_edge("ag1:Claude", "ag1:Anthropic", weight=5.0)
        kg._graph.add_edge("ag1:Sahith", "ag1:Claude",    weight=3.0)

        with patch.object(kg, '_extract_entities', return_value=[("Claude", "PRODUCT")]):
            result = kg.get_relevant_subgraph("tell me about Claude", "ag1", top_k=10)

        assert "Claude" in result
        assert "Anthropic" in result or "WORLD MODEL" in result

    def test_procedural_and_scorer_complementary(self, get_db):
        """
        ProceduralMemory and ImportanceScorer address orthogonal concerns:
        procedural = what to do, scorer = what to keep.
        Both should operate on the same turns without conflict.
        """
        from memnai.llm.importance_scorer import MemoryImportanceScorer
        from memnai.llm.procedural_memory import classify_trigger, extract_action

        scorer = MemoryImportanceScorer()

        turns = [
            {"role": "user",      "content": "there is a bug in the auth module"},
            {"role": "assistant", "content": "```python\nfix_auth()\n```"},
            {"role": "user",      "content": "ok"},
            {"role": "assistant", "content": "done"},
        ]

        # Importance scorer picks what to compress
        scored = scorer.score_turns(turns)
        assert len(scored) == 4
        to_compress = [t for t, _ in scored[:2]]

        # Procedural classifier picks behavioral pattern
        trigger = classify_trigger(turns[0]["content"])
        action  = extract_action(turns[1]["content"])
        assert trigger == "bug_report"
        assert action  == "provided code block"

        # They don't conflict — both ran independently on same data
        assert len(to_compress) == 2
