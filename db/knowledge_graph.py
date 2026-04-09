"""
AgentMem OS — Entity Knowledge Graph (Global Map)
===================================================
Novel Algorithm #3: Persistent cross-session entity co-occurrence graph.

This fills the previously empty 10% "Global Map" slot in the ContextAssembler.

Architecture:
  - spaCy NER extracts named entities from every turn at save time
  - NetworkX maintains an in-memory weighted graph
  - SQLite persists nodes and edges across restarts
  - On context assembly, the subgraph most relevant to the current query
    is serialized and injected as a structured world model

Why this matters:
  Without a knowledge graph, an agent re-reads 1000 turns to learn that
  "Alice is the lead engineer and Alice works with Bob on the payments module."
  With the graph, this is O(1) lookup on the Alice node.

Graph structure:
  Nodes: named entities (text, type, mention_count)
  Edges: co-occurrence in same turn (weighted by co-occurrence frequency)

Example serialized output:
  [WORLD MODEL]
  Entities: Alice (PERSON, 12 mentions), FastAPI (PRODUCT, 8 mentions)
  Relationships: Alice ↔ FastAPI (co-occurred 5x), Alice ↔ Bob (3x)
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime

import networkx as nx
from loguru import logger


class EntityKnowledgeGraph:
    """
    Persistent entity co-occurrence knowledge graph.

    Usage:
        kg = EntityKnowledgeGraph(get_db_session)
        # After saving a turn:
        kg.ingest_turn(session_id, agent_id, turn_content)
        # At context assembly time:
        world_model_str = kg.get_relevant_subgraph(query, agent_id, top_k=10)
    """

    def __init__(self, get_db_session):
        self.get_db = get_db_session
        self._nlp = None
        self._graph: nx.Graph = nx.Graph()  # In-memory graph; synced with DB
        self._loaded_agents: Set[str] = set()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def ingest_turn(
        self,
        session_id: str,
        agent_id: Optional[str],
        content: str,
    ) -> int:
        """
        Extract entities from a turn and update the graph.
        Returns the number of new entities found.

        Called automatically by ConversationStore.save_turn().
        """
        entities = self._extract_entities(content)
        if not entities:
            return 0

        db = self.get_db()
        try:
            from memnai.db.models import KnowledgeGraphNode, KnowledgeGraphEdge

            # Upsert nodes
            node_ids = {}
            for text, etype in entities:
                node = (
                    db.query(KnowledgeGraphNode)
                    .filter(
                        KnowledgeGraphNode.entity_text == text,
                        KnowledgeGraphNode.agent_id == agent_id,
                    )
                    .first()
                )
                if node:
                    node.mention_count += 1
                    node.last_seen = datetime.utcnow()
                else:
                    node = KnowledgeGraphNode(
                        agent_id=agent_id,
                        session_id=session_id,
                        entity_text=text,
                        entity_type=etype,
                    )
                    db.add(node)
                    db.flush()  # get node.id

                node_ids[text] = node.id

                # Update in-memory graph
                node_key = f"{agent_id}:{text}"
                if not self._graph.has_node(node_key):
                    self._graph.add_node(
                        node_key,
                        text=text,
                        entity_type=etype,
                        mention_count=1,
                        agent_id=agent_id,
                    )
                else:
                    self._graph.nodes[node_key]["mention_count"] = (
                        self._graph.nodes[node_key].get("mention_count", 0) + 1
                    )

            # Upsert edges (co-occurrence within this turn)
            entity_list = list(node_ids.items())
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    text_a, id_a = entity_list[i]
                    text_b, id_b = entity_list[j]

                    # Ensure consistent ordering for DB lookup
                    src, tgt = (id_a, id_b) if id_a < id_b else (id_b, id_a)
                    src_txt = text_a if id_a < id_b else text_b
                    tgt_txt = text_b if id_a < id_b else text_a

                    edge = (
                        db.query(KnowledgeGraphEdge)
                        .filter(
                            KnowledgeGraphEdge.source_id == src,
                            KnowledgeGraphEdge.target_id == tgt,
                        )
                        .first()
                    )
                    if edge:
                        edge.weight += 1.0
                        edge.last_updated = datetime.utcnow()
                    else:
                        edge = KnowledgeGraphEdge(
                            source_id=src,
                            target_id=tgt,
                            weight=1.0,
                            session_id=session_id,
                        )
                        db.add(edge)

                    # Update in-memory graph
                    key_a = f"{agent_id}:{src_txt}"
                    key_b = f"{agent_id}:{tgt_txt}"
                    if self._graph.has_edge(key_a, key_b):
                        self._graph[key_a][key_b]["weight"] += 1.0
                    else:
                        self._graph.add_edge(key_a, key_b, weight=1.0)

            db.commit()
            return len(entities)

        except Exception as e:
            logger.warning(f"[KnowledgeGraph] ingest_turn failed: {e}")
            db.rollback()
            return 0
        finally:
            db.close()

    def get_relevant_subgraph(
        self,
        query: str,
        agent_id: Optional[str],
        top_k: int = 10,
        max_hops: int = 2,
    ) -> str:
        """
        Find entities in the query → expand subgraph up to max_hops →
        serialize to text for context assembly.

        This is what fills the 10% Global Map slot.

        Returns empty string if graph is empty.
        """
        if self._graph.number_of_nodes() == 0:
            self._load_graph_from_db(agent_id)

        if self._graph.number_of_nodes() == 0:
            return ""

        # Find query entities
        query_entities = self._extract_entities(query)
        if not query_entities:
            # Fallback: return top-k most mentioned entities
            return self._top_entities_summary(agent_id, top_k)

        # Build seed nodes from query
        seed_nodes = set()
        for text, _ in query_entities:
            node_key = f"{agent_id}:{text}"
            if self._graph.has_node(node_key):
                seed_nodes.add(node_key)

        if not seed_nodes:
            return self._top_entities_summary(agent_id, top_k)

        # BFS expansion up to max_hops
        subgraph_nodes = set(seed_nodes)
        frontier = set(seed_nodes)

        for _ in range(max_hops):
            next_frontier = set()
            for node in frontier:
                if self._graph.has_node(node):
                    neighbors = set(self._graph.neighbors(node))
                    # Only expand to strongly connected neighbors (weight >= 2)
                    strong_neighbors = {
                        n for n in neighbors
                        if self._graph[node][n].get("weight", 0) >= 2
                        and n not in subgraph_nodes
                    }
                    next_frontier.update(strong_neighbors)
            subgraph_nodes.update(next_frontier)
            frontier = next_frontier

            if len(subgraph_nodes) >= top_k * 2:
                break

        # Trim to top_k by mention_count
        sorted_nodes = sorted(
            subgraph_nodes,
            key=lambda n: self._graph.nodes[n].get("mention_count", 0),
            reverse=True,
        )[:top_k]

        return self._serialize_subgraph(sorted_nodes, agent_id)

    def get_entity_count(self, agent_id: Optional[str] = None) -> int:
        """Return number of entities in the graph for this agent."""
        return sum(
            1 for n in self._graph.nodes
            if str(n).startswith(f"{agent_id}:")
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Entity Extraction (spaCy NER)
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract (entity_text, entity_type) pairs using spaCy NER.
        Filters to meaningful entity types only.
        Falls back to regex heuristic if spaCy unavailable.
        """
        # Target entity types
        TARGET_TYPES = {
            "PERSON", "ORG", "PRODUCT", "GPE",        # High signal
            "WORK_OF_ART", "EVENT", "LANGUAGE",        # Medium signal
            "FAC", "LOC",                              # Location
        }

        try:
            nlp = self._get_nlp()
            doc = nlp(text[:5000])  # cap at 5k chars for speed
            entities = []
            seen = set()
            for ent in doc.ents:
                if ent.label_ in TARGET_TYPES:
                    clean = ent.text.strip()
                    if len(clean) >= 2 and clean.lower() not in seen:
                        seen.add(clean.lower())
                        entities.append((clean, ent.label_))
            return entities

        except Exception:
            return self._extract_entities_regex(text)

    def _extract_entities_regex(self, text: str) -> List[Tuple[str, str]]:
        """
        Fallback entity extractor: finds capitalized word sequences as proxies
        for named entities when spaCy is unavailable.

        Handles two cases:
          • Normal multi-word entities ("New York", "Sam Altman") — kept intact
            as a single entity by the greedy match, deduplicated by first word.
          • Pathological input ("Alice Alice Alice Google") — the greedy matcher
            swallows everything into one string.  Detected by presence of repeated
            words inside the match; handled by emitting each unique unseen word
            as its own entity so deduplication still works correctly.
        """
        pattern = re.compile(r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b')
        matches = pattern.findall(text)

        stopwords = {"The", "This", "That", "With", "From", "When", "What"}
        seen_words: set = set()
        entities: List[Tuple[str, str]] = []

        for m in matches:
            if len(m) <= 2:
                continue

            words = m.split()
            unique_words = list(dict.fromkeys(words))   # deduplicate, preserve order
            is_pathological = len(unique_words) < len(words)

            # Strip stopwords from the beginning of the match; if nothing useful
            # remains (e.g. "The This That") skip the match entirely.
            meaningful = [w for w in unique_words if w not in stopwords and len(w) > 2]
            if not meaningful:
                continue

            if is_pathological:
                # Greedy match ate repeated words (e.g. "Alice Alice Alice Google").
                # Emit each unique, unseen, meaningful word as its own entity.
                for word in meaningful:
                    if word not in seen_words:
                        seen_words.add(word)
                        entities.append((word, "UNKNOWN"))
            else:
                # Normal multi-word entity ("New York") or single word ("Alice").
                # Filter if the lead word is a stopword; use first meaningful word
                # as the canonical representative if needed.
                canonical = next((w for w in words if w not in stopwords and len(w) > 2), None)
                if canonical and canonical not in seen_words:
                    seen_words.update(words)
                    # Rebuild the entity from the first meaningful word onward
                    # so "The New York" becomes "New York", not "The New York".
                    start = words.index(canonical)
                    entity_text = " ".join(words[start:])
                    entities.append((entity_text, "UNKNOWN"))

        return entities[:15]

    # ──────────────────────────────────────────────────────────────────────────
    # Graph Serialization
    # ──────────────────────────────────────────────────────────────────────────

    def _serialize_subgraph(self, node_keys: List[str], agent_id: Optional[str]) -> str:
        """
        Convert a subgraph to a compact text representation for context injection.

        Output format:
          [WORLD MODEL]
          Entities: Alice (PERSON, 12×), FastAPI (PRODUCT, 5×)
          Relationships: Alice ↔ Bob (8×), Alice ↔ FastAPI (3×)
        """
        if not node_keys:
            return ""

        lines = ["[WORLD MODEL]"]

        # Entities
        entity_parts = []
        for nk in node_keys:
            data = self._graph.nodes[nk]
            entity_parts.append(
                f"{data.get('text', nk)} ({data.get('entity_type','?')}, "
                f"{data.get('mention_count', 0)}×)"
            )
        lines.append("Entities: " + ", ".join(entity_parts))

        # Relationships (edges within subgraph, sorted by weight)
        edges_seen = set()
        edge_parts = []
        for nk in node_keys:
            for neighbor in self._graph.neighbors(nk):
                if neighbor in node_keys:
                    edge_key = tuple(sorted([nk, neighbor]))
                    if edge_key not in edges_seen:
                        edges_seen.add(edge_key)
                        weight = self._graph[nk][neighbor].get("weight", 1)
                        a_text = self._graph.nodes[nk].get("text", nk)
                        b_text = self._graph.nodes[neighbor].get("text", neighbor)
                        edge_parts.append(
                            f"{a_text} ↔ {b_text} ({int(weight)}×)"
                        )

        if edge_parts:
            # Sort by weight descending, take top 10
            edge_parts.sort(key=lambda x: -int(x.split("(")[1].split("×")[0]))
            lines.append("Relationships: " + ", ".join(edge_parts[:10]))

        return "\n".join(lines)

    def _top_entities_summary(self, agent_id: Optional[str], top_k: int) -> str:
        """Fallback: just list the most frequently mentioned entities."""
        nodes = [
            (nk, data) for nk, data in self._graph.nodes(data=True)
            if str(nk).startswith(f"{agent_id}:")
        ]
        if not nodes:
            return ""

        top = sorted(nodes, key=lambda x: x[1].get("mention_count", 0), reverse=True)[:top_k]
        parts = [
            f"{d.get('text', nk)} ({d.get('entity_type', '?')}, {d.get('mention_count', 0)}×)"
            for nk, d in top
        ]
        return "[WORLD MODEL]\nKey Entities: " + ", ".join(parts)

    # ──────────────────────────────────────────────────────────────────────────
    # DB Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def _load_graph_from_db(self, agent_id: Optional[str]):
        """Load graph from SQLite into NetworkX (called once on first access)."""
        db = self.get_db()
        try:
            from memnai.db.models import KnowledgeGraphNode, KnowledgeGraphEdge

            nodes = db.query(KnowledgeGraphNode).filter(
                KnowledgeGraphNode.agent_id == agent_id
            ).all()

            id_to_key = {}
            for node in nodes:
                key = f"{agent_id}:{node.entity_text}"
                id_to_key[node.id] = key
                self._graph.add_node(
                    key,
                    text=node.entity_text,
                    entity_type=node.entity_type,
                    mention_count=node.mention_count,
                    agent_id=agent_id,
                )

            node_ids = {n.id for n in nodes}
            edges = db.query(KnowledgeGraphEdge).filter(
                KnowledgeGraphEdge.source_id.in_(node_ids),
                KnowledgeGraphEdge.target_id.in_(node_ids),
            ).all()

            for edge in edges:
                src_key = id_to_key.get(edge.source_id)
                tgt_key = id_to_key.get(edge.target_id)
                if src_key and tgt_key:
                    self._graph.add_edge(src_key, tgt_key, weight=edge.weight)

            self._loaded_agents.add(agent_id)
            logger.debug(
                f"[KnowledgeGraph] Loaded {len(nodes)} nodes, {len(edges)} edges "
                f"for agent={agent_id}"
            )

        except Exception as e:
            logger.warning(f"[KnowledgeGraph] DB load failed: {e}")
        finally:
            db.close()

    def _get_nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "spaCy model not found. Run: python -m spacy download en_core_web_sm"
                )
        return self._nlp
