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


# ──────────────────────────────────────────────────────────────────────────
# Temporal KG — typed relation extraction (LAUNCH_ROADMAP.md Phase 6
# Priority 2, ported from X-MemoryArch's graph_builder.py /
# entity_ner.py::extract_relations()).
#
# Relation-trigger vocabulary is REUSED from memory/conflict_detector.py's
# _CONTRADICTION_PAIRS rather than duplicated — the roadmap calls this out
# explicitly: "Priority 1's category taxonomy and Priority 2's relation
# types are the same four concepts from two tables." Each entry there is
# (positive_pattern, negative_pattern, category); this module only needs
# the positive verb-phrase vocabulary, wrapped in a subject/object capture
# template matching X-MemoryArch's _WORKS_AT_PATTERNS structure.
# ──────────────────────────────────────────────────────────────────────────

# Relations where a subject can only have ONE active instance at a time —
# a newer edge of these types for the same (subject, relation_type)
# supersedes the previous one. CO_OCCURS is never superseded (historical
# co-occurrence is a permanent fact).
SUPERSEDABLE_RELATIONS: frozenset = frozenset({"WORKS_AT", "LIVES_AT", "STUDIES_AT"})

_RELATION_TYPE_FOR_CATEGORY = {
    "employment": "WORKS_AT",
    "location": "LIVES_AT",
    "education": "STUDIES_AT",
}
_RELATION_CONFIDENCE = {"WORKS_AT": 0.85, "LIVES_AT": 0.80, "STUDIES_AT": 0.80}

_relation_patterns: Optional[List[Tuple["re.Pattern", str]]] = None


def _build_relation_patterns() -> List[Tuple["re.Pattern", str]]:
    """
    Build (compiled_pattern, relation_type) pairs once, lazily — importing
    conflict_detector at call time (not module import time) avoids a
    circular import, since conflict_detector.py doesn't import this module
    but both live under agentmem_os and get loaded early in the app.
    """
    global _relation_patterns
    if _relation_patterns is not None:
        return _relation_patterns

    from agentmem_os.memory.conflict_detector import _CONTRADICTION_PAIRS

    patterns = []
    for positive, _negative, category in _CONTRADICTION_PAIRS:
        rtype = _RELATION_TYPE_FOR_CATEGORY.get(category)
        if not rtype:
            continue
        verb_phrase = positive.removeprefix(r"\b(").removesuffix(r")\b")
        # conflict_detector.py's employment pattern only has "work(?:ing)?"
        # — no third-person "works" form (unlike its location pattern,
        # which already has "live(?:s)?"). That asymmetry doesn't matter
        # for conflict_detector's own first-person-leaning boolean-presence
        # use, but this extractor needs third-person subject-verb-object
        # matching (the subject capture group requires a proper noun, not
        # "I") to work at all — patched here rather than in
        # conflict_detector.py itself, which is out of scope for this port
        # and tracked separately under Phase 6 Priority 1.
        if category == "employment":
            verb_phrase = verb_phrase.replace("work(?:ing)?", "work(?:s|ing)?")
        template = (
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:' + verb_phrase + r')\s+'
            r'([A-Z][A-Za-z\s&]+?)(?=[.,;\n]|\s+and\s|\s+but\s|$)'
        )
        patterns.append((re.compile(template, re.IGNORECASE), rtype))
    _relation_patterns = patterns
    return patterns


# ──────────────────────────────────────────────────────────────────────────
# Entity extraction — MODULE-LEVEL so every consumer (turn ingestion here,
# the Stage-3 fact→entity linker in db/fact_entities.py) extracts from the
# SAME NER configuration into the SAME entity space. A second extractor
# with drifted TARGET_TYPES would silently split the node space in two.
# ──────────────────────────────────────────────────────────────────────────

_shared_nlp = None

# Target entity types (unchanged from the original class-level extractor)
_TARGET_TYPES = {
    "PERSON", "ORG", "PRODUCT", "GPE",        # High signal
    "WORK_OF_ART", "EVENT", "LANGUAGE",        # Medium signal
    "FAC", "LOC",                              # Location
}


def _get_shared_nlp():
    """Lazy-load the shared spaCy model (module-wide singleton — the
    ~50MB pipeline must never load once per consumer)."""
    global _shared_nlp
    if _shared_nlp is None:
        import spacy
        try:
            _shared_nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
    return _shared_nlp


def extract_entity_mentions(text: str) -> List[Tuple[str, str]]:
    """
    (entity_text, entity_type) pairs via spaCy NER, filtered to
    _TARGET_TYPES, deduplicated casefold, regex heuristic fallback when
    spaCy is unavailable. This IS the former
    EntityKnowledgeGraph._extract_entities body, verbatim behavior.
    """
    try:
        nlp = _get_shared_nlp()
        doc = nlp(text[:5000])  # cap at 5k chars for speed
        entities = []
        seen = set()
        for ent in doc.ents:
            if ent.label_ in _TARGET_TYPES:
                clean = ent.text.strip()
                if len(clean) >= 2 and clean.lower() not in seen:
                    seen.add(clean.lower())
                    entities.append((clean, ent.label_))
        return entities
    except Exception:
        return _extract_entities_regex_fallback(text)


def _extract_entities_regex_fallback(text: str) -> List[Tuple[str, str]]:
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


class RelationMention:
    __slots__ = ("subject", "relation_type", "object_", "confidence")

    def __init__(self, subject: str, relation_type: str, object_: str, confidence: float):
        self.subject = subject
        self.relation_type = relation_type
        self.object_ = object_
        self.confidence = confidence


def extract_relations(text: str) -> List[RelationMention]:
    """
    Extract explicit typed relations (WORKS_AT/LIVES_AT/STUDIES_AT) from
    text via regex — zero LLM calls, matching conflict_detector.py's own
    zero-LLM design philosophy. Ported from X-MemoryArch's
    entity_ner.py::extract_relations(), restructured around
    conflict_detector.py's own vocabulary instead of a second pattern set.
    """
    results = []
    for pattern, rtype in _build_relation_patterns():
        for m in pattern.finditer(text):
            subj = m.group(1).strip()
            obj = m.group(2).strip().rstrip(".,;")
            if len(subj) > 1 and len(obj) > 1:
                results.append(RelationMention(
                    subject=subj, relation_type=rtype, object_=obj,
                    confidence=_RELATION_CONFIDENCE[rtype],
                ))
    return results


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

        db = self.get_db()
        try:
            from agentmem_os.db.models import KnowledgeGraphNode, KnowledgeGraphEdge

            # Cross-lingual: Indic-script tokens the English NER can't
            # reliably see (verified: en_core_web_sm catches बेंगलुरु as GPE
            # in one sentence, misses गूगल entirely in another) only become
            # entities when they alias-match an already-known entity at τ —
            # so common Hindi/Tamil words never flood the graph.
            entities, alias_pending = self._augment_with_script_aliases(
                db, entities, content, agent_id
            )
            if not entities:
                return 0

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

                    # CO_OCCURS lookup must match BOTH storage shapes
                    # (Stage-3 G3 R1 found the real bug): the ORM default
                    # writes the STRING 'CO_OCCURS' on every insert —
                    # 34,905/34,905 rows in the dev DB, zero NULL — so the
                    # old IS-NULL-only filter never found the existing
                    # edge and CREATED A NEW ROW PER CO-OCCURRENCE (1,979
                    # duplicate pairs accumulated; the in-memory loader
                    # then kept only the last row's weight). The filter
                    # still excludes typed WORKS_AT/ALIAS_OF edges, whose
                    # weight must never absorb co-occurrence counts.
                    edge = (
                        db.query(KnowledgeGraphEdge)
                        .filter(
                            KnowledgeGraphEdge.source_id == src,
                            KnowledgeGraphEdge.target_id == tgt,
                            (KnowledgeGraphEdge.relation_type.is_(None))
                            | (KnowledgeGraphEdge.relation_type == "CO_OCCURS"),
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
                        self._graph.add_edge(key_a, key_b, weight=1.0, relation_type="CO_OCCURS")

            # Typed relations (WORKS_AT/LIVES_AT/STUDIES_AT) — bi-temporal,
            # supersedable, separate from the CO_OCCURS edges above. Only
            # writes to the DB, not self._graph — invalidate the in-memory
            # cache when it changes anything so the next as_of=None query
            # reloads fresh instead of silently missing the new/superseded
            # edge (see _ingest_typed_relations' own docstring).
            typed_changed = self._ingest_typed_relations(db, node_ids, content, agent_id, session_id)
            alias_changed = self._ingest_alias_edges(db, node_ids, alias_pending, agent_id, session_id)
            if typed_changed or alias_changed:
                self._graph.clear()

            db.commit()
            return len(entities)

        except Exception as e:
            logger.warning(f"[KnowledgeGraph] ingest_turn failed: {e}")
            db.rollback()
            return 0
        finally:
            db.close()

    def _ingest_typed_relations(
        self, db, node_ids: Dict[str, int], content: str,
        agent_id: Optional[str], session_id: str,
    ) -> bool:
        """
        Extract typed relations (WORKS_AT/LIVES_AT/STUDIES_AT) from this
        turn and create/supersede bi-temporal edges. node_ids is the
        {entity_text: node.id} map already built earlier in this same
        ingest_turn() call — subject/object text from relation extraction
        is matched against it case-insensitively (the relation regex and
        the NER pass can capture slightly different casing for the same
        mention). A relation whose subject/object NER didn't also catch
        this turn is silently skipped — matches X-MemoryArch's own
        behavior (it only links relations to entities the same extraction
        pass found).

        Returns True if any typed edge was created/superseded — the
        caller (ingest_turn) uses this to invalidate the in-memory graph
        cache, since this method only writes to the DB, not self._graph
        (unlike the CO_OCCURS loop just above it in ingest_turn, which
        updates self._graph directly). Verified empirically that skipping
        this invalidation leaves the cache silently missing typed edges
        that were correctly persisted to the DB — the first version of
        this method didn't return anything and that exact bug shipped.
        """
        from agentmem_os.db.models import KnowledgeGraphEdge, KnowledgeGraphNode

        relations = extract_relations(content)
        if not relations:
            return False

        lower_lookup = {text.lower(): nid for text, nid in node_ids.items()}
        now = datetime.utcnow()
        changed = False

        for rel in relations:
            src_id = lower_lookup.get(rel.subject.lower())
            tgt_id = lower_lookup.get(rel.object_.lower())
            if src_id is None or tgt_id is None or src_id == tgt_id:
                continue

            active = None
            if rel.relation_type in SUPERSEDABLE_RELATIONS:
                active = (
                    db.query(KnowledgeGraphEdge)
                    .filter(
                        KnowledgeGraphEdge.source_id == src_id,
                        KnowledgeGraphEdge.relation_type == rel.relation_type,
                        KnowledgeGraphEdge.valid_until.is_(None),
                    )
                    .first()
                )
                if active and active.target_id == tgt_id:
                    continue  # same fact re-stated — nothing to change

            new_edge = KnowledgeGraphEdge(
                source_id=src_id, target_id=tgt_id, weight=1.0,
                session_id=session_id, relation_type=rel.relation_type,
                confidence=rel.confidence, valid_from=now, valid_until=None,
            )
            db.add(new_edge)
            db.flush()  # need new_edge.id for superseded_by

            if active:
                active.valid_until = now
                active.superseded_by = new_edge.id

            node = db.query(KnowledgeGraphNode).filter(KnowledgeGraphNode.id == src_id).first()
            if node:
                node.last_confirmed_at = now

            changed = True

        return changed

    def _augment_with_script_aliases(
        self, db, entities: List[Tuple[str, str]], content: str, agent_id: Optional[str],
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, float]]]:
        """
        Alias-gated cross-lingual extraction (db/entity_aliases.py).

        Two cases feed alias_pending [(new_text, existing_text, similarity)]:
          • Indic-script tokens NER missed — become ("token", "FOREIGN")
            entities ONLY if they alias-match a known entity at τ. A token
            matching nothing is dropped, deliberately: without multilingual
            NER we can't tell a new Hindi entity from a common Hindi word,
            so we only link mentions of entities the graph already knows.
          • Indic-script entities NER did catch (gets a node regardless) —
            still checked so the ALIAS_OF edge to the other-script node
            gets created.

        No-op (returns inputs unchanged) when the resolver is disabled or
        the multilingual extra isn't installed.
        """
        resolver = self._get_alias_resolver()
        if resolver is None or not resolver.enabled:
            return entities, []

        from agentmem_os.db.entity_aliases import contains_indic
        from agentmem_os.db.models import KnowledgeGraphNode

        ner_texts = {t for t, _ in entities}
        script_missed = [
            t for t in resolver.extract_script_tokens(content) if t not in ner_texts
        ]
        script_caught = [t for t in ner_texts if contains_indic(t)]
        if not script_missed and not script_caught:
            return entities, []

        rows = (
            db.query(KnowledgeGraphNode.entity_text)
            .filter(KnowledgeGraphNode.agent_id == agent_id)
            .all()
        )
        # Same-turn Indic NER spans are excluded from the candidate pool —
        # they haven't passed the gate themselves yet, and letting them
        # serve as anchors would let junk bootstrap junk ('की' gating in
        # by matching NER's hallucinated 'की मीटिंग' span). DB rows stay:
        # any stored Indic node already passed the gate when it was new.
        candidates = {r[0] for r in rows} | {t for t in ner_texts if not contains_indic(t)}

        alias_pending: List[Tuple[str, str, float]] = []
        for tok in script_missed:
            matches = resolver.find_aliases(tok, candidates - {tok})
            if matches:
                entities.append((tok, "FOREIGN"))
                alias_pending.append((tok, matches[0][0], matches[0][1]))
        for tok in script_caught:
            matches = resolver.find_aliases(tok, candidates - {tok})
            if matches:
                alias_pending.append((tok, matches[0][0], matches[0][1]))
            else:
                # en_core_web_sm's entity spans over Indic text are noise in
                # BOTH directions (verified: misses गूगल entirely, tags
                # 'की मीटिंग' as an entity). With resolution on, an Indic
                # span that matches nothing at τ is dropped like any other
                # unmatched script token — an NER tag doesn't exempt it.
                entities = [(t, ty) for (t, ty) in entities if t != tok]

        return entities, alias_pending

    def _ingest_alias_edges(
        self, db, node_ids: Dict[str, int],
        alias_pending: List[Tuple[str, str, float]],
        agent_id: Optional[str], session_id: str,
    ) -> bool:
        """
        Persist ALIAS_OF edges for the pairs _augment_with_script_aliases
        matched. Non-destructive by design: both nodes stay, the edge
        carries the cosine similarity in its confidence column, and
        ALIAS_OF is deliberately NOT in SUPERSEDABLE_RELATIONS — aliases
        don't expire, and _load_graph_from_db keeps them while BFS
        traverses them unconditionally (typed-edge rule), so retrieval
        needs no changes to follow them.

        Returns True if anything was written — caller invalidates the
        in-memory cache, same contract as _ingest_typed_relations (and for
        the same reason: writing DB-only while the cache diverges is the
        exact bug class that shipped once already).
        """
        if not alias_pending:
            return False

        from agentmem_os.db.models import KnowledgeGraphEdge, KnowledgeGraphNode

        now = datetime.utcnow()
        changed = False
        for new_text, existing_text, sim in alias_pending:
            src_id = node_ids.get(new_text)
            tgt_id = node_ids.get(existing_text)
            if tgt_id is None:
                row = (
                    db.query(KnowledgeGraphNode)
                    .filter(
                        KnowledgeGraphNode.entity_text == existing_text,
                        KnowledgeGraphNode.agent_id == agent_id,
                    )
                    .first()
                )
                tgt_id = row.id if row else None
            if src_id is None or tgt_id is None or src_id == tgt_id:
                continue

            a, b = (src_id, tgt_id) if src_id < tgt_id else (tgt_id, src_id)
            exists = (
                db.query(KnowledgeGraphEdge)
                .filter(
                    KnowledgeGraphEdge.source_id == a,
                    KnowledgeGraphEdge.target_id == b,
                    KnowledgeGraphEdge.relation_type == "ALIAS_OF",
                )
                .first()
            )
            if exists:
                continue

            db.add(KnowledgeGraphEdge(
                source_id=a, target_id=b, weight=1.0, session_id=session_id,
                relation_type="ALIAS_OF", confidence=sim,
                valid_from=now, valid_until=None,
            ))
            changed = True

        return changed

    def _get_alias_resolver(self):
        """Lazy, exception-safe — None when the multilingual extra is absent."""
        try:
            from agentmem_os.db.entity_aliases import get_alias_resolver
            return get_alias_resolver()
        except Exception:
            return None

    def _alias_seed_fallback(
        self, query: str, query_entities: List[Tuple[str, str]],
        agent_id: Optional[str], graph: nx.Graph,
    ) -> Set[str]:
        """
        Node keys to seed for Indic-script query mentions that have no
        exact node in the graph. Empty set when the resolver is off or
        nothing qualifies — the caller's behavior is then identical to
        the pre-cross-lingual code path.
        """
        resolver = self._get_alias_resolver()
        if resolver is None or not resolver.enabled:
            return set()

        from agentmem_os.db.entity_aliases import contains_indic

        pending = [
            t for t, _ in query_entities
            if contains_indic(t) and not graph.has_node(f"{agent_id}:{t}")
        ]
        for tok in resolver.extract_script_tokens(query):
            if not graph.has_node(f"{agent_id}:{tok}") and tok not in pending:
                pending.append(tok)
        if not pending:
            return set()

        prefix = f"{agent_id}:"
        node_texts = {
            data.get("text")
            for key, data in graph.nodes(data=True)
            if str(key).startswith(prefix) and data.get("text")
        }
        if not node_texts:
            return set()

        seeds = set()
        for tok in pending:
            matches = resolver.find_aliases(tok, node_texts)
            if matches:
                seeds.add(f"{agent_id}:{matches[0][0]}")
        return seeds

    def get_relevant_subgraph(
        self,
        query: str,
        agent_id: Optional[str],
        top_k: int = 10,
        max_hops: int = 2,
        as_of: Optional[datetime] = None,
    ) -> str:
        """
        Find entities in the query → expand subgraph up to max_hops →
        serialize to text for context assembly.

        This is what fills the 10% Global Map slot.

        as_of: point-in-time query (LAUNCH_ROADMAP.md Phase 6 Priority 2,
        ported from X-MemoryArch's graph_retrieval.py). None (default) uses
        the cached in-memory graph, which only ever holds CURRENTLY-ACTIVE
        typed edges (valid_until IS NULL) plus all CO_OCCURS edges —
        superseded typed edges are filtered out at load time, not at query
        time, so the default fast path never sees stale facts. A non-None
        as_of builds a fresh, uncached point-in-time graph on every call
        (matches X-MemoryArch's own design — point-in-time queries are
        rare one-offs, not worth the complexity of incrementally
        invalidating the cache for).

        Returns empty string if graph is empty.
        """
        if as_of is None:
            if self._graph.number_of_nodes() == 0:
                self._load_graph_from_db(agent_id)
            graph = self._graph
        else:
            graph = self._build_point_in_time_graph(agent_id, as_of)

        if graph.number_of_nodes() == 0:
            return ""

        # Find query entities
        query_entities = self._extract_entities(query)

        # Build seed nodes from query
        seed_nodes = set()
        for text, _ in query_entities:
            node_key = f"{agent_id}:{text}"
            if graph.has_node(node_key):
                seed_nodes.add(node_key)

        # Cross-lingual seed fallback: an Indic-script mention with no
        # exact node (NER-missed, or stored under another language) seeds
        # the best alias-matched node instead — a Hindi query lands on
        # memory stored in English. Latin-script query text never enters
        # this path, so English-query behavior is unchanged.
        seed_nodes |= self._alias_seed_fallback(query, query_entities, agent_id, graph)

        if not seed_nodes:
            # Fallback: return top-k most mentioned entities
            return self._top_entities_summary(agent_id, top_k, graph=graph)

        # BFS expansion up to max_hops
        subgraph_nodes = set(seed_nodes)
        frontier = set(seed_nodes)

        for _ in range(max_hops):
            next_frontier = set()
            for node in frontier:
                if graph.has_node(node):
                    neighbors = set(graph.neighbors(node))
                    # Expand to CO_OCCURS neighbors only if strongly
                    # connected (weight >= 2 — filters one-off incidental
                    # mentions) OR to any typed-relation neighbor
                    # unconditionally — a single "X works at Y" statement
                    # is an explicit, high-confidence fact (confidence
                    # 0.80-0.85) the moment it's stated, not noise that
                    # needs repetition to "prove" strength the way raw
                    # co-occurrence does. Typed edges keep weight=1.0
                    # fixed (see _ingest_typed_relations), so gating them
                    # on the same >=2 threshold as CO_OCCURS would mean a
                    # stated fact could never surface in retrieval at all.
                    strong_neighbors = {
                        n for n in neighbors
                        if n not in subgraph_nodes and (
                            graph[node][n].get("relation_type", "CO_OCCURS") != "CO_OCCURS"
                            or graph[node][n].get("weight", 0) >= 2
                        )
                    }
                    next_frontier.update(strong_neighbors)
            subgraph_nodes.update(next_frontier)
            frontier = next_frontier

            if len(subgraph_nodes) >= top_k * 2:
                break

        # Trim to top_k by mention_count
        sorted_nodes = sorted(
            subgraph_nodes,
            key=lambda n: graph.nodes[n].get("mention_count", 0),
            reverse=True,
        )[:top_k]

        return self._serialize_subgraph(sorted_nodes, agent_id, graph=graph)

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
        """Delegates to the module-level shared extractor (Stage 3
        refactor) — turn ingestion and the fact→entity linker must
        extract into ONE entity space, byte-identically."""
        return extract_entity_mentions(text)

    def _extract_entities_regex(self, text: str) -> List[Tuple[str, str]]:
        return _extract_entities_regex_fallback(text)

    # ──────────────────────────────────────────────────────────────────────────
    # Graph Serialization
    # ──────────────────────────────────────────────────────────────────────────

    def _serialize_subgraph(
        self, node_keys: List[str], agent_id: Optional[str], graph: Optional[nx.Graph] = None,
    ) -> str:
        """
        Convert a subgraph to a compact text representation for context injection.

        Output format:
          [WORLD MODEL]
          Entities: Alice (PERSON, 12×), FastAPI (PRODUCT, 5×)
          Relationships: Alice ↔ Bob (8×), Alice ↔ FastAPI (3×)
          Aliases: Google = गूगल
          Facts: Alice WORKS_AT City Hospital (since 2026-03-01)

        Typed relations (WORKS_AT/LIVES_AT/STUDIES_AT) get their own "Facts"
        line with the relation name and validity window — CO_OCCURS edges
        stay in the undifferentiated "Relationships" line as before.
        """
        graph = graph if graph is not None else self._graph
        if not node_keys:
            return ""

        lines = ["[WORLD MODEL]"]

        # Entities
        entity_parts = []
        for nk in node_keys:
            data = graph.nodes[nk]
            entity_parts.append(
                f"{data.get('text', nk)} ({data.get('entity_type','?')}, "
                f"{data.get('mention_count', 0)}×)"
            )
        lines.append("Entities: " + ", ".join(entity_parts))

        # Relationships (CO_OCCURS edges within subgraph, sorted by weight)
        # and Facts (typed relations, sorted by validity start)
        edges_seen = set()
        edge_parts = []
        fact_parts = []
        alias_parts = []
        for nk in node_keys:
            for neighbor in graph.neighbors(nk):
                if neighbor in node_keys:
                    edge_key = tuple(sorted([nk, neighbor]))
                    if edge_key in edges_seen:
                        continue
                    edges_seen.add(edge_key)
                    edata = graph[nk][neighbor]
                    a_text = graph.nodes[nk].get("text", nk)
                    b_text = graph.nodes[neighbor].get("text", neighbor)
                    rtype = edata.get("relation_type", "CO_OCCURS")
                    if rtype in SUPERSEDABLE_RELATIONS:
                        valid_from = edata.get("valid_from")
                        since = f" (since {valid_from.date()})" if valid_from else ""
                        fact_parts.append((valid_from, f"{a_text} {rtype} {b_text}{since}"))
                    elif rtype == "ALIAS_OF":
                        alias_parts.append(f"{a_text} = {b_text}")
                    else:
                        weight = edata.get("weight", 1)
                        edge_parts.append(f"{a_text} ↔ {b_text} ({int(weight)}×)")

        if edge_parts:
            edge_parts.sort(key=lambda x: -int(x.split("(")[1].split("×")[0]))
            lines.append("Relationships: " + ", ".join(edge_parts[:10]))

        if alias_parts:
            lines.append("Aliases: " + ", ".join(alias_parts[:10]))

        if fact_parts:
            fact_parts.sort(key=lambda x: x[0] or datetime.min, reverse=True)
            lines.append("Facts: " + ", ".join(f for _, f in fact_parts[:10]))

        return "\n".join(lines)

    def _top_entities_summary(
        self, agent_id: Optional[str], top_k: int, graph: Optional[nx.Graph] = None,
    ) -> str:
        """Fallback: just list the most frequently mentioned entities."""
        graph = graph if graph is not None else self._graph
        nodes = [
            (nk, data) for nk, data in graph.nodes(data=True)
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
        """
        Load graph from SQLite into NetworkX (called once on first access).

        Bi-temporal filtering (LAUNCH_ROADMAP.md Phase 6 Priority 2): loads
        ALL CO_OCCURS edges (historical co-occurrence never expires) but
        only CURRENTLY-ACTIVE typed edges (valid_until IS NULL) — a
        superseded WORKS_AT/LIVES_AT/STUDIES_AT fact is deliberately left
        out of the cached "current state" graph. Point-in-time queries use
        _build_point_in_time_graph() instead, which doesn't share this cache.
        """
        db = self.get_db()
        try:
            from agentmem_os.db.models import KnowledgeGraphNode, KnowledgeGraphEdge

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
            all_edges = db.query(KnowledgeGraphEdge).filter(
                KnowledgeGraphEdge.source_id.in_(node_ids),
                KnowledgeGraphEdge.target_id.in_(node_ids),
            ).all()
            edges = [
                e for e in all_edges
                if e.relation_type not in SUPERSEDABLE_RELATIONS or e.valid_until is None
            ]

            for edge in edges:
                src_key = id_to_key.get(edge.source_id)
                tgt_key = id_to_key.get(edge.target_id)
                if src_key and tgt_key:
                    self._graph.add_edge(
                        src_key, tgt_key, weight=edge.weight,
                        relation_type=edge.relation_type or "CO_OCCURS",
                        confidence=edge.confidence, valid_from=edge.valid_from,
                    )

            self._loaded_agents.add(agent_id)
            logger.debug(
                f"[KnowledgeGraph] Loaded {len(nodes)} nodes, {len(edges)} active edges "
                f"({len(all_edges) - len(edges)} superseded, filtered) for agent={agent_id}"
            )

        except Exception as e:
            logger.warning(f"[KnowledgeGraph] DB load failed: {e}")
        finally:
            db.close()

    def _build_point_in_time_graph(self, agent_id: Optional[str], as_of: datetime) -> nx.Graph:
        """
        Build a fresh, uncached graph reflecting the KG's state at a given
        point in time — an edge is included if it was already valid_from
        by as_of and either never superseded or not yet superseded by
        as_of. Ported from X-MemoryArch's graph_retrieval.py's own as_of
        handling; not merged into the cached self._graph since point-in-
        time queries are rare one-offs, not worth cache-invalidation
        complexity for.
        """
        graph = nx.Graph()
        db = self.get_db()
        try:
            from agentmem_os.db.models import KnowledgeGraphNode, KnowledgeGraphEdge

            nodes = db.query(KnowledgeGraphNode).filter(
                KnowledgeGraphNode.agent_id == agent_id
            ).all()
            id_to_key = {}
            for node in nodes:
                key = f"{agent_id}:{node.entity_text}"
                id_to_key[node.id] = key
                graph.add_node(
                    key, text=node.entity_text, entity_type=node.entity_type,
                    mention_count=node.mention_count, agent_id=agent_id,
                )

            node_ids = {n.id for n in nodes}
            all_edges = db.query(KnowledgeGraphEdge).filter(
                KnowledgeGraphEdge.source_id.in_(node_ids),
                KnowledgeGraphEdge.target_id.in_(node_ids),
            ).all()

            for edge in all_edges:
                if edge.relation_type in SUPERSEDABLE_RELATIONS:
                    if edge.valid_from and edge.valid_from > as_of:
                        continue  # not yet true at as_of
                    if edge.valid_until and edge.valid_until <= as_of:
                        continue  # already superseded by as_of
                src_key = id_to_key.get(edge.source_id)
                tgt_key = id_to_key.get(edge.target_id)
                if src_key and tgt_key:
                    graph.add_edge(
                        src_key, tgt_key, weight=edge.weight,
                        relation_type=edge.relation_type or "CO_OCCURS",
                        confidence=edge.confidence, valid_from=edge.valid_from,
                    )
            return graph
        except Exception as e:
            logger.warning(f"[KnowledgeGraph] point-in-time load failed: {e}")
            return graph
        finally:
            db.close()

    def _get_nlp(self):
        """Delegates to the module-level shared model (Stage 3 refactor —
        one ~50MB pipeline per process, not per consumer)."""
        return _get_shared_nlp()
