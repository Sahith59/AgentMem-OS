"""
AgentMem OS — Fact→Entity Linker (Consolidation v2, Stage 3)
=============================================================
Links semantic facts to KG nodes so "facts about entity X" is an indexed
query, cross-lingually. The join table (semantic_fact_entities) is the
QUERY path; SemanticFact.entities (JSON) is a display cache only —
SQLite cannot index array membership (one index entry per row, ever).

Design decisions of record (CONSOLIDATION_V2_BUILD_LOG.md, Stage 3):

  - Links point at the SURFACE-form node, never an alias-canonical node.
    Graphiti canonical-links because its dedup MERGES nodes; our τ=0.90
    alias resolver has a MEASURED false positive (Chennai/China 0.9010),
    so canonical-linking would hard-wire that error class into fact
    provenance. Cross-lingual unity comes from ALIAS_OF traversal in
    facts_for_entity() instead — a false alias stays inspectable
    retrieval noise, never silent misattribution.
  - The linker CREATES nodes (not just links): benchmark pipelines write
    Turn rows directly and bypass ConversationStore.save_turn, so the KG
    may be EMPTY when consolidation runs — facts are the feeder there.
    Creating from a fact does NOT bump an existing node's mention_count:
    that column counts TURN mentions and a distilled restatement of the
    same turns must not inflate it.
  - Indic-script surfaces pass the SAME alias gate as turn ingestion
    (db/entity_aliases.py): they become nodes only when they alias-match
    a known entity at τ — otherwise skipped and recorded. Without
    multilingual NER we cannot tell a new Hindi entity from a common
    Hindi word (same rationale, same τ, same non-destructive ALIAS_OF
    edge).
  - Fact storage NEVER fails because linking failed. Links are
    recoverable metadata: failures are counted, reported, AND persisted
    (ConsolidationLog.link_failure), and the cursor-paged link_missing()
    drain visits each candidate once per drain (crash between
    facts-commit and linking, pre-Stage-3 backfill, and the
    orphaned-entity class Graphiti #1083 shows cleanup must be able to
    reach). TWO recovery depths, precisely bounded (G3 R3 B1: the
    zero-link sweep alone was sold as recovering skipped surfaces — it
    does not, a fact that linked ANY surface was never a candidate):
    the default sweep recovers UNLINKED FACTS only; recovering skipped
    SURFACES on partially-linked facts (e.g. an Indic mention whose
    anchor arrived later) requires link_missing(rescan_all=True), which
    re-plans every fact in scope. Zero-surface facts (~22% of the real
    G2 sample) remain candidates on every default drain — visible cost,
    disclosed, never a starvation.
  - Planning (resolver model load + embeddings) is SEPARATED from
    applying: plan_surfaces() runs before the batch transaction,
    link_fact(plan=) writes under the lock. Caller-owned sessions
    REFUSE to plan in-session (G3 R1 B1: an in-batch model load held
    the DB-wide write lock ~87s and killed a competing writer).

Concurrency model (Stage-1 discipline — guarded at the DATABASE):

  - Caller-owned sessions (db=): the consolidation batch has already
    flushed the fact row in THIS transaction, so the SQLite write lock
    (RESERVED) is held until commit — no other writer can interleave,
    and check-then-insert inside the batch is cross-process safe (the
    exact R4-1 mechanism the fact store's re-affirmation uses). A
    caller-owned IntegrityError therefore means a real bug or a caller
    that broke the precondition (linked before any write) — it stays
    LOUD; the store never commits or closes a caller's session.
  - Store-owned sessions (link_missing, standalone link_fact): the
    check runs under a read snapshot, so a cross-process race IS
    possible; the UNIQUE guards (uq_kg_nodes_scope_text, uq_fact_entity)
    make the DB the authority and the owned transaction retries once
    after rollback + re-read. No savepoints — pysqlite's transactional
    quirks make SAVEPOINT the least-trustworthy tool in this stack, and
    the two shapes above cover both ownership modes without it.
"""

from datetime import datetime

from loguru import logger
from sqlalchemy.exc import IntegrityError

# entity_aliases is ALWAYS importable (pure re/os/threading at module
# level — the heavy sentence-transformers dependency is lazy inside it
# and reports resolver.enabled=False when absent). Importing it here
# instead of defensively at call sites keeps ONE behavior for "resolver
# unavailable": Indic surfaces are SKIPPED, never admitted unvetted — a
# call-time except path that fell back to "treat as non-Indic" would
# quietly bypass the alias gate exactly when the machinery is broken.
from agentmem_os.db.entity_aliases import contains_indic, get_alias_resolver
from agentmem_os.db.knowledge_graph import extract_entity_mentions
from agentmem_os.db.semantic_facts import make_scope_key

_ALIAS_TYPE = "FOREIGN"   # matches turn-path alias admissions
_ALIAS_CLOSURE_CAP = 10   # alias-closure depth guard (corrupt-data stop)
_FACT_ID_BOUND = 30000    # facts_for_entity IN() guard (param limit)
_CLOSURE_NODE_BOUND = 5000  # alias-closure node-set guard, loud


def _is_link_race(err: IntegrityError) -> bool:
    """True only for the two Stage-3 dedup guards — anything else (FK
    violation, NOT NULL) is a real failure and must stay loud. SQLite
    names COLUMNS for table constraints but the INDEX NAME for
    functional unique indexes — match all spellings."""
    msg = str(err.orig)
    return (
        "uq_kg_nodes_scope_text" in msg
        or "uq_fact_entity" in msg
        or ("UNIQUE constraint failed" in msg
            and ("kg_nodes" in msg or "semantic_fact_entities" in msg))
    )


class FactEntityLinker:
    """Constructed with a session factory (same pattern as
    SemanticFactStore / EntityKnowledgeGraph) so tests can bind an
    isolated engine."""

    def __init__(self, get_db_session):
        self.get_db = get_db_session

    # ── Extraction (pure compute — call BEFORE opening a batch txn) ─────────

    @staticmethod
    def extract_surfaces(fact_text: str) -> list:
        """(surface, entity_type) pairs from canonical fact text, via the
        SAME module-level NER the turn path uses — one entity space.

        Indic-script tokens NER cannot see (en_core_web_sm) are appended
        as FOREIGN candidates — the turn path's _augment_with_script_
        aliases rule, without which a Hindi fact text is entity-INVISIBLE
        (zero links, alias gate never consulted). extract_script_tokens
        is pure regex — no model loads here; ADMISSION is still decided
        by the alias gate in _plan_surfaces (exact node or τ-match), so
        common Hindi words never flood the graph."""
        from agentmem_os.db.entity_aliases import EntityAliasResolver

        surfaces = extract_entity_mentions(fact_text)
        seen = {s.casefold() for s, _ in surfaces}
        for tok in EntityAliasResolver.extract_script_tokens(fact_text):
            if tok.casefold() not in seen:
                surfaces.append((tok, _ALIAS_TYPE))
        return surfaces

    # ── Write path ──────────────────────────────────────────────────────────

    def plan_surfaces(self, surfaces: list, agent_id: str = None) -> dict:
        """
        Compute the linking decisions for pre-extracted surfaces —
        reads + resolver compute ONLY, no writes. Returns
        {"agent_id":…, "plan":[(surface, etype, via, anchor, conf)…],
        "skipped":[(surface, reason)…]} — the ONLY value link_fact's
        plan= accepts (shape and agent scope are validated at apply,
        G3 R2 M1).

        G3 R1 B1: this is where the sentence-transformers model may
        LOAD (~87s measured cold) and embeddings run; callers holding
        the SQLite write lock (the consolidation batch) MUST call this
        BEFORE the transaction and hand the plan to link_fact — the
        first version planned in-batch and a competing writer died on
        busy_timeout. There is deliberately NO db= parameter (G3 R2 M5:
        accepting a caller session here would run the model load under
        that caller's write lock — the exact hazard link_fact refuses
        loudly). Plans go stale gracefully: apply re-checks node AND
        anchor existence, and a surface skipped because its anchor
        didn't exist yet (e.g. created later in the same batch) is
        picked up by link_missing(rescan_all=True) — the DEFAULT
        zero-link sweep cannot see a partially-linked fact (R3 B1).
        """
        surfaces = self._dedup_surfaces(surfaces)
        if not surfaces:
            return {"agent_id": agent_id, "plan": [], "skipped": []}
        session = self.get_db()
        try:
            plan, skipped = self._plan_surfaces(session, surfaces, agent_id)
            return {"agent_id": agent_id, "plan": plan, "skipped": skipped}
        finally:
            session.close()

    @staticmethod
    def _validate_plan(plan, agent_id):
        """Loud shape+scope validation (G3 R2 M1: apply validated
        nothing — a plan computed for agent A applied under agent Z
        would write into the wrong scope, and malformed via values
        would land in linked_via unchecked)."""
        if (not isinstance(plan, dict)
                or not {"agent_id", "plan", "skipped"} <= set(plan)):
            raise ValueError(
                "plan= must be the dict returned by plan_surfaces()")
        if plan["agent_id"] != agent_id:
            raise ValueError(
                f"plan was computed for agent {plan['agent_id']!r} but is "
                f"being applied under {agent_id!r} — scopes must match")
        for entry in plan["plan"]:
            if (not isinstance(entry, (list, tuple)) or len(entry) != 5
                    or not isinstance(entry[0], str)
                    or len(entry[0].strip()) < 2
                    or entry[0] != entry[0].strip()
                    or entry[2] not in ("ner", "alias")
                    or (entry[2] == "alias"
                        and not isinstance(entry[3], str))):
                raise ValueError(f"malformed plan entry: {entry!r}")
        for entry in plan["skipped"]:
            if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
                raise ValueError(f"malformed skipped entry: {entry!r}")
        return list(plan["plan"]), list(plan["skipped"])

    def link_fact(self, fact_id: int, fact_text: str = None,
                  surfaces: list = None, plan=None, agent_id: str = None,
                  session_id: str = None, db=None) -> dict:
        """
        Link one fact to KG nodes. Three input shapes:
          - plan= (the DICT plan_surfaces() returns — validated for
            shape and agent scope): APPLY a precomputed plan — writes
            only, no resolver compute. REQUIRED shape for caller-owned
            batches that already hold the write lock (G3 R1 B1).
          - surfaces= : plan here, then apply. Safe store-owned (reads
            and compute happen before this session's first write).
          - fact_text= : extract, plan, apply.
        Returns {"linked": [(surface, node_id, via)],
                 "skipped": [(surface, reason)], "alias_edges": int}.

        Caller-owned db: writes flush into the caller's transaction
        (precondition: the caller has already written in it — see module
        docstring); store-owned: one short transaction, one retry on a
        lost cross-process race.
        """
        if plan is not None:
            plan = self._validate_plan(plan, agent_id)
        else:
            if db is not None:
                # Not a style preference — a loud contract (G3 R1 B1):
                # in-session planning can load the alias model (~87s
                # measured) while the caller's batch holds the SQLite
                # write lock, killing competing writers on busy_timeout.
                raise ValueError(
                    "caller-owned sessions must pass plan= (from "
                    "plan_surfaces(), computed BEFORE the transaction) — "
                    "planning may load the alias model under your write lock")
            if surfaces is None:
                if fact_text is None:
                    raise ValueError("pass fact_text or surfaces")
                surfaces = self.extract_surfaces(fact_text)
            surfaces = self._dedup_surfaces(surfaces)
            if not surfaces:
                return {"linked": [], "skipped": [], "alias_edges": 0}

        if db is not None:
            return self._link_in_session(db, False, fact_id, surfaces,
                                         agent_id, session_id, plan)
        session = self.get_db()
        try:
            try:
                return self._link_in_session(session, True, fact_id,
                                             surfaces, agent_id,
                                             session_id, plan)
            except IntegrityError as e:
                if not _is_link_race(e):
                    session.rollback()
                    raise
                # Lost a cross-process race on a node or link row — the
                # DB is the dedup authority; re-read and go again. A
                # second loss in a row is not a race pattern, it is a
                # bug: stay loud (no retry loop — Stage-1 lesson).
                session.rollback()
                logger.debug("[FactEntityLinker] insert race → retry once")
                return self._link_in_session(session, True, fact_id,
                                             surfaces, agent_id,
                                             session_id, plan)
        finally:
            session.close()

    def _link_in_session(self, session, owns, fact_id, surfaces,
                         agent_id, session_id, precomputed=None) -> dict:
        from agentmem_os.db.models import SemanticFact, SemanticFactEntity

        fact = session.query(SemanticFact).filter(
            SemanticFact.id == fact_id).first()
        if fact is None:
            raise ValueError(f"fact {fact_id} not found")

        if precomputed is not None:
            plan, skipped = precomputed
            # COPY — the same validated tuple is reused by the
            # store-owned retry path, and apply appends anchor-miss
            # skips; without the copy a retry double-appends (R3 minor).
            skipped = list(skipped)
        else:
            # Store-owned only: reads + compute run BEFORE this
            # session's first write, so no write lock is held during a
            # model load. Caller-owned batches must pass plan= (B1).
            plan, skipped = self._plan_surfaces(session, surfaces, agent_id)

        linked, alias_edges = [], 0
        for surface, etype, via, anchor_text, confidence in plan:
            anchor = None
            if via == "alias":
                # Anchor FIRST, node second (G3 R2 M1): the surface was
                # admitted BECAUSE it τ-matched this anchor; if the
                # anchor is gone by apply time (plan staleness, or a
                # resolver that returned junk), admitting anyway would
                # create an UNGATED Indic node — the exact junk class
                # the gate exists to keep out. Skip loudly instead; the
                # rescan_all=True sweep re-plans against fresh
                # state later (the default zero-link sweep cannot
                # see a partially-linked fact — R3 B1).
                anchor = self._get_node(session, agent_id, anchor_text)
                if anchor is None:
                    skipped.append(
                        (surface, "alias anchor not found at apply"))
                    continue
            node = self._get_or_create_node(
                session, agent_id, surface, etype, session_id)
            existing_link = (
                session.query(SemanticFactEntity)
                .filter(SemanticFactEntity.fact_id == fact_id,
                        SemanticFactEntity.node_id == node.id)
                .first()
            )
            if existing_link is None:
                session.add(SemanticFactEntity(
                    fact_id=fact_id, node_id=node.id, surface_text=surface,
                    linked_via=via, confidence=confidence,
                ))
                session.flush()
                linked.append((surface, node.id, via))
            if anchor is not None:
                alias_edges += self._ensure_alias_edge(
                    session, node, anchor, confidence, session_id)

        if owns:
            session.commit()
        return {"linked": linked, "skipped": skipped,
                "alias_edges": alias_edges}

    @staticmethod
    def _dedup_surfaces(surfaces) -> list:
        seen, out = set(), []
        for surface, etype in surfaces:
            surface = (surface or "").strip()
            if len(surface) >= 2 and surface.casefold() not in seen:
                seen.add(surface.casefold())
                out.append((surface, etype))
        return out

    def _plan_surfaces(self, session, surfaces, agent_id):
        """
        Decide per surface: link directly ('ner'), link through the
        alias gate ('alias', with the matched anchor + similarity), or
        skip (recorded reason). Indic-script surfaces follow the SAME
        gate as turn ingestion: an exact node means it already passed
        the gate when created; otherwise it must alias-match a known
        entity at τ or it does not enter the graph.
        """
        from agentmem_os.db.models import KnowledgeGraphNode

        resolver = get_alias_resolver()
        plan, skipped = [], []
        candidates = None
        for surface, etype in surfaces:
            if not contains_indic(surface):
                plan.append((surface, etype, "ner", None, None))
                continue
            exists = (
                session.query(KnowledgeGraphNode.id)
                .filter(KnowledgeGraphNode.agent_id == agent_id,
                        KnowledgeGraphNode.entity_text == surface)
                .first()
            )
            if exists is not None:
                plan.append((surface, etype, "ner", None, None))
                continue
            if not resolver.enabled:
                skipped.append((surface, "indic surface, alias resolver unavailable"))
                continue
            if candidates is None:
                rows = (session.query(KnowledgeGraphNode.entity_text)
                        .filter(KnowledgeGraphNode.agent_id == agent_id)
                        .all())
                # STRICTER than the turn path, deliberately (G3 R1
                # corrected an earlier comment claiming parity): the turn
                # path also offers same-turn non-Indic NER spans as
                # anchors; here only STORED nodes anchor — every stored
                # Indic node already passed this gate when it was
                # created, and a fact's own unvetted surfaces never
                # bootstrap each other.
                candidates = {r[0] for r in rows}
            matches = resolver.find_aliases(surface, candidates - {surface})
            if matches:
                anchor, sim = matches[0]
                plan.append((surface, _ALIAS_TYPE, "alias", anchor, sim))
            else:
                skipped.append((surface, "indic surface, no alias anchor at tau"))
        return plan, skipped

    @staticmethod
    def _get_or_create_node(session, agent_id, surface, etype, session_id):
        """Check-then-insert node upsert. Safe under a held write lock
        (caller batch); a store-owned race surfaces as IntegrityError on
        uq_kg_nodes_scope_text and is retried by link_fact. Existing
        nodes are NOT mutated (mention_count counts turn mentions)."""
        from agentmem_os.db.models import KnowledgeGraphNode

        node = (
            session.query(KnowledgeGraphNode)
            .filter(KnowledgeGraphNode.agent_id == agent_id,
                    KnowledgeGraphNode.entity_text == surface)
            .first()
        )
        if node is None:
            node = KnowledgeGraphNode(
                agent_id=agent_id, session_id=session_id,
                entity_text=surface, entity_type=etype, mention_count=1,
            )
            session.add(node)
            session.flush()   # need node.id; unique index guards the race
        return node

    @staticmethod
    def _get_node(session, agent_id, entity_text):
        from agentmem_os.db.models import KnowledgeGraphNode
        if entity_text is None:
            return None
        return (
            session.query(KnowledgeGraphNode)
            .filter(KnowledgeGraphNode.agent_id == agent_id,
                    KnowledgeGraphNode.entity_text == entity_text)
            .first()
        )

    @staticmethod
    def _ensure_alias_edge(session, node, anchor, confidence,
                           session_id) -> int:
        """ALIAS_OF edge to the (already-resolved) anchor node,
        non-destructive, ordered pair, created only if absent —
        byte-compatible with the turn path's _ingest_alias_edges
        (kg_edges has no unique constraint; a cross-process double
        insert of the same alias pair is harmless duplicate metadata,
        disclosed, same as the turn path)."""
        from agentmem_os.db.models import KnowledgeGraphEdge

        if anchor.id == node.id:
            return 0
        a, b = (node.id, anchor.id) if node.id < anchor.id else (anchor.id, node.id)
        exists = (
            session.query(KnowledgeGraphEdge)
            .filter(KnowledgeGraphEdge.source_id == a,
                    KnowledgeGraphEdge.target_id == b,
                    KnowledgeGraphEdge.relation_type == "ALIAS_OF")
            .first()
        )
        if exists is not None:
            return 0
        session.add(KnowledgeGraphEdge(
            source_id=a, target_id=b, weight=1.0, session_id=session_id,
            relation_type="ALIAS_OF", confidence=confidence,
            valid_from=datetime.utcnow(), valid_until=None,
        ))
        session.flush()
        return 1

    # ── Read path ───────────────────────────────────────────────────────────

    def facts_for_entity(self, entity_text: str, agent_id: str = None,
                         user_id: str = None, follow_aliases: bool = True,
                         fact_type: str = None, limit: int = 100,
                         db=None) -> list:
        """
        Live facts linked to an entity, timeline-ordered (t_occurred asc,
        undated last — matches facts_overlapping's convention). Seed
        resolution: case-insensitive node match for (agent scope, text);
        an Indic-script query with no matching node falls back to the
        alias resolver (the turn path's _alias_seed_fallback rule —
        Latin queries never enter the embedding path, so English
        behavior is deterministic). follow_aliases then takes the FULL
        ALIAS_OF closure (depth-capped), so a Hindi query reaches facts
        linked under the English surface, through variant chains, and
        vice versa. Fact rows are additionally scope-filtered by
        (agent_id, user_id) — node scope is the agent axis only.
        """
        from agentmem_os.db.models import (
            KnowledgeGraphEdge, KnowledgeGraphNode, SemanticFact,
            SemanticFactEntity,
        )

        if not entity_text or not str(entity_text).strip():
            raise ValueError("entity_text must be a non-empty string")
        from sqlalchemy import func, or_

        session, owns = self._acquire(db)
        try:
            # Case-insensitive seed (G3 R1 M4: 1.5% of real KG groups
            # are case variants — "The Big Island"/"the Big Island" —
            # and node identity is case-SENSITIVE by turn-path parity;
            # the READ side must not go blind over casing). The EXACT
            # arm is load-bearing, not redundant (G3 R2 B1: SQLite's
            # lower() folds ASCII ONLY, Python's folds Unicode — a
            # lower()==lower() comparison alone turned byte-identical
            # non-ASCII queries like 'Übermensch' into silent empties,
            # 11 affected entity_texts in the real KG). Case folding is
            # therefore ASCII-only, disclosed; non-ASCII variants match
            # exactly or through ALIAS_OF.
            seeds = (
                session.query(KnowledgeGraphNode.id)
                .filter(KnowledgeGraphNode.agent_id == agent_id)
                .filter(or_(
                    KnowledgeGraphNode.entity_text == entity_text,
                    func.lower(KnowledgeGraphNode.entity_text)
                    == entity_text.lower()))
                .all()
            )
            node_ids = {s[0] for s in seeds}
            if not node_ids:
                node_ids = self._alias_seed_nodes(session, entity_text, agent_id)
            if not node_ids:
                return []
            if follow_aliases:
                # ALIAS_OF CLOSURE, not one hop (G3 R1 M3: a real chain
                # चेन्नई—चेन्नई।—Chennai left one-hop reads returning a
                # SUBSET of a merged entity's facts — silent partial
                # unity is worse than none). Alias subgraphs are tiny;
                # the depth cap is a corrupt-data guard, same idiom as
                # the supersession chain walk.
                frontier = set(node_ids)
                for _ in range(_ALIAS_CLOSURE_CAP):
                    if not frontier:
                        break
                    edges = (
                        session.query(KnowledgeGraphEdge.source_id,
                                      KnowledgeGraphEdge.target_id)
                        .filter(KnowledgeGraphEdge.relation_type == "ALIAS_OF")
                        .filter((KnowledgeGraphEdge.source_id.in_(frontier))
                                | (KnowledgeGraphEdge.target_id.in_(frontier)))
                        .all()
                    )
                    new = {n for s, t in edges for n in (s, t)} - node_ids
                    node_ids |= new
                    frontier = new
                    if len(node_ids) > _CLOSURE_NODE_BOUND:
                        # R4-m4: the closure feeds IN() clauses — a
                        # pathological alias blob must stop LOUDLY, not
                        # blow the parameter budget.
                        logger.warning(
                            f"[FactEntityLinker] ALIAS_OF closure for "
                            f"{entity_text!r} exceeded "
                            f"{_CLOSURE_NODE_BOUND} nodes — "
                            "stopping expansion; results may be partial")
                        frontier = set()
                if frontier:
                    # Cap exhausted with expansion still in progress —
                    # results are a truncated subset. Never silent (G3
                    # R2 minor): a >10-deep alias chain is corrupt-data
                    # territory, but the reader must know the read was
                    # partial.
                    logger.warning(
                        f"[FactEntityLinker] ALIAS_OF closure for "
                        f"{entity_text!r} truncated at depth "
                        f"{_ALIAS_CLOSURE_CAP} — results may be partial")

            # Defensive bound (R3 minor): SQLite's parameter limit
            # is finite and an unbounded IN() list grows with entity
            # popularity. 30k covers any realistic entity; truncation
            # is LOUD.
            fact_ids = [r[0] for r in (
                session.query(SemanticFactEntity.fact_id)
                .filter(SemanticFactEntity.node_id.in_(node_ids))
                .distinct().limit(_FACT_ID_BOUND).all())]
            if len(fact_ids) == _FACT_ID_BOUND:
                logger.warning(
                    f"[FactEntityLinker] facts_for_entity({entity_text!r}) "
                    f"hit the {_FACT_ID_BOUND} fact-id bound — "
                    "results may be partial")
            if not fact_ids:
                return []
            scope_key = make_scope_key(agent_id, user_id)
            q = (
                session.query(SemanticFact)
                .filter(SemanticFact.id.in_(fact_ids),
                        SemanticFact.superseded_by.is_(None),
                        SemanticFact.scope_key == scope_key)
            )
            if fact_type is not None:
                q = q.filter(SemanticFact.fact_type == fact_type)
            # nullslast explicitly: SQLite's ASC default is NULLS FIRST,
            # which would open every timeline with the undated facts.
            return (q.order_by(SemanticFact.t_occurred.asc().nullslast(),
                               SemanticFact.id.asc())
                    .limit(limit).all())
        finally:
            if owns:
                session.close()

    def _alias_seed_nodes(self, session, entity_text, agent_id) -> set:
        """Indic-script query text with no exact node → best alias-matched
        node ids. Empty set when the resolver is off or nothing clears τ."""
        from agentmem_os.db.models import KnowledgeGraphNode

        if not contains_indic(entity_text):
            return set()
        resolver = get_alias_resolver()
        if not resolver.enabled:
            return set()
        rows = (session.query(KnowledgeGraphNode.id,
                              KnowledgeGraphNode.entity_text)
                .filter(KnowledgeGraphNode.agent_id == agent_id)
                .all())
        by_text = {}
        for nid, text_ in rows:
            by_text.setdefault(text_, nid)
        matches = resolver.find_aliases(entity_text, set(by_text))
        return {by_text[matches[0][0]]} if matches else set()

    # ── Recovery sweep ──────────────────────────────────────────────────────

    def link_missing(self, agent_id: str = None, user_id: str = None,
                     limit: int = 500, after_id: int = 0,
                     rescan_all: bool = False) -> dict:
        """
        Recovery sweep, TWO depths (G3 R3 B1 — the depths must never be
        conflated: the shallow sweep was sold as recovering skipped
        surfaces and it structurally cannot):

          rescan_all=False (default) — facts with ZERO join rows: crash
          recovery (facts committed, linking never ran) and pre-Stage-3
          backfill. Cheap. Does NOT revisit partially-linked facts.

          rescan_all=True — EVERY fact in scope re-plans against
          current graph state: recovers surfaces skipped earlier (an
          Indic mention whose alias anchor arrived later, a stale-plan
          skip). link_fact is idempotent, so relinking is safe; cost is
          NER + planning across the scope — run it after anchor-bearing
          ingests, not per-session.

        Each fact links in its own transaction (a bad fact cannot take
        the sweep down); per-fact failures are recorded, not raised.

        CURSOR-paged (G3 R1 B3): a fact whose NER finds nothing — 22%
        of real facts in the G2 sample — stays at zero join rows
        forever, so the earlier LIMIT-without-cursor shape re-swept the
        same zero-surface facts every call and facts past the limit
        were NEVER reached, and "rerun until swept=0" never terminated.
        Drain loop of record:
            cursor = 0
            while True:
                r = linker.link_missing(after_id=cursor, ...)
                if r["candidates"] == 0: break
                cursor = r["next_after_id"]
        One full drain visits every candidate once PER DRAIN (ids are
        monotonic within a drain; SQLite may reuse a deleted max-rowid
        without AUTOINCREMENT, so "once ever" is not claimed).
        """
        from sqlalchemy import exists
        from agentmem_os.db.models import SemanticFact, SemanticFactEntity

        scope_key = make_scope_key(agent_id, user_id)
        session = self.get_db()
        try:
            q = (
                session.query(SemanticFact.id, SemanticFact.fact_text,
                              SemanticFact.source_session_id)
                .filter(SemanticFact.scope_key == scope_key,
                        SemanticFact.id > after_id)
            )
            if not rescan_all:
                q = q.filter(~exists().where(
                    SemanticFactEntity.fact_id == SemanticFact.id))
            unlinked = q.order_by(SemanticFact.id.asc()).limit(limit).all()
        finally:
            session.close()

        swept = linked_total = 0
        failures = []
        for fid, text_, sess_id in unlinked:
            try:
                report = self.link_fact(fid, fact_text=text_,
                                        agent_id=agent_id, session_id=sess_id)
                linked_total += len(report["linked"])
                swept += 1
            except Exception as e:
                failures.append((fid, str(e)))
                logger.warning(f"[FactEntityLinker] sweep failed on fact "
                               f"{fid}: {e}")
        return {"candidates": len(unlinked), "swept": swept,
                "links_created": linked_total, "failures": failures,
                "next_after_id": (unlinked[-1][0] if unlinked else after_id)}

    # ── Session plumbing (same contract as SemanticFactStore) ───────────────

    def _acquire(self, db):
        if db is not None:
            return db, False
        return self.get_db(), True
