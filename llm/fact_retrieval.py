"""
AgentMem OS — Facts-first retrieval (Consolidation v2, Stage 5).

The read half of the semantic tier: rank the scope's CURRENT facts
against a query and render them as the context assembler's first
semantic block — dated, chronologically presented, supersession-aware.

Ranking design (grounded in this repo's measurements, not taste):

  PRIMARY — lexical TF-IDF cosine over fact texts, in the exact
  configuration that won the 6-variant LoCoMo evidence diagnostic
  (benchmarks/real_code_utils.install_best_chroma: plain TF-IDF 11/30
  vs dense 7/30, and rank-fusing a weak signal into the strong lexical
  one DILUTED it to 7/30). That dilution measurement is why there is
  NO rank fusion here.

  RECALL FLOOR — facts entity-linked to the query's surfaces through
  Stage 3's join table with full ALIAS_OF closure are APPENDED after
  the lexical ranking when lexical missed them, ordered by
  query-surface multiplicity then recency. They fill budget the
  lexical arm left unclaimed and never displace a lexical hit. This
  arm is the reach lexical cannot have: a Devanagari query finds
  English facts through alias closure with zero shared tokens.

Caps, all disclosed: the lexical arm scans the newest
_LEXICAL_SCAN_CAP current facts by EVENT date with NULLs last —
which means undated facts (structurally most of the preference and
identity classes) are the first cut at scale and then rank through
the entity floor only (G3 R1 m3; queued in the open-items ledger);
the entity floor reads at most _QUERY_SURFACE_CAP query surfaces,
_PER_ENTITY_CAP facts each, through Stage 3's own closure bounds.

Read-only by construction: no writes anywhere on this path. The B1
lesson, precisely scoped (G3 R1 M7 broke the earlier "no model load
can EVER sit under a database lock" wording): the shared NER load
runs BEFORE any DB session opens; the one residual is the Indic
alias-resolver fallback inside facts_for_entity, which can cold-load
the sentence encoder while that read session is open — a bounded
stall under a WAL READ lock (writers are not blocked), reachable
only by Indic-script surfaces with no exact node match. Disclosed,
not denied.
"""

import re

from loguru import logger

_LEXICAL_SCAN_CAP = 500
# Surface candidates are CHEAP to probe (one indexed seed lookup each;
# Latin surfaces never enter the embedding path), so the cap is generous
# — G3 R1 M2 measured the old cap of 8 spending 5 slots on merged
# multi-word spans and dropping 9 of the 10 sub-words the fix existed
# to produce.
_QUERY_SURFACE_CAP = 24
_PER_ENTITY_CAP = 100
_LEXICAL_FLOOR = 0.01  # the champion adapter's own cutoff
# ContextAssembler._fit_to_budget's char fast path cuts at
# token_budget × 4 BEFORE its token check (G3 R4 B1) — the rendered
# block must satisfy that gate too, or long-word prose (~5.9
# chars/token measured) gets head-cut despite being token-compliant.
# COUPLING: if the assembler's factor ever changes, the high-ratio
# sweep pin (test_rank0_survives_high_ratio_content) goes red. The
# measured ratios are a property of TokenCounter's default ENCODING
# too (gpt-4o → o200k_base, G3 R5 n3) — a counter default change
# means re-measuring them, and counter parity itself is pinned by
# test_token_counters_share_the_unit.
_CALLER_CHAR_FACTOR = 4

# Fact text is LLM-extracted from user turns — it is UNTRUSTED for
# rendering purposes (G3 R1 M1: an embedded newline forged a whole
# ranked LINE with a fabricated date/type; an embedded "[change
# history:" forged a supersession story). What _sanitize neuters is
# this renderer's machine-parsable vocabulary: line structure
# (whitespace incl. zero-width chars collapses — G3 R2 m3 measured a
# ZWSP inside the marker bypassing \s+), the history marker (bracket
# homoglyphs included), and inline [YYYY/MM/DD] date stamps (rewritten
# to parentheses so a mid-line stamp can't impersonate a line's
# authoritative leading stamp). DISCLOSED RESIDUAL: the CONTENT may
# still say anything — it is user-derived prose; lies inside the text
# are extraction-validation's problem, not the renderer's.
_ZERO_WIDTH_RE = re.compile("[\\u200b\\u200c\\u200d\\u2060\\ufeff]")
_HISTORY_MARKER_RE = re.compile(
    r"[\[［❲【〔]\s*change\s+history\s*:", re.IGNORECASE)
_INLINE_STAMP_RE = re.compile(
    r"\[\s*(?:noted\s+)?(\d{4}/\d{2}/\d{2})\s*\]")


def _sanitize(text: str) -> str:
    text = _ZERO_WIDTH_RE.sub("", text)
    text = " ".join(text.split())
    text = _HISTORY_MARKER_RE.sub("(change history:", text)
    return _INLINE_STAMP_RE.sub(r"(\1)", text)


class FactRetriever:
    """Rank and render current semantic facts for a query. Strictly
    read-only; every failure propagates loudly to the caller (the
    assembler decides containment policy, not this module)."""

    def __init__(self, get_db_session):
        self.get_db = get_db_session
        self._store = None
        self._linker = None
        self._counter = None

    # ── Retrieval ───────────────────────────────────────────────────────────

    def retrieve(self, query: str, agent_id: str = None,
                 user_id: str = None) -> list:
        """Ranked current facts for the query — lexical-primary,
        entity-floor (module docstring). Returns [] for empty queries,
        empty stores, and token-free queries with no linked entities.
        Superseded and cancelled facts never appear (current_facts and
        facts_for_entity both exclude them)."""
        if not query or not str(query).strip():
            return []

        # B1: surface extraction may load the shared NER model — it runs
        # to completion before any DB session below opens.
        uniq_surfaces = self._query_surfaces(query)

        store = self._get_store()
        scan = store.current_facts(agent_id=agent_id, user_id=user_id,
                                   limit=_LEXICAL_SCAN_CAP)
        if len(scan) == _LEXICAL_SCAN_CAP:
            logger.warning(
                f"[FactRetriever] lexical scan hit the {_LEXICAL_SCAN_CAP}"
                "-fact cap — older facts rank through the entity floor only")
        ranked = self._lexical_rank(query, scan)
        seen = {f.id for f in ranked}

        if len(uniq_surfaces) > _QUERY_SURFACE_CAP:
            logger.warning(
                f"[FactRetriever] query yielded {len(uniq_surfaces)} "
                f"surfaces; entity floor reads the first "
                f"{_QUERY_SURFACE_CAP} only")
        linker = self._get_linker()
        floor = {}
        for surface in uniq_surfaces[:_QUERY_SURFACE_CAP]:
            for f in linker.facts_for_entity(
                    surface, agent_id=agent_id, user_id=user_id,
                    limit=_PER_ENTITY_CAP):
                if f.id in seen:
                    continue
                entry = floor.setdefault(f.id, [f, 0])
                entry[1] += 1

        # Multiplicity desc, then event date desc (dates are
        # 'YYYY/MM/DD' strings — lexicographic == chronological), then
        # id desc; reverse=True applies all three at once, and undated
        # facts ('' is minimal) land last.
        extras = sorted(
            floor.values(),
            key=lambda e: (e[1],
                           e[0].t_occurred or e[0].t_mentioned or "",
                           e[0].id),
            reverse=True)
        return ranked + [f for f, _ in extras]

    @staticmethod
    def _query_surfaces(query: str) -> list:
        """Candidate entity surfaces for a QUERY — deliberately more
        generous than the write path's extract_surfaces, because queries
        are fragmentary in ways turn text is not (measured in G1: bare
        'Rachel' has no sentence context so NER returns nothing, and
        adjacent names merge into one 'Rachel Priya' span that matches
        no node). Union of: NER surfaces (script tokens included), the
        turn path's own regex fallback, and the individual capitalized
        words of multi-word spans — full spans ordered first.

        Generosity is safe here because ADMISSION is still decided by
        node existence in facts_for_entity: a junk candidate ('Where')
        finds no node and reads nothing. The write path must stay
        strict — it CREATES nodes; this path only looks them up."""
        from agentmem_os.db.fact_entities import FactEntityLinker
        from agentmem_os.db.knowledge_graph import (
            _extract_entities_regex_fallback,
        )

        spans = [s for s, _ in FactEntityLinker.extract_surfaces(query)]
        spans += [s for s, _ in _extract_entities_regex_fallback(query)]
        # INTERLEAVE each span with its own sub-words (G3 R1 M2: sub-
        # words appended after ALL spans meant the surface cap spent
        # itself on merged spans — the ones that match no node — and
        # dropped the sub-words the fix existed to produce; interleaving
        # distributes the cap across entities instead).
        out = []
        for s in spans:
            out.append(s)
            if " " in s:
                out += [w for w in s.split()
                        if w[:1].isupper() and len(w) > 1]
        return list(dict.fromkeys(out))

    @staticmethod
    def _lexical_rank(query: str, facts: list) -> list:
        """TF-IDF cosine ranking in the measured-champion configuration.
        ValueError from an empty vocabulary (e.g. stopword-only fact
        texts) means NO lexical signal exists — that is a semantic
        empty, not a failure, so it returns []. Import errors and
        everything else propagate: a broken dependency must be loud."""
        texts = [f.fact_text for f in facts]
        if not texts:
            return []
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(max_features=512, sublinear_tf=True, min_df=1)
        try:
            matrix = vec.fit_transform(texts)
            q_vec = vec.transform([query])
        except ValueError:
            return []
        sims = cosine_similarity(q_vec, matrix)[0]
        order = sorted(range(len(facts)),
                       key=lambda i: (-sims[i], -facts[i].id))
        return [facts[i] for i in order if sims[i] > _LEXICAL_FLOOR]

    # ── Rendering ───────────────────────────────────────────────────────────

    def build_block(self, query: str, agent_id: str = None,
                    user_id: str = None, token_budget: int = 1000) -> str:
        """Retrieve, select to budget, render.

        Selection is RANK-based and the block NEVER exceeds
        token_budget — measured with the same TokenCounter the
        assembler truncates with, in TOKENS (G3 R2 B1: the R1 fix
        budgeted CHARS against a chars≈tokens×4 proxy; rendered fact
        blocks measure 3.68-3.84 chars/token — always below 4 — so
        near-full blocks exceeded the token budget and the assembler's
        head-keeping truncation deleted the chronologically-newest =
        rank-0 fact at 83% of swept budgets. A proxy is a fast path,
        never a contract; the budget is enforced in the unit the
        downstream cut uses). G3 R4 B1 closed the ALTERNATION, not
        just the branch: _fit_to_budget cuts TWICE — a char fast path
        (len > tokens×4, keep="head") ahead of the token check — and
        ordinary long-common-word prose measures ~5.9 chars/token, so
        a token-compliant block was still head-cut at every production
        budget (every earlier fixture sat under 4.0 chars/token — a
        hidden parameter of every sweep). The block therefore
        satisfies BOTH constraints: tokens <= token_budget AND chars
        <= token_budget × _CALLER_CHAR_FACTOR. The post-sort trim
        (_trim_to_budget) is the exact guarantee in both units; the
        fill's per-line estimates only decide candidate order. One
        disclosed exception: the rank-0 fact is always included even
        if its single line exceeds the budget — at a tiny budget this
        can overshoot by an order of magnitude (measured 15.9× at
        token_budget=20); the caller then truncates that one line's
        TAIL, which is rank-safe: one fact, head kept.

        Rank discipline on fill: stop at the first non-fit instead of
        letting a lower-ranked short fact leapfrog a higher-ranked one
        that didn't fit — the block must read as "the best evidence,
        in time order", not "the evidence that happened to be short".

        Presentation is CHRONOLOGICAL ascending, undated facts last —
        rank decides WHAT survives, time decides HOW it reads. Facts
        that superseded a predecessor render their transition line
        (the change story is the knowledge-update differentiator; CPP:
        presentation moved 0.36→0.61 with retrieval held fixed).

        Returns "" when nothing is retrieved — the caller's byte-level
        no-regress path depends on that exact contract."""
        facts = self.retrieve(query, agent_id=agent_id, user_id=user_id)
        if not facts:
            return ""

        store = self._get_store()
        counter = self._get_counter()
        with_history = self._predecessor_targets(
            [f.id for f in facts], agent_id=agent_id, user_id=user_id)

        # Incremental fill (G3 R3 M2: re-tokenizing the whole
        # accumulating block per candidate was O(n²) — measured 1150ms
        # of pure tokenization at the module's own 500-fact scan cap,
        # on the synchronous recall path). Per-line counts + 1 per
        # join estimate the running token total; chars are tracked
        # exactly (len is free); ONE exact token count decides only at
        # the boundary; the post-sort trim below is the exact
        # guarantee in both units regardless of estimate error.
        char_cap = token_budget * _CALLER_CHAR_FACTOR
        picked, lines, running, running_chars = [], [], 0, 0
        for f in facts:
            line = self._render_line(f, with_history, store)
            line_tokens = counter.count(line)
            if picked:
                if running_chars + 1 + len(line) > char_cap:
                    break
                estimate = running + 1 + line_tokens
                if estimate > token_budget:
                    exact = counter.count("\n".join(lines + [line]))
                    if exact > token_budget:
                        break
                    running = exact
                else:
                    running = estimate
                running_chars += 1 + len(line)
            else:
                running = line_tokens
                running_chars = len(line)
            picked.append((f, line))
            lines.append(line)

        picked.sort(key=lambda fl: ((fl[0].t_occurred
                                     or fl[0].t_mentioned or "~"),
                                    fl[0].id))
        return self._trim_to_budget(picked, facts, counter, token_budget)

    @staticmethod
    def _trim_to_budget(picked, ranked_facts, counter,
                        token_budget: int) -> str:
        """The EXACT budget guarantee IN BOTH UNITS, applied after the
        chronological sort: the final block must satisfy tokens <=
        token_budget AND chars <= token_budget × _CALLER_CHAR_FACTOR
        (G3 R4 B1: the caller's _fit_to_budget cuts twice, chars
        first — a block correct in tokens alone was still head-cut on
        ~5.9 chars/token prose). If either constraint fails
        (fill-time token estimates are per-line sums — BPE joins can
        land either side of them), drop the lowest-RANKED survivors —
        never the newest — until it fits. This is the working
        mechanism the fill's estimates defer to, and its drop order
        is pinned by a constructed-counter test (G3 R3 m1: with real
        o200k_base content and real counts the token trim fired at 0 of
        115 swept budgets — reachable only when estimates
        under-count, so the pin constructs that case instead of
        claiming it)."""
        block = "\n".join(line for _, line in picked)
        if len(picked) <= 1:
            return block
        char_cap = token_budget * _CALLER_CHAR_FACTOR
        rank_of = {f.id: i for i, f in enumerate(ranked_facts)}
        while len(picked) > 1 and (len(block) > char_cap
                                   or counter.count(block) > token_budget):
            worst = max(picked, key=lambda fl: rank_of[fl[0].id])
            picked.remove(worst)
            block = "\n".join(line for _, line in picked)
        return block

    @staticmethod
    def _render_line(f, with_history: set, store) -> str:
        if f.t_occurred:
            stamp = f"[{f.t_occurred}]"
        else:
            # §5.1: undated facts anchor on when they were said.
            stamp = f"[noted {f.t_mentioned}]"
        kind = f.fact_type or "fact"
        if f.event_status == "planned":
            kind += ", planned"
        line = f"{stamp} ({kind}) {_sanitize(f.fact_text)}"
        if f.id in with_history:
            transition = store.transition_text(f.id)
            if transition:
                line += f" [change history: {_sanitize(transition)}]"
        return line

    def _predecessor_targets(self, fact_ids: list, agent_id: str = None,
                             user_id: str = None) -> set:
        """Which of these facts have at least one superseded
        predecessor pointing at them. Scope-filtered as defense in
        depth (G3 R1 m4): supersede() already rejects cross-scope
        links, but a direct DB write must not leak out-of-scope text
        into the prompt through a transition line."""
        if not fact_ids:
            return set()
        from agentmem_os.db.models import SemanticFact
        from agentmem_os.db.semantic_facts import make_scope_key
        scope_key = make_scope_key(agent_id, user_id)
        db = self.get_db()
        try:
            rows = (db.query(SemanticFact.superseded_by)
                    .filter(SemanticFact.superseded_by.in_(fact_ids),
                            SemanticFact.scope_key == scope_key)
                    .distinct().all())
            return {r[0] for r in rows}
        finally:
            db.close()

    # ── Lazy collaborators ──────────────────────────────────────────────────

    def _get_store(self):
        if self._store is None:
            from agentmem_os.db.semantic_facts import SemanticFactStore
            self._store = SemanticFactStore(self.get_db)
        return self._store

    def _get_linker(self):
        if self._linker is None:
            from agentmem_os.db.fact_entities import FactEntityLinker
            self._linker = FactEntityLinker(self.get_db)
        return self._linker

    def _get_counter(self):
        if self._counter is None:
            from agentmem_os.llm.token_counter import TokenCounter
            self._counter = TokenCounter()
        return self._counter
