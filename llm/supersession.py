"""
AgentMem OS — Supersession Judge (Consolidation v2, Stage 4)
=============================================================
Decides which new semantic facts supersede which old ones — the
JUDGMENT layer over Stage 1's mechanism (SemanticFactStore.supersede:
atomic, cycle-guarded, invalidate-never-delete).

Design of record (CONSOLIDATION_V2_BUILD_LOG.md, Stage 4 — every rule
sourced from the research pass):

  - Shape: batched-shortlist, single-new-fact judgment (Graphiti
    v0.29.3's verified pattern). Shortlist = live facts, same scope,
    same fact_type, sharing >=1 entity node via semantic_fact_entities,
    capped at _SHORTLIST_CAP ranked by most-recent t_mentioned. No
    candidates -> NO LLM call.
  - Type scope, conservative by construction: 'state' and 'preference'
    ONLY. identity is excluded (highest cost-of-error — recorded scope
    call); events are excluded EXCEPT planned->cancelled, which the
    judgment may PROPOSE as an attribute update (no bespoke
    cancellation detector exists anywhere — we are first to mechanize,
    disclosed; reachability gated on the parked plans-as-events prompt
    decision).
  - The LLM only PROPOSES; deterministic gates DECIDE (rebuilt after
    G3 S4-R1 broke the first gate set two ways):
      1. proposed ids must be in the shortlist (else dropped, logged);
      2. TYPE guard: only same-fact_type facts supersede — planned
         events sit in the shortlist strictly as cancellation
         candidates (R1-B1: without this, the reserved planned pool
         handed the model a future-dated event that always won on
         domain time and invalidated a true current state);
      3. INDEPENDENT co-signal (R1-B2: similarity may never approve
         what similarity admitted): node-guarded polarity flip, OR the
         digit-aware metric-update signal (masked texts EXACTLY
         identical, >=2 aligned numeric tokens, EXACTLY ONE normalized
         value differs), OR — for entity-pool candidates only —
         lexical similarity >= _COSIGNAL_COSINE (llama3.1-8B alone is
         NOT defensible: zero-shot 7-8B anchors run 33.6-63.8%);
      4. DIRECTION from DOMAIN time, interval-aware and SAME-AXIS
         (R1-Ma2): occurrence intervals compare only against occurrence
         intervals (strict precedence decides; overlap is ambiguous =
         skip), otherwise the mention axis both sides always have;
         REVERSAL applies when the "old" side is domain-later —
         processing order is never trusted as truth order;
      5. supersede() and mark_event_cancelled() are the only writes,
         applied in ONE transaction WITH the audit row (R1-B3:
         applied-but-unaudited must be impossible); the model has no
         path to a delete, and every action is REVERSIBLE.
  - Reasoning-FIRST output schema, grammar-forced: {"reasoning", then
    "superseded_ids", then "cancelled_ids"} — graphiti#1666 measured
    46.7% -> 93.3% on the same small model from this ordering alone.
  - Failure bias is structural: a MISSED supersession degrades to
    today's shipped behavior (stale duplicate visible, history intact);
    a WRONG one is blocked by gates 1-3 and reversible if it lands.
  - Every judgment is PERSISTED (supersession_judgments): shortlist,
    co-signals, raw model output, applied actions, dropped proposals
    WITH reasons, or the whole-judgment skip reason. A fact with no
    judgment row has never been judged — that absence is the recovery
    sweep's candidate marker (the link_missing pattern).
  - Compute discipline (the Stage-3 R1-B1 law): judgment runs on its
    OWN sessions after the consolidation batch commits — LLM compute
    never happens under a held write lock, and there is deliberately
    no db= passthrough on the judging entry points.
"""

import json
import os
import urllib.request
from datetime import datetime

from loguru import logger

# conflict_detector is imported LAZILY (inside the functions that need
# it): its module chain reaches db.engine, whose import runs init_db()
# against whatever DB path resolves — the R3/R1 import-time-migration
# trap (G3 S4-R1 Ma7: this module added the first llm -> db.engine
# module-level edge and re-armed it; the critic's own run migrated the
# production DB again proving the point).

# Env-configurable (open-items ledger #5; load-bearing from the
# cluster run on: SLURM array tasks may share a node, so each
# task needs its own Ollama port). Default unchanged.
OLLAMA_URL = os.environ.get(
    "AGENTMEM_OS_OLLAMA_URL", "http://localhost:11434") \
    .rstrip("/") + "/api/generate"
DEFAULT_MODEL = "llama3.1:latest"
_SHORTLIST_CAP = 12
_PLANNED_SLOT_CAP = 4   # reserved cancellation-candidate slots
_LEXICAL_SCAN_CAP = 300  # newest live same-type facts scanned lexically
# Entity-pool approval threshold. HONEST PROVENANCE (R2 minor — the
# old comment mis-sourced this to forget_about, which uses keyword
# overlap and no cosine at all): 0.25 is OUR choice, 2.5x the repo's
# shipped 0.10 topic gate, applied ONLY where admission came from the
# entity join table so the similarity is not approving itself.
_COSIGNAL_COSINE = 0.25
# Deterministic cancellation cues (R2-B2: cancellation was a PURE LLM
# verdict — 'prefers oat milk' cancelled a Tokyo flight). Vocabulary-
# gated like polarity flips: the NEW text must actually SAY something
# got called off.
import re as _re_mod
_CANCEL_CUE_RE = _re_mod.compile(
    r"\b(cancel(?:l)?(?:s|ed|ing|ation)?|calls? off|called off|"
    r"calling off|postponed|fell through|scrapped|shelved|axed|"
    r"backed out|pulled out|no longer (?:going|attending|happening|"
    r"taking place)|not (?:going|happening) (?:to|anymore|after all)|"
    r"won'?t be (?:going|attending|happening))\b", _re_mod.IGNORECASE)
# Periods after 1-3-letter tokens are treated as abbreviations, not
# clause enders — ANY case (G3 S4-R6 Ma1 caught 'Dr. Meyer'; R7-B1
# caught the capitalized-only guard missing 'p.m.'/'vs.' at the same
# 5/6 false-bind rate). Splitting there DELETED the words naming the
# specific thing, and a smaller named set is strictly more likely to
# be a subset — every splitter FALSE SPLIT moves toward BINDING. The
# trade is deliberate and safe-direction: a MISSED split ('...the
# gym.' ends a sentence) makes the clause GROW into the next sentence,
# and growth adds words = the REFUSING direction (pinned both ways).
# Each lookbehind anchors at the TOKEN start (non-letter before the
# 1-3 letters) — without the anchor, every word ENDING in 2-3 letters
# blocked its sentence period ('marathon.' matched via 'on').
_CLAUSE_SPLIT_RE = _re_mod.compile(
    r"(?<![^A-Za-z][A-Za-z])(?<![^A-Za-z][A-Za-z][A-Za-z])"
    r"(?<![^A-Za-z][A-Za-z][A-Za-z][A-Za-z])\."
    r"|[;!?]|\band\b|\bbut\b")
# 4-char floor needs its own stop extension (the >=5 rule loses
# distinctive short words like 'yoga'/'rome' — R4-B1's measured
# separator lives there — but admits structural 4-char words).
_S4_EXTRA_STOP = frozenset(
    "user says said plan week year went also will when then them they "
    "from with that this have been into onto over some many much very "
    "just like each attend plans".split())


def _clause_words(text: str) -> set:
    """Content words at a 4-char floor: conflict_detector's stopwords
    plus the structural 4-char extension. Cue SPANS are removed before
    extraction (R4 fix-of-fix: per-word fullmatch let 'called' leak out
    of the two-word cue 'called off' and poison containment)."""
    from agentmem_os.memory.conflict_detector import _CONTENT_STOPWORDS
    text = _CANCEL_CUE_RE.sub(" ", text)
    return {w for w in _re_mod.findall(r"[a-z]{4,}", text.lower())
            if w not in _CONTENT_STOPWORDS and w not in _S4_EXTRA_STOP}


def _cancellation_binds(new_text: str, plan_text: str) -> tuple:
    """Deterministic cancellation binding, v3 (G3 S4-R4 B1 replaced the
    v2 shared-word COUNT, whose axis was orthogonal to correctness —
    two GENERIC shared words ('weekend training') bound false cancels
    3/3 on the real model while verbatim TRUE cancellations of
    short-named plans refused; the critic also measured that coverage
    RATIOS do not separate). The rule that does separate, on every
    measured case: CONTAINMENT —

        you can only cancel a thing you NAME, and everything the
        cancelling clause names must be part of the plan.

    Per non-NEGATED cue occurrence (every occurrence checked — R4-m1:
    checking only the first let a negated second mention through), the
    clause containing the cue is split out (sentence enders plus
    and/but conjunctions), its object words extracted at a 4-char
    floor, and the bind holds iff that set is NON-EMPTY and a SUBSET of
    the plan's words: 'cancelled the weekend training session with the
    physiotherapist' names session+physiotherapist — not in the
    marathon-camp plan → refuse; 'cancelled the Rome marathon' names
    rome+marathon — both in the plan → bind.

    Returns (binds, reason|None). Disclosed misses (mild class):
    pronoun-only cancellations ('cancelled it') name nothing and
    refuse; plans whose only distinctive words are under 4 chars are
    structurally hard to cancel.
    """
    from agentmem_os.memory.conflict_detector import _NEGATIONS

    matches = list(_CANCEL_CUE_RE.finditer(new_text))
    if not matches:
        return False, "no cancellation cue in the new fact's text"
    plan_words = _clause_words(plan_text)
    # Clause boundaries by POSITION (R5-Ma1: selecting the clause by
    # cue STRING judged every occurrence against the FIRST clause
    # containing that surface form — a negated first mention plus a
    # real second one produced a measured wrong write; span-accurate
    # negation demands span-accurate clauses).
    bounds = [0] + [b.end() for b in _CLAUSE_SPLIT_RE.finditer(new_text)] \
        + [len(new_text)]
    reasons = []
    for m in matches:
        c_start = max(b for b in bounds if b <= m.start())
        c_end = min(b for b in bounds if b > m.start())
        # The negation lookback deliberately CROSSES clause boundaries
        # (R7-Ma1 REVERTED R6-m1's clause-bounding: narrowing this
        # window can only hide negations, which is the BINDING
        # direction — 'did not cancel X and cancel Y' must refuse as
        # scope-ambiguous. Negation errs WIDE; clause naming errs
        # NARROW; the two features have OPPOSITE safe directions).
        window = new_text[max(0, m.start() - 40):m.start()]
        if _NEGATIONS.search(window):
            reasons.append("cue is NEGATED")
            continue
        clause = new_text[c_start:c_end]
        named = _clause_words(clause)
        if not named:
            reasons.append("cue clause names nothing")
            continue
        outside = named - plan_words
        if outside:
            reasons.append("cue clause names things outside the plan "
                           f"({', '.join(sorted(outside)[:4])})")
            continue
        return True, None
    return False, ("cancellation gate refused every cue occurrence: "
                   + "; ".join(reasons))


def _mask_numbers(text: str) -> str:
    """Casefolded, whitespace-collapsed text with every digit run
    (incl. , . : internals) replaced by '#'."""
    masked = _re_mod.sub(r"\d[\d,.:]*", "#", text.casefold())
    return " ".join(masked.split())


def _norm_nums(text: str) -> list:
    """Numeric tokens, comma-stripped, float-normalized where cleanly
    parseable ('2,000'=='2000', '70'=='70.0'); colon forms (25:31) stay
    strings."""
    out = []
    for tok in _re_mod.findall(r"\d[\d,.:]*", text):
        # commas strip ONLY in grouping shapes (1,234 / 12,345,678) —
        # '1,5' is a European decimal and must not collapse to 15
        # (R3-m1, reproduced end-to-end).
        if _re_mod.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", tok):
            tok = tok.replace(",", "")
        tok = tok.rstrip(".:")
        if ":" not in tok:
            try:
                tok = repr(float(tok))
            except ValueError:
                pass
        out.append(tok)
    return out


def _metric_update(new_text: str, cand_text: str) -> bool:
    """The metric-update replacement shape, DIGIT-AWARE (R2-B1: the
    first version compared TF-IDF cosines whose tokenizer never saw
    digits — 'stripped' similarity equaled plain similarity on 20k
    randomized pairs, and '5K run' was superseded by '10K run'):
    the number-MASKED texts must be EXACTLY identical ('Friday' vs
    'Monday', 'laptop' vs 'desk' refuse), the numeric token counts must
    align, and EXACTLY ONE aligned NORMALIZED value may differ — a
    metric update changes THE value; when two numbers differ, one of
    them is an identity component ('5K run' vs '10K run' masks
    identically but differs in TWO positions → a different race, not
    an update; '$2,000' vs '$2000' differs in ZERO → restatement).
    Disclosed cost: a genuine simultaneous two-value update ('120 over
    80' → '135 over 90') is missed — the chosen mild class.
    """
    if _mask_numbers(new_text) != _mask_numbers(cand_text):
        return False
    a, b = _norm_nums(new_text), _norm_nums(cand_text)
    # The length check is NOT dead (R3-Ma1: a literal '#' in the source
    # text also masks to '#', so equal masks do NOT imply equal token
    # counts — fuzz found 53/300k pairs where the zip misaligned and
    # returned True on a crooked comparison).
    if len(a) != len(b) or not a:
        return False
    # R3-Ma7: with ONE number, value-update ('child is 7'→'is 8') and
    # identity-difference ('apartment 12B'→'14B', 'Route 66'→'95') are
    # deterministically indistinguishable — and a real model proposed
    # the apartment class 3/3. Single-number pairs REFUSE; the miss
    # ('rent is 1800'→'1900') is the chosen mild class, disclosed.
    if len(a) < 2:
        return False
    return sum(1 for x, y in zip(a, b) if x != y) == 1

# Reasoning FIRST — field order in a grammar-forced schema is
# load-bearing (46.7%→93.3%, graphiti#1666).
JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "superseded_ids": {"type": "array", "items": {"type": "integer"}},
        "cancelled_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["reasoning", "superseded_ids", "cancelled_ids"],
}

_PROMPT = """You maintain a user's fact memory. A NEW fact has arrived. Below are EXISTING facts about the same entities.

Decide which EXISTING facts the NEW fact REPLACES as the current truth (the user's situation changed: new job replaces old job, new best time replaces old best time, moved cities, changed preference). Rules:
- Same situation UPDATED with different details = superseded.
- A DIFFERENT thing entirely (another topic, another event, coexisting facts like two hobbies) = NOT superseded.
- A mere restatement of the same fact = NOT superseded.
- An existing PLANNED event that the NEW fact says was called off = cancelled.
- When unsure, choose nothing — an empty list is a good answer.

NEW fact: {new_fact}

EXISTING facts:
{candidates}

Think step by step in "reasoning" FIRST, then list the ids."""


def _domain_direction(new_snapshot, cand):
    """
    Interval-aware, SAME-AXIS domain-time direction (G3 S4-R1 Ma2: the
    first version compared t_occurred-or-t_mentioned scalars, silently
    mixing axes, and a month-interval fact could lose to a point date
    INSIDE its own interval).

      - Both sides carry occurrence dates: compare INTERVALS; a side
        loses only if its interval ends strictly before the other
        begins. Overlap or touch = ambiguous = None (skip).
      - Otherwise: compare on the mention axis, which BOTH sides always
        have — one side's occurrence date is never compared against the
        other side's mention date.

    Returns (old_id, winner_id, t_invalid) or None when ambiguous.
    t_invalid = the winner's domain start on the axis that decided.
    """
    n_start, n_end = new_snapshot["occ"]
    c_start, c_end = cand["occ"]
    if n_start is not None and c_start is not None:
        if c_end < n_start:
            return (cand["id"], new_snapshot["id"], n_start)
        if n_end < c_start:
            return (new_snapshot["id"], cand["id"], c_start)
        return None
    n_m, c_m = new_snapshot["mentioned"], cand["mentioned"]
    if c_m < n_m:
        return (cand["id"], new_snapshot["id"], n_m)
    if n_m < c_m:
        return (new_snapshot["id"], cand["id"], c_m)
    return None


class SupersessionJudge:
    """Constructed with a session factory (the repo-standard pattern)
    so tests bind an isolated engine."""

    def __init__(self, get_db_session, model: str = DEFAULT_MODEL,
                 timeout: int = 600):
        self.get_db = get_db_session
        self.model = model
        self.timeout = timeout

    # ── LLM boundary (mocked in G1; real in G2) ─────────────────────────────

    def _llm(self, prompt: str) -> dict:
        import os as _os
        api_model = _os.environ.get("AGENTMEM_OS_SUPERSESSION_API_MODEL")
        if api_model:
            # F-20: cheap API judge for the high-volume yes/no calls.
            import time as _t
            body = {"model": api_model,
                    "messages": [{"role": "user", "content": prompt +
                                  "\n\nReturn ONLY a JSON object with "
                                  "EXACTLY these fields: {\"reasoning\": "
                                  "string, \"superseded_ids\": [int], "
                                  "\"cancelled_ids\": [int]}. Use empty "
                                  "lists when nothing applies."}],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 800, "temperature": 0}
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Authorization":
                         f"Bearer {_os.environ['OPENAI_API_KEY']}",
                         "Content-Type": "application/json"})
            last = None
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        out = json.loads(r.read())
                    return json.loads(out["choices"][0]["message"]["content"])
                except Exception as e:
                    last = e
                    _t.sleep(2 * (attempt + 1))
            raise ValueError(f"supersession API failed after retries: {last}")
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps({
                "model": self.model, "prompt": prompt, "stream": False,
                "format": JUDGMENT_SCHEMA,
                "options": {"temperature": 0, "num_predict": 800},
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.loads(r.read())
            if "response" not in body:
                raise ValueError(f"Ollama reply missing 'response': {body}")
            return json.loads(body["response"])

    # ── Judgment ────────────────────────────────────────────────────────────

    def judge_fact(self, fact_id: int, session_id: str = None) -> dict:
        """
        Judge ONE fact against its shortlist. Own sessions throughout
        (never under a caller's lock). Always writes exactly one
        supersession_judgments row. Returns
        {"superseded": [(old_id, new_id)], "cancelled": [ids],
         "dropped": [(id, reason)], "skipped": str|None}.
        """
        from agentmem_os.db.models import SemanticFact, SupersessionJudgment

        session = self.get_db()
        try:
            fact = session.query(SemanticFact).filter(
                SemanticFact.id == fact_id).first()
            if fact is None:
                raise ValueError(f"fact {fact_id} not found")
            skip = None
            if fact.superseded_by is not None:
                skip = "fact already superseded"
            elif fact.fact_type not in ("state", "preference"):
                skip = (f"fact_type {fact.fact_type!r} out of judgment "
                        "scope (states/preferences only; identity "
                        "excluded by design)")
            if skip:
                self._log(session, fact_id, session_id, [], None, None,
                          None, None, skip)
                session.commit()
                return {"superseded": [], "cancelled": [], "dropped": [],
                        "skipped": skip}
            shortlist = self._shortlist(session, fact)
            if not shortlist:
                # Honest skip (Ma5: the old wording blamed entities when
                # THREE pools had run and come up empty).
                skip = ("no candidates from any pool "
                        "(entity, lexical, planned)")
                self._log(session, fact_id, session_id, [], None, None,
                          None, None, skip)
                session.commit()
                return {"superseded": [], "cancelled": [], "dropped": [],
                        "skipped": skip}
            # Snapshot everything needed after the session closes —
            # judgment compute must not hold a session open. Domain time
            # is an INTERVAL (start, end) plus the mention date, so the
            # direction check can be interval-aware and same-axis (Ma2).
            new_snapshot = {
                "id": fact.id, "text": fact.fact_text,
                "type": fact.fact_type,
                "occ": (fact.t_occurred,
                        fact.t_occurred_end or fact.t_occurred),
                "mentioned": fact.t_mentioned,
            }
            cand_snapshot = [
                {"id": c.id, "text": c.fact_text, "type": c.fact_type,
                 "status": c.event_status, "pool": pool,
                 "occ": (c.t_occurred, c.t_occurred_end or c.t_occurred),
                 "mentioned": c.t_mentioned}
                for c, pool in shortlist]
            # REAL subjects (R2-B4: content-word overlap matched
            # 'Stanford' against 'Stanford Stadium' — string overlap is
            # not a subject): the Stage-3 entity join table, one query
            # for the new fact + all candidates.
            from agentmem_os.db.models import SemanticFactEntity
            all_ids = [fact.id] + [c["id"] for c in cand_snapshot]
            node_map = {}
            for fid, nid in (session.query(SemanticFactEntity.fact_id,
                                           SemanticFactEntity.node_id)
                             .filter(SemanticFactEntity.fact_id
                                     .in_(all_ids)).all()):
                node_map.setdefault(fid, set()).add(nid)
            new_nodes = node_map.get(fact.id, set())
            for c in cand_snapshot:
                c["shared_nodes"] = bool(
                    new_nodes & node_map.get(c["id"], set()))
        finally:
            session.close()

        cosignals = {
            c["id"]: self._cosignal(new_snapshot["text"], c["text"],
                                    c["pool"], c["shared_nodes"])
            for c in cand_snapshot
        }
        prompt = _PROMPT.format(
            new_fact=new_snapshot["text"],
            candidates="\n".join(
                f"[{c['id']}] ({c['type']}"
                + (f"/{c['status']}" if c["status"] else "")
                + f") {c['text']}"
                for c in cand_snapshot))
        raw = self._llm(prompt)

        return self._apply(fact_id, session_id, new_snapshot,
                           cand_snapshot, cosignals, raw)

    def _apply(self, fact_id, session_id, new_snapshot, cand_snapshot,
               cosignals, raw) -> dict:
        """Deterministic gates over the model's proposals, then ONE
        transaction applying every surviving action TOGETHER WITH the
        audit row (G3 S4-R1 B3: actions and audit committed separately
        meant a crash window where a supersession existed with no
        judgment row AND the fact was unsweepable — applied-but-
        unaudited is a lie the record must be unable to tell). A store
        refusal (race) rolls the transaction back, moves that action to
        dropped, and retries without it — bounded by the action count,
        every retry audited."""
        from agentmem_os.db.models import SemanticFact
        from agentmem_os.db.semantic_facts import SemanticFactStore

        by_id = {c["id"]: c for c in cand_snapshot}
        dropped = []
        super_plan, cancel_plan = [], []
        new_id = new_snapshot["id"]
        if not isinstance(raw, dict):
            dropped.append((str(raw)[:40], "model output not an object"))
            raw = {}

        for cid in _int_list(raw.get("superseded_ids"), dropped,
                             "superseded_ids"):
            if cid not in by_id:
                dropped.append((cid, "not in shortlist"))
                continue
            cand = by_id[cid]
            # B1: ONLY same-type facts may supersede each other. Planned
            # events sit in the shortlist STRICTLY as cancellation
            # candidates — without this guard the reserved planned pool
            # (a G1 fix) handed the model a future-dated event that,
            # being domain-later, ALWAYS won: a planned event
            # invalidated a true current state (the dangerous class,
            # reproduced by R1).
            if cand["type"] != new_snapshot["type"]:
                dropped.append((cid, f"type mismatch ({cand['type']} vs "
                                     f"{new_snapshot['type']}) — only "
                                     "same-type facts supersede; events "
                                     "are cancellation candidates only"))
                continue
            if not cosignals[cid]["agrees"]:
                dropped.append((cid, "co-signal disagreement (no subject-"
                                     "guarded polarity flip, no metric-"
                                     "update signal, and similarity may "
                                     "not approve what similarity "
                                     "admitted) — LLM-only verdicts "
                                     "never act"))
                continue
            direction = _domain_direction(new_snapshot, cand)
            if direction is None:
                dropped.append((cid, "domain-time tie/overlap — direction "
                                     "ambiguous, conservative skip"))
                continue
            old_id, winner_id, t_inv = direction
            super_plan.append((cid, old_id, winner_id, t_inv))

        for cid in _int_list(raw.get("cancelled_ids"), dropped,
                             "cancelled_ids"):
            if cid not in by_id:
                dropped.append((cid, "not in shortlist"))
                continue
            cand = by_id[cid]
            if cand["type"] != "event" or cand["status"] != "planned":
                dropped.append((cid, f"cancellation only applies to "
                                     f"planned events; candidate is "
                                     f"{cand['type']}/{cand['status']}"))
                continue
            # R3-B1/R4-B1: the binding gate is INDEPENDENT of how
            # the plan entered the shortlist — per non-negated cue
            # occurrence, position-accurate clause, CONTAINMENT (the
            # clause's object words must be non-empty and a subset of
            # the plan's words). Earlier gates re-ran the pool's own
            # admission test or counted generic shared words; both
            # could not reject what mattered.
            binds, why = _cancellation_binds(new_snapshot["text"],
                                             cand["text"])
            if not binds:
                dropped.append((cid, f"cancellation gate: {why} — "
                                     "LLM-only cancellations never act"))
                continue
            cancel_plan.append(cid)

        # ONE transaction: all surviving actions + the audit row commit
        # together. Store refusals (races) drop the refused action and
        # retry the remainder — bounded, every exclusion recorded.
        store = SemanticFactStore(self.get_db)
        applied_super, applied_cancel = [], []
        while True:
            txn = self.get_db()
            try:
                failed = None
                for entry in super_plan:
                    cid, old_id, winner_id, t_inv = entry
                    try:
                        store.supersede(old_id, winner_id,
                                        t_invalid=t_inv, db=txn)
                    except ValueError as e:
                        failed = (entry, None, f"store refused: {e}")
                        break
                if failed is None:
                    for cid in cancel_plan:
                        try:
                            store.mark_event_cancelled(cid, db=txn)
                        except ValueError as e:
                            failed = (None, cid, f"store refused: {e}")
                            break
                if failed is not None:
                    txn.rollback()
                    s_entry, c_entry, reason = failed
                    if s_entry is not None:
                        super_plan.remove(s_entry)
                        dropped.append((s_entry[0], reason))
                    else:
                        cancel_plan.remove(c_entry)
                        dropped.append((c_entry, reason))
                    continue
                applied_super = [(o, w) for _, o, w, _ in super_plan]
                applied_cancel = list(cancel_plan)
                self._log(txn, fact_id, session_id,
                          [{"id": c["id"], "pool": c["pool"],
                            "occ": c["occ"], "mentioned": c["mentioned"]}
                           for c in cand_snapshot],
                          cosignals, raw,
                          {"superseded": applied_super,
                           "cancelled": applied_cancel},
                          dropped, None,
                          new_domain={"occ": new_snapshot["occ"],
                                      "mentioned": new_snapshot["mentioned"]})
                txn.commit()
                break
            finally:
                txn.close()
        return {"superseded": applied_super, "cancelled": applied_cancel,
                "dropped": dropped, "skipped": None}

    # ── Shortlist + co-signal ───────────────────────────────────────────────

    def _shortlist(self, session, fact) -> list:
        """Live facts, same scope, from THREE reserved pools (each cap
        disclosed in the audit row):

          1. entity peers — same fact_type, sharing >=1 entity node
             (Stage 3's join table), cap _SHORTLIST_CAP;
          2. lexical peers — same fact_type, TF-IDF cosine >=
             _COSIGNAL_COSINE against the newest _LEXICAL_SCAN_CAP live
             facts (G2 finding, fixed at gate time: personal-metric
             facts — "personal best 5K time", the ARCHETYPAL
             knowledge-update class — carry NO named entity, so an
             entity-only shortlist starved and the judge never fired on
             the real corpus case. Graphiti's candidate search is TEXT
             search; entity-only was our shortcut, now corrected. Facts
             older than the scan window are reachable through the
             entity pool only — disclosed bound);
          3. planned events (cancellation candidates), cap
             _PLANNED_SLOT_CAP (G1 finding: a single pool let recent
             peers crowd these out).
        """
        from agentmem_os.db.models import SemanticFact, SemanticFactEntity

        node_ids = [r[0] for r in session.query(SemanticFactEntity.node_id)
                    .filter(SemanticFactEntity.fact_id == fact.id).all()]
        peer_fact_ids = set()
        if node_ids:
            peer_fact_ids = {r[0] for r in (
                session.query(SemanticFactEntity.fact_id)
                .filter(SemanticFactEntity.node_id.in_(node_ids))
                .distinct().all())} - {fact.id}

        def _pool(id_filter, type_filter, cap):
            q = (
                session.query(SemanticFact)
                .filter(SemanticFact.superseded_by.is_(None),
                        SemanticFact.scope_key == fact.scope_key,
                        SemanticFact.id != fact.id)
                .filter(type_filter)
                .order_by(SemanticFact.t_mentioned.desc(),
                          SemanticFact.id.desc())
            )
            if id_filter is not None:
                q = q.filter(SemanticFact.id.in_(id_filter))
            return q.limit(cap).all()

        from agentmem_os.memory.conflict_detector import _tfidf_cosine

        peers = (_pool(peer_fact_ids,
                       SemanticFact.fact_type == fact.fact_type,
                       _SHORTLIST_CAP)
                 if peer_fact_ids else [])
        seen = {f.id for f in peers}
        out = [(f, "entity") for f in peers]

        scan = _pool(None, SemanticFact.fact_type == fact.fact_type,
                     _LEXICAL_SCAN_CAP)
        for cand in scan:
            if len(seen) >= _SHORTLIST_CAP:
                break
            if cand.id in seen:
                continue
            if _tfidf_cosine(fact.fact_text, cand.fact_text) \
                    >= _COSIGNAL_COSINE:
                out.append((cand, "lexical"))
                seen.add(cand.id)

        # Planned pool: entity-shared when the fact has links; for
        # entity-less facts the pool is TOPICALLY filtered (R2-B2: the
        # unrestricted fallback showed every entity-less fact 4
        # arbitrary live plans — 'oat milk' saw the Tokyo flight).
        if peer_fact_ids:
            planned = _pool(peer_fact_ids,
                            (SemanticFact.fact_type == "event")
                            & (SemanticFact.event_status == "planned"),
                            _PLANNED_SLOT_CAP)
        else:
            from agentmem_os.memory.conflict_detector import (
                _content_word_overlap)
            planned = [
                c for c in _pool(None,
                                 (SemanticFact.fact_type == "event")
                                 & (SemanticFact.event_status == "planned"),
                                 _LEXICAL_SCAN_CAP)
                if _content_word_overlap(fact.fact_text, c.fact_text)
            ][:_PLANNED_SLOT_CAP]
        out += [(p, "planned") for p in planned if p.id not in seen]
        return out

    @staticmethod
    def _cosignal(new_text: str, cand_text: str, pool: str,
                  shared_nodes: bool) -> dict:
        """Deterministic agreement signal, per candidate. INDEPENDENCE
        RULE (G3 S4-R1 B2: the lexical pool admitted on cosine>=0.25 and
        the gate then agreed on the SAME cosine — a vacuous gate that
        let 'allergic to peanuts' be invalidated by 'allergic to
        shellfish', the exact Mem0 #1674 class): a candidate admitted BY
        similarity can never be approved BY that same similarity.

          - polarity flip counts only with a SHARED ENTITY NODE
            (Stage 3's join table — R2-B4: string overlap matched
            'Stanford' against 'Stanford Stadium');
          - metric-update signal: digit-aware _metric_update — masked
            texts exactly identical, token counts equal, >=2 numbers,
            EXACTLY ONE normalized value differs (no cosine anywhere
            in it — R2-B1 killed the cosine version);
          - bare cosine >= _COSIGNAL_COSINE is acceptable ONLY for
            entity-pool candidates, where admission came from the join
            table, not from text similarity.
        """
        from agentmem_os.memory.conflict_detector import (
            _has_polarity_flip, _tfidf_cosine)

        flip, category = _has_polarity_flip(cand_text, new_text)
        # Subject guard = a SHARED ENTITY NODE (R2-B4: content-word
        # overlap matched 'Stanford'/'Stanford Stadium' and
        # 'Berlin'/'Berlin Hauptbahnhof' — different NER nodes, so the
        # join table refuses what string overlap accepted). Entity-less
        # facts cannot flip; their only route is the metric signal,
        # whose masked-identity requirement is stricter than any
        # vocabulary. Disclosed cost: a true flip whose two phrasings
        # link different nodes is missed — the chosen mild class.
        flip = bool(flip and shared_nodes)
        cosine = round(_tfidf_cosine(new_text, cand_text), 4)
        metric_update = _metric_update(new_text, cand_text)

        agrees = (flip or metric_update
                  or (pool == "entity" and cosine >= _COSIGNAL_COSINE))
        return {"flip": flip, "category": category, "cosine": cosine,
                "metric_update": metric_update,
                "masked_equal": _mask_numbers(new_text)
                == _mask_numbers(cand_text),
                "shared_nodes": shared_nodes, "pool": pool,
                "agrees": agrees}

    # ── Audit row ───────────────────────────────────────────────────────────

    def _log(self, session, fact_id, session_id, shortlist_entries,
             cosignals, raw, applied, dropped, skipped, new_domain=None):
        """Audit row. shortlist_entries carry per-candidate POOL
        provenance and domain times, and the row records the constants
        that shaped the judgment (Ma4: without these, a judgment could
        not be re-audited against what the gates actually saw)."""
        from agentmem_os.db.models import SupersessionJudgment

        session.add(SupersessionJudgment(
            fact_id=fact_id, session_id=session_id, model=self.model,
            shortlist_json=json.dumps(
                {"candidates": shortlist_entries,
                 "new_domain": new_domain,
                 "peer_cap": _SHORTLIST_CAP,
                 "planned_cap": _PLANNED_SLOT_CAP,
                 "lexical_scan_cap": _LEXICAL_SCAN_CAP,
                 "cosignal_cosine": _COSIGNAL_COSINE}),
            cosignal_json=json.dumps(cosignals) if cosignals else None,
            raw_output_json=json.dumps(raw) if raw else None,
            applied_json=json.dumps(applied) if applied else None,
            dropped_json=json.dumps(dropped) if dropped else None,
            skipped=skipped,
        ))

    # ── Recovery sweep (the link_missing pattern) ───────────────────────────

    def judge_missing(self, agent_id: str = None, user_id: str = None,
                      limit: int = 200, after_id: int = 0) -> dict:
        """
        Judge live state/preference facts that have NO judgment row —
        crash recovery (facts committed, judge never ran) and backfill.
        Cursor-paged; each fact judged independently (a bad fact cannot
        take the sweep down); failures recorded, not raised. Drain:
        loop with after_id until candidates == 0 (once per drain).
        """
        from sqlalchemy import exists
        from agentmem_os.db.models import SemanticFact, SupersessionJudgment
        from agentmem_os.db.semantic_facts import make_scope_key

        scope_key = make_scope_key(agent_id, user_id)
        session = self.get_db()
        try:
            unjudged = (
                session.query(SemanticFact.id)
                .filter(SemanticFact.scope_key == scope_key,
                        SemanticFact.id > after_id,
                        SemanticFact.superseded_by.is_(None),
                        SemanticFact.fact_type.in_(("state", "preference")))
                .filter(~exists().where(
                    SupersessionJudgment.fact_id == SemanticFact.id))
                .order_by(SemanticFact.id.asc())
                .limit(limit).all()
            )
        finally:
            session.close()

        judged = supersessions = 0
        failures = []
        for (fid,) in unjudged:
            try:
                r = self.judge_fact(fid)
                judged += 1
                supersessions += len(r["superseded"])
            except Exception as e:
                failures.append((fid, str(e)))
                logger.warning(f"[SupersessionJudge] sweep failed on "
                               f"fact {fid}: {e}")
        return {"candidates": len(unjudged), "judged": judged,
                "supersessions": supersessions, "failures": failures,
                "next_after_id": (unjudged[-1][0] if unjudged
                                  else after_id)}


def _int_list(value, dropped, field) -> list:
    """Model output hygiene: non-list or non-int content is dropped
    LOUDLY into the audit trail, never coerced; duplicate ids collapse
    to one (R2 minor: a doubled id used to be audited as a phantom
    concurrency race)."""
    if not isinstance(value, list):
        dropped.append((str(value)[:40], f"{field} not a list"))
        return []
    out, seen = [], set()
    for v in value:
        if isinstance(v, bool) or not isinstance(v, int):
            dropped.append((str(v)[:40], f"non-integer id in {field}"))
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out

