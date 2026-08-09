"""
AgentMem OS — Consolidation Engine v2 (distillation, not compression)
=====================================================================
Stage 2 of CONSOLIDATION_V2_DESIGN.md: at session end, distill raw turns
(episodic) into dated atomic SemanticFacts (semantic) via a LOCAL model.
Episodes are KEPT — this engine adds knowledge, never deletes evidence.

What exists in THIS file (G3 R1+R2 corrected; every line implemented):
  1. EXTRACT  — schema-constrained local-LLM call (Ollama `format`
                grammar), zero-shot prompt; the server's actual
                prompt_eval_count is read back and a context-window clamp
                is REPORTED, never silent.
  2. VALIDATE — calendar-real dates; future-dated events stored as dated
                events with event_status='planned' (F7, founder-resolved
                2026-08-06 — the marker replaced the "pending F7" warning;
                retyping corrupted dedup identity, R3-B3, and stays
                banned); unparseable dates drop the date and keep the
                fact; vague-quantifier-vs-cited-digits WARNS, never
                rejects.
  3. SUPPORT GATE — a fact is accepted only with USER-turn evidence:
                stemmed-token overlap; ONE numeric-mention parser on
                both sides (digits incl. glued units, decimals, word
                numerals to twelve — ABOVE TWELVE UNCHECKED, disclosed)
                with complete-date-expression exemption and per-turn
                licensing recorded in the report (R2-B1..R5-B3).
                Citations rank ALL roles by evidence strength before the
                cap, and the cap is disclosed per fact.
  4. WRITE    — one atomic caller-owned batch (extraction AND entity NER
                before the transaction); ConsolidationLog rows persist
                truncation, rejection, and entity-link counts.
  5. LINK     — Stage 3: accepted facts link to KG nodes through
                FactEntityLinker inside the same batch (write-lock-held
                check-then-insert; a linking failure never takes the
                facts down — compute-level failures leave the batch
                intact and link_missing() recovers unlinked facts
                (skipped SURFACES need rescan_all=True — R3 B1); a
                DB-level failure aborts the batch LOUDLY at the next
                flush or commit, the Stage-1 contract).

NOT built yet (disclosed, tracked in the build log): Tier 2-3 semantic
dedup (embedding shortlist + batched LLM adjudication); per-event count
fields ("three times" as count=3); deterministic relative-date resolution
(the model resolves ~50% of relative dates wrong — measured, disclosed —
a deterministic resolver is queued). Undated facts store t_occurred=NULL
by FOUNDER DECISION (2026-08-06): DESIGN §5.1's session-date default is
revoked — "when" is only ever a user-stated time.
"""

import json
import os
import re
import urllib.request
from datetime import datetime

from loguru import logger

from agentmem_os.db.semantic_facts import SemanticFactStore, normalize_date

# Env-configurable (open-items ledger #5; load-bearing from the
# cluster run on: SLURM array tasks may share a node, so each
# task needs its own Ollama port). Default unchanged.
OLLAMA_URL = os.environ.get(
    "AGENTMEM_OS_OLLAMA_URL", "http://localhost:11434") \
    .rstrip("/") + "/api/generate"
DEFAULT_MODEL = "llama3.1:latest"
TRANSCRIPT_CAP = 36000
# Output ceiling. 2000 silently truncated fact-dense sessions
# (Gate C: 3/3,631 lost, one of them gold evidence).
NUM_PREDICT = 4000
NUM_PREDICT_MAX = 16000
# Escalation-only remedy for temperature-0 repetition loops.
# 1.05 measured as the sweet spot: 1.15 over-suppressed
# (9 facts -> 2), 1.05 broke the loop and kept all 9.
ANTI_REPEAT_OPTS = {"repeat_penalty": 1.05, "repeat_last_n": 1024}
NUM_CTX = 10240
CITE_CAP = 8
STAMP_SCAN_TURNS = 3   # session-date stamps are headers; scanning every
                       # turn let user content hijack the date (R2-M1)

FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "fact_type": {"type": "string",
                                  "enum": ["event", "state",
                                           "preference", "identity"]},
                    "t_occurred": {"type": ["string", "null"]},
                },
                "required": ["text", "fact_type", "t_occurred"],
            },
        }
    },
    "required": ["facts"],
}

_WORD_NUMS = {"one": "1", "once": "1", "two": "2", "twice": "2",
              "three": "3", "thrice": "3", "four": "4", "five": "5",
              "six": "6", "seven": "7", "eight": "8", "nine": "9",
              "ten": "10", "eleven": "11", "twelve": "12"}
_DIGIT = re.compile(r"\d")
# Token = any run of non-space, non-ASCII-punct chars. Keeps Devanagari
# words INTACT (R5-M2: \w splits at combining marks, collapsing a whole
# Hindi sentence to one token and making the support gate inert).
_TOKEN = re.compile(r"[^\s!-/:-@\[-`{-~]{3,}")
_STOP = frozenset("""the user that this with have been from they them were
about their would could should there where which while during session dated
was has had are you for not but all can get got its his her our out now one
and the very much many more also just like some went goes going
""".split())
_MONTHS_RE = ("january|february|march|april|may|june|july|august|"
              "september|october|november|december")
# Complete date-expression shapes, longest-first so "February 2023" can
# never half-match as day-form and orphan digits (R5-B1: the old pattern
# ate "20" of the year and left "23" as a phantom number that destroyed
# true count-bearing facts on the real sample).
_DATE_EXPR_RE = re.compile(
    rf"\d{{4}}[/-]\d{{1,2}}[/-]\d{{1,2}}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS_RE})(?:,?\s+\d{{4}})?"
    rf"|(?:{_MONTHS_RE})\s+\d{{4}}(?!\d)"
    rf"|(?:{_MONTHS_RE})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?",
    re.IGNORECASE)
# Inline per-line timestamps ("[2023/05/20 (Sat) 14:05]") prefix 100% of
# corpus user lines — their digits are calendar plumbing and must never
# license a tool fact's numbers (R5-B3).
_INLINE_STAMP_RE = re.compile(
    r"\[\d{4}[/-]\d{1,2}[/-]\d{1,2}[^\]]*\]"
    r"|session dated\s*\d{4}[/-]\d{1,2}[/-]\d{1,2}"
    r"(?:\s*\([^)]*\))?(?:\s*\d{1,2}:\d{2})?", re.IGNORECASE)
# Digit runs WITHOUT a trailing boundary requirement: "10s", "16GB",
# "4213ms", "3x" all yield their number (R5-B2: \b after digits made
# formatting decide acceptance).
_MENTION_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD_NUM_RES = {re.compile(rf"\b{w}\b"): v for w, v in _WORD_NUMS.items()}


def _norm_val(raw: str) -> str:
    v = raw.replace(",", "")
    if "." not in v:
        v = v.lstrip("0") or "0"
    return v


def _quantity_values(text: str, strip_stamps: bool = False):
    """One parser for BOTH sides of the numbers gate. Returns
    (values, exempted): every numeric mention (digits incl. glued units,
    decimals, whole-word numerals) normalized (commas, zero-padding);
    mentions whose span lies INSIDE a complete date expression are
    classified date-components and exempted — with the exempted list
    returned for the audit trail (R5-M5)."""
    if strip_stamps:
        text = _INLINE_STAMP_RE.sub(" ", text)
    date_spans = [(m.start(), m.end()) for m in _DATE_EXPR_RE.finditer(text)]
    values, exempted = set(), []
    for m in _MENTION_RE.finditer(text):
        v = _norm_val(m.group(0))
        if any(a <= m.start() and m.end() <= b for a, b in date_spans):
            exempted.append(v)
        else:
            values.add(v)
    low = text.lower()
    for rx, val in _WORD_NUM_RES.items():
        if rx.search(low):
            values.add(val)
    return values, exempted


_STAMP_RE = re.compile(
    r"(session dated\s*)?\d{4}[/-]\d{1,2}[/-]\d{1,2}"
    r"(\s*\([^)]*\))?(\s*\d{1,2}:\d{2})?", re.IGNORECASE)


def _stem(tok: str) -> str:
    """Light suffix stripping so 'microchipping' meets 'microchipped'
    (R2-M6). ASCII-only — suffix surgery on Devanagari would corrupt."""
    if not tok.isascii():
        return tok
    for suf in ("ing", "ed", "es", "s"):
        if len(tok) > 4 and tok.endswith(suf):
            return tok[:-len(suf)]
    return tok


def _tokens(text: str) -> set:
    return {_stem(t) for t in _TOKEN.findall(text.lower())
            if t not in _STOP}


def _event_status(fact_type: str, t_occ: str, session_date: str):
    """F7 SINGLE SOURCE (founder-resolved 2026-08-06): the validator
    warning and the stored marker MUST come from this one comparison —
    two spellings of "is this future-dated?" is the drift class R3-B3
    grew from. Events only; 'planned' when the occurrence START is past
    the session date at mention time (a month interval overlapping the
    session date is not clearly future → 'occurred'); undated events are
    'occurred' per the extractor contract (events already happened).
    Non-events: None.

    REACHABILITY, disclosed plainly (G3 R1 M5): the extraction prompt
    types ALL plans — dated or not — as 'state', so this marker fires
    only when the model emits a future-dated EVENT against instruction.
    Measured on the G2 sample: 0/23 facts carried 'planned'; three plan
    facts were typed state per the prompt. The marker is a correctness
    safety net for that disobedience class, not the primary plan
    representation. Whether dated plans should instead extract as
    planned EVENTS is a prompt-policy change that would alter measured
    Gate-B extraction behavior — queued as a founder decision, not
    slipped in mid-arc."""
    if fact_type != "event":
        return None
    if t_occ and normalize_date(t_occ)[0] > session_date:
        return "planned"
    return "occurred"


class ConsolidationV2:
    def __init__(self, get_db_session, model: str = DEFAULT_MODEL,
                 timeout: int = 600):
        self.get_db = get_db_session
        self.model = model
        self.timeout = timeout
        self.store = SemanticFactStore(get_db_session)
        from agentmem_os.db.fact_entities import FactEntityLinker
        from agentmem_os.llm.supersession import SupersessionJudge
        self.linker = FactEntityLinker(get_db_session)
        self.judge = SupersessionJudge(get_db_session, model=model,
                                       timeout=timeout)
        self._last_prompt_eval = None

    # ── 1. Extraction ───────────────────────────────────────────────────────

    def _llm(self, prompt: str, num_predict: int = NUM_PREDICT,
             anti_repeat: bool = False) -> dict:
        """Extraction call. OUTPUT truncation is detected and retried,
        never swallowed (Gate C finding, 2026-08-09): a fact-dense
        session that exceeded num_predict returned JSON cut mid-string,
        the JSONDecodeError killed the WHOLE session, and 3 of 3,631
        sessions vanished — one of them GOLD EVIDENCE for an
        aggregation question, i.e. a silently depressed benchmark
        score. Ollama reports done_reason='length' when it hits the
        cap, so the cap is now observable: retry once at double, then
        fail with a message that names the real cause instead of a
        parser error."""
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps({
                "model": self.model, "prompt": prompt, "stream": False,
                "format": FACTS_SCHEMA,
                "options": {"temperature": 0, "num_ctx": NUM_CTX,
                            "num_predict": num_predict,
                            **(ANTI_REPEAT_OPTS if anti_repeat else {})},
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.loads(r.read())
        if "response" not in body:
            raise ValueError(f"Ollama reply missing 'response': {body}")
        self._last_prompt_eval = body.get("prompt_eval_count")
        if body.get("done_reason") == "length":
            if num_predict >= NUM_PREDICT_MAX:
                raise ValueError(
                    f"extraction output hit the num_predict ceiling "
                    f"({num_predict}) even after retry — the session is "
                    f"too fact-dense for this cap; raise NUM_PREDICT_MAX "
                    f"rather than losing the session")
            if not anti_repeat:
                # Root cause first (measured 2026-08-09): truncation
                # here is DEGENERATION, not density — llama3.1 at
                # temperature 0 loops on near-duplicate facts
                # ("wants to fine-tune VGG16" / "interested in
                # exploring VGG16" / "wants to fine-t..."), so raising
                # the ceiling only feeds the loop. A mild repetition
                # penalty over a long lookback breaks it: the two
                # degenerate sessions of 3,631 went from unbounded to
                # 9 facts each. The penalty is ESCALATION-ONLY —
                # applying it globally would change every extraction
                # (measured: penalty 1.15 cut a session from 9 facts
                # to 2) and invalidate already-extracted corpora.
                logger.warning(
                    f"[ConsolidationV2] output truncated at "
                    f"num_predict={num_predict} — retrying with "
                    f"anti-repetition {ANTI_REPEAT_OPTS}")
                return self._llm(prompt, num_predict=num_predict,
                                 anti_repeat=True)
            logger.warning(
                f"[ConsolidationV2] still truncated with anti-repetition "
                f"at num_predict={num_predict} — retrying at "
                f"{num_predict * 2}")
            return self._llm(prompt,
                             num_predict=min(num_predict * 2,
                                             NUM_PREDICT_MAX),
                             anti_repeat=True)
        return json.loads(body["response"])

    # ── 2/3. Validation + support gate ──────────────────────────────────────

    @staticmethod
    def evaluate_fact(fact: dict, session_date: str, turn_data: list):
        """
        Returns (problems, fact_type, t_occurred, cited, cite_note,
        warns, number_audit):
        problems non-empty => REJECT. Recoverable issues (bad date format,
        future-dated event, vague quantifier) adjust or warn — they never
        destroy an otherwise-supported fact (R2-M2/M5).
        """
        problems, warns = [], []
        text = (fact.get("text") or "").strip()
        fact_type = fact.get("fact_type")
        if len(text) < 8:
            problems.append("text too short to be a self-contained fact")
        if fact_type not in ("event", "state", "preference", "identity"):
            problems.append(f"bad fact_type {fact_type!r}")

        t_occ = None
        raw_date = fact.get("t_occurred")
        if raw_date is not None and not isinstance(raw_date, str):
            warns.append(f"non-string t_occurred {raw_date!r} dropped")
            raw_date = None
        if raw_date:
            try:
                normalize_date(raw_date)
                t_occ = raw_date
            except ValueError:
                warns.append(f"unparseable t_occurred {raw_date!r} dropped")
        # F7 (founder-resolved 2026-08-06): a future-dated EVENT is a plan
        # — stored as a dated event with event_status='planned'. History:
        # R1 stored it as occurred; R2's fix deleted it; R3's retype
        # corrupted dedup identity (R3-B3: the text-only state hash merged
        # different plan dates — retyping stays banned). The warning and
        # the stored marker share ONE detector, _event_status().
        if _event_status(fact_type, t_occ, session_date) == "planned":
            warns.append("future-dated event — stored event_status='planned' (F7)")

        # Support gate + citations (R2-B1/B2): score EVERY turn by stemmed
        # overlap; rank citations by strength, not turn id; require USER
        # evidence, and every NUMBER in the fact must appear in user
        # evidence — tool output's numbers can't ride in on a shared word.
        ftoks = _tokens(text)
        need = 1 if len(ftoks) <= 4 else 2
        scored = []
        for tid, role, content, _ in turn_data:
            overlap = len(ftoks & _tokens(content))
            if overlap >= need:
                scored.append((overlap, tid, role, content))
        scored.sort(key=lambda s: (-s[0], s[1]))
        cited = [tid for _, tid, _, _ in scored[:CITE_CAP]]
        cite_note = (f"{len(cited)} of {len(scored)} supporting turns"
                     if len(scored) > CITE_CAP else None)

        user_content = " ".join(c for _, _, role, c in scored
                                if role == "user")
        user_support = any(role == "user" for _, _, role, _ in scored)

        # Numbers gate — ONE mention parser with span provenance on both
        # sides (R2..R5 all blocked here on surface-form spellings; see
        # _quantity_values). Order matters: with NO user evidence at all,
        # the true cause is "no user support" — never "numbers" (R4-M3,
        # R5-M4).
        number_audit = None
        user_turns_scored = [(tid, c) for _, tid, role, c in scored
                             if role == "user"]
        if not user_support:
            problems.append(
                "no supporting USER turn — assistant/system knowledge "
                "is not a user fact (Mem0 #4573 class)")
        else:
            fact_vals, fact_exempt = _quantity_values(text)
            if fact_vals:
                licensed = {}
                for tid, content in user_turns_scored:
                    tvals, _ = _quantity_values(content, strip_stamps=True)
                    for v in tvals & fact_vals:
                        licensed.setdefault(v, []).append(tid)
                # word numerals in pooled user speech license too
                pooled_low = _INLINE_STAMP_RE.sub(
                    " ", user_content).lower()
                for rx, val in _WORD_NUM_RES.items():
                    if val in fact_vals and rx.search(pooled_low):
                        licensed.setdefault(val, [])
                missing = sorted(fact_vals - set(licensed))
                number_audit = {"claimed": sorted(fact_vals),
                                "date_exempt": fact_exempt,
                                "licensed_by": licensed}
                if missing:
                    user_support = False
                    problems.append(
                        f"numbers {missing} not found in user-stated "
                        "content")

        # Dropped-count guard: WARNS only (R2-M5 — a price or a year in
        # the same turn is not proof the user stated a count).
        stripped_user = _STAMP_RE.sub(" ", user_content)
        if (_DIGIT.search(stripped_user)
                and re.search(r"\b(several|many|some|a few|multiple|"
                              r"numerous|a couple of) "
                              r"(times|rides|occasions|days|weeks|months)\b",
                              text.lower())):
            warns.append("vague quantifier while cited user content has "
                         "numbers — possible dropped count")

        return (problems, fact_type, t_occ, cited, cite_note, warns, number_audit)

    # ── Orchestration ───────────────────────────────────────────────────────

    def consolidate_session(self, session_id: str, agent_id: str = None,
                            user_id: str = None,
                            lang_source: str = "en") -> dict:
        from agentmem_os.db.models import ConsolidationLog, Turn

        started = datetime.utcnow()
        db = self.get_db()
        try:
            turns = (db.query(Turn)
                     .filter(Turn.session_id == session_id)
                     .order_by(Turn.id.asc()).all())
            turn_data = [(t.id, t.role, t.content, t.created_at)
                         for t in turns]
        finally:
            db.close()
        if not turn_data:
            return {"session_id": session_id, "skipped": "no turns"}

        session_date, date_note = self._session_date(turn_data)
        transcript = "\n".join(
            f"[turn {tid}] {role.upper()}: {content}"
            for tid, role, content, _ in turn_data)
        truncated_chars = max(0, len(transcript) - TRANSCRIPT_CAP)
        if truncated_chars:
            logger.warning(f"[ConsolidationV2] {session_id}: transcript "
                           f"truncated by {truncated_chars} chars")
            transcript = transcript[:TRANSCRIPT_CAP]

        self._last_prompt_eval = None
        raw = self._llm(self._prompt(session_date, transcript))
        candidates = raw.get("facts", [])
        prompt_tokens = self._last_prompt_eval
        ctx_clamped = bool(prompt_tokens and prompt_tokens >= NUM_CTX)
        if ctx_clamped:
            # R2-M3: the char cap does not bound the token window — dense
            # scripts (Devanagari) clamp server-side. Never silent.
            logger.warning(f"[ConsolidationV2] {session_id}: prompt hit "
                           f"num_ctx={NUM_CTX} — model saw a clipped "
                           "transcript")

        accepted, rejected, warnings, numbers_audit = [], [], [], []
        for f in candidates:
            (problems, ftype, t_occ, cited, cite_note, warns,
             n_audit) = self.evaluate_fact(f, session_date, turn_data)
            if n_audit:
                numbers_audit.append((str(f.get("text"))[:60], n_audit))
            for w in warns:
                warnings.append((str(f.get("text"))[:60], w))
            if cite_note and not problems:
                warnings.append((str(f.get("text"))[:60], cite_note))
            if problems:
                rejected.append((f, problems))
            else:
                # Stage 3: NER AND alias planning run BEFORE the batch
                # transaction — same rule as extraction: no model
                # compute while any write lock is held that this loop
                # controls (G3 R1 B1: in-batch planning loaded the
                # ~87s-cold alias model (Stage 6: mostly NETWORK,
                # ~6s offline-first) under the DB-wide write lock
                # and killed a competing writer). plan_surfaces uses
                # its own short READ session here. Known consequence,
                # disclosed: an anchor node created later in THIS batch
                # is invisible to plans made now — such surfaces land
                # in skipped, recoverable ONLY by the
                # link_missing(rescan_all=True) deep sweep — the
                # default zero-link sweep cannot see a partially
                # linked fact (R3 B1).
                surfaces = self.linker.extract_surfaces(f["text"])
                plan = self.linker.plan_surfaces(surfaces, agent_id=agent_id)
                accepted.append((f, ftype, t_occ, cited, surfaces, plan))

        created = reaffirmed = entities_linked = 0
        link_skipped, link_failure = [], None
        judge_queue = []
        batch = self.get_db()
        try:
            for f, ftype, t_occ, cited, surfaces, plan in accepted:
                fact, was_created = self.store.add_fact(
                    f["text"], fact_type=ftype,
                    t_mentioned=session_date, t_occurred=t_occ,
                    source_session_id=session_id, source_turn_ids=cited,
                    extraction_model=self.model, lang_source=lang_source,
                    agent_id=agent_id, user_id=user_id,
                    event_status=_event_status(ftype, t_occ, session_date),
                    entities=[s for s, _ in surfaces],   # display cache;
                                                         # the join table
                                                         # is the query path
                    db=batch,
                )
                created += was_created
                reaffirmed += (not was_created)
                if was_created and ftype in ("state", "preference"):
                    # Stage 4: judged post-commit. CREATED facts only —
                    # a re-affirmed fact was judged at creation
                    # (disclosed: links added later can enable new
                    # candidates; judge_missing has no re-judge path by
                    # design, operators re-judge via judge_fact).
                    judge_queue.append(fact.id)
                # Stage 3 linking, same batch (write lock already held by
                # the add_fact flush/UPDATE above), APPLY-ONLY — the plan
                # was computed before the transaction (B1). Failure
                # policy of record: a linking exception never takes the
                # FACTS down — after the first failure the rest of the
                # batch skips linking, and the failure is PERSISTED in
                # the log row (R1-M2: the count alone could not
                # distinguish "suspended" from "nothing to link");
                # link_missing() recovers later. If the failure poisoned
                # the session at the DB layer, the NEXT flush or the
                # commit raises LOUDLY (the next add_fact's flush hits
                # it first when more facts follow) — the Stage-1 abort
                # contract, callers retry.
                if link_failure is None:
                    try:
                        rep = self.linker.link_fact(
                            fact.id, plan=plan, agent_id=agent_id,
                            session_id=session_id, db=batch)
                        entities_linked += len(rep["linked"])
                        link_skipped.extend(rep["skipped"])
                    except Exception as e:
                        link_failure = f"{type(e).__name__}: {e}"
                        logger.warning(
                            f"[ConsolidationV2] {session_id}: entity "
                            f"linking failed on fact {fact.id} — linking "
                            f"suspended for this batch: {link_failure}")
            batch.add(ConsolidationLog(
                session_id=session_id, turns_processed=len(turn_data),
                summaries_generated=created,
                truncated_chars=truncated_chars,      # persisted (R2-M4)
                rejected_count=len(rejected),          # persisted (R1-M7)
                rejections_json=json.dumps(
                    [(str(f.get("text"))[:120], p) for f, p in rejected]),
                entities_linked=entities_linked,       # NEW link rows only
                link_failure=link_failure,             # persisted (R1-M2)
                duration_seconds=(datetime.utcnow() - started).total_seconds(),
                triggered_by="consolidation_v2",
            ))
            batch.commit()
        except Exception:
            batch.rollback()
            raise
        finally:
            batch.close()

        # R4-M5: the recovery sweep must have a PRODUCT caller, not
        # live only in docstrings ("recovery today is a human writing
        # the drain loop in a REPL"). After a suspended-linking commit,
        # run the bounded default-depth drain for this scope — every
        # fact whose linking was suspended has zero link rows and is
        # exactly what the default sweep covers. (The suspended-run
        # FIRST fact may hold partial links if it failed mid-apply;
        # that residue needs the deep sweep — disclosed in the report.)
        # BEST-EFFORT (R5): the facts and log row are already
        # committed — a recovery-side fault must never destroy the
        # report; it is recorded, not raised. Recovered links are
        # written back into the persisted log row (R5: 12 links in the
        # DB against a persisted 0 is an incoherent audit).
        link_recovery = None
        if link_failure is not None:
            try:
                link_recovery = self.recover_links(agent_id=agent_id,
                                                   user_id=user_id)
                if link_recovery["links_created"]:
                    patch = self.get_db()
                    try:
                        row = (patch.query(ConsolidationLog)
                               .filter(ConsolidationLog.session_id
                                       == session_id)
                               .order_by(ConsolidationLog.id.desc())
                               .first())
                        row.entities_linked = (
                            (row.entities_linked or 0)
                            + link_recovery["links_created"])
                        row.link_failure = (
                            f"{link_failure} | recovered "
                            f"{link_recovery['links_created']} links "
                            f"post-commit")
                        patch.commit()
                    finally:
                        patch.close()
            except Exception as e:
                link_recovery = {"error": f"{type(e).__name__}: {e}"}
                logger.warning(f"[ConsolidationV2] {session_id}: link "
                               f"recovery itself failed: {e}")

        # Stage 4: supersession judgment — AFTER the batch commits (LLM
        # compute never under a write lock, the R1-B1 law), BEST-EFFORT
        # (facts and log are already durable; a judge fault is recorded
        # in report + log row, never raised), per-fact isolation (one
        # bad judgment cannot take the rest down).
        supersession = None
        judge_failure = None
        if judge_queue:
            supersession = {"judged": 0, "superseded": [], "cancelled": [],
                            "dropped": 0, "failures": []}
            for fid in judge_queue:
                try:
                    jr = self.judge.judge_fact(fid, session_id=session_id)
                    supersession["judged"] += 1
                    supersession["superseded"].extend(jr["superseded"])
                    supersession["cancelled"].extend(jr["cancelled"])
                    supersession["dropped"] += len(jr["dropped"])
                except Exception as e:
                    supersession["failures"].append((fid, str(e)))
                    logger.warning(f"[ConsolidationV2] {session_id}: "
                                   f"judgment failed on fact {fid}: {e}")
            if supersession["failures"]:
                judge_failure = (f"{len(supersession['failures'])} of "
                                 f"{len(judge_queue)} judgments failed: "
                                 f"{supersession['failures'][0][1][:120]}")
                try:
                    patch = self.get_db()
                    try:
                        row = (patch.query(ConsolidationLog)
                               .filter(ConsolidationLog.session_id
                                       == session_id)
                               .order_by(ConsolidationLog.id.desc())
                               .first())
                        row.judge_failure = judge_failure
                        patch.commit()
                    finally:
                        patch.close()
                except Exception as e:
                    logger.warning(f"[ConsolidationV2] {session_id}: could "
                                   f"not persist judge_failure: {e}")

        report = {
            "session_id": session_id, "turns": len(turn_data),
            "candidates": len(candidates), "created": created,
            "reaffirmed": reaffirmed, "rejected": len(rejected),
            "rejections": [(str(f.get("text"))[:80], p)
                           for f, p in rejected],
            "warnings": warnings,
            "truncated_chars": truncated_chars,
            "prompt_tokens": prompt_tokens,
            "ctx_clamped": ctx_clamped,
            "session_date_note": date_note,
            "entities_linked": entities_linked,     # Stage 3
            "entity_links_skipped": link_skipped,   # (surface, reason)
            "link_failure": link_failure,           # None = clean run
            "link_recovery": link_recovery,         # auto-drain (R4-M5)
            "supersession": supersession,           # Stage 4 judgments
            "judge_failure": judge_failure,         # None = clean
            "numbers_audit": numbers_audit,   # which values licensed by
                                              # which turns; exempted date
                                              # digits (R5-M5)
            "model": self.model, "session_date": session_date,
        }
        logger.info(f"[ConsolidationV2] {report}")
        return report

    def recover_links(self, agent_id: str = None, user_id: str = None,
                      deep: bool = False, limit: int = 500,
                      max_rounds: int = 200) -> dict:
        """
        The PRODUCT-side drain loop for the linker's recovery sweep
        (R4-M5 — the compensating control must be callable, not a
        docstring recipe). Cursor-drains link_missing to completion:
        deep=False recovers unlinked facts (auto-invoked after a
        link_failure commit); deep=True re-plans every fact in scope
        (run after anchor-bearing ingests — recovers surfaces skipped
        earlier). max_rounds is a runaway guard, LOUD when hit; the
        numeric bound is max_rounds × limit items per call.
        Cost disclosure (R5): the drain is SCOPE-wide, not batch-scoped
        — under a persistent linker fault every auto-invocation
        re-attempts all still-unlinked facts in the scope, O(sessions ×
        scope) until the fault clears; the bound is max_rounds × limit.
        """
        cursor = rounds = swept = links = 0
        failures = []
        while rounds < max_rounds:
            r = self.linker.link_missing(agent_id=agent_id, user_id=user_id,
                                         limit=limit, after_id=cursor,
                                         rescan_all=deep)
            if r["candidates"] == 0:
                break
            cursor = r["next_after_id"]
            swept += r["swept"]
            links += r["links_created"]
            failures.extend(r["failures"])
            rounds += 1
        else:
            logger.warning(f"[ConsolidationV2] recover_links hit "
                           f"max_rounds={max_rounds} — drain incomplete")
        result = {"rounds": rounds, "swept": swept, "links_created": links,
                  "failures": failures, "deep": deep,
                  "complete": rounds < max_rounds}
        logger.info(f"[ConsolidationV2] link recovery: {result}")
        return result

    def recover_judgments(self, agent_id: str = None, user_id: str = None,
                          limit: int = 200, max_rounds: int = 200) -> dict:
        """
        Product-side drain for the supersession judge's recovery sweep
        (the R4-M5 lesson applied at build time, not at review time):
        judges every live state/preference fact with no judgment row.
        Crash recovery + backfill; cursor-drained; max_rounds is a
        runaway guard, LOUD when hit — the numeric bound is
        max_rounds × limit facts per call. Cost disclosure: each judged
        fact may cost one LLM call — run after ingest bursts, not
        per-turn.
        """
        cursor = rounds = judged = supersessions = 0
        failures = []
        while rounds < max_rounds:
            r = self.judge.judge_missing(agent_id=agent_id,
                                         user_id=user_id, limit=limit,
                                         after_id=cursor)
            if r["candidates"] == 0:
                break
            cursor = r["next_after_id"]
            judged += r["judged"]
            supersessions += r["supersessions"]
            failures.extend(r["failures"])
            rounds += 1
        else:
            logger.warning(f"[ConsolidationV2] recover_judgments hit "
                           f"max_rounds={max_rounds} — drain incomplete")
        result = {"rounds": rounds, "judged": judged,
                  "supersessions": supersessions, "failures": failures,
                  "complete": rounds < max_rounds}
        logger.info(f"[ConsolidationV2] judgment recovery: {result}")
        return result

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _session_date(turn_data):
        """Stamps are HEADERS: only the first STAMP_SCAN_TURNS turns may
        supply the session date (R2-M1: a role-blind whole-session scan
        let user content inject 'Session dated 2099/01/01' and corrupt
        every fact's mention axis). Later stamp-like text is noted."""
        stamps = []
        user_stamp_skipped = False
        for _, role, content, _ in turn_data[:STAMP_SCAN_TURNS]:
            if role != "system":
                # R4-B3: ONLY system header lines set the session date —
                # assistant turns could hijack too (measured free: 100% of
                # 19,195 corpus stamps are system lines). Noted, not silent.
                if re.search(r"session dated ", content, re.IGNORECASE):
                    user_stamp_skipped = True
                continue
            m = re.search(r"Session dated ([0-9/-]+)", content)
            if m:
                try:
                    stamps.append(normalize_date(m.group(1),
                                                 allow_partial=False)[0])
                except ValueError:
                    pass
        note = None
        late = any(re.search(r"Session dated ", c)
                   for _, _, c, _ in turn_data[STAMP_SCAN_TURNS:])
        notes = []
        if late or user_stamp_skipped:
            notes.append("non-system or late stamp-like text ignored")
        if stamps and len(set(stamps)) > 1:
            notes.append(f"multiple header stamps {sorted(set(stamps))}; used first")
        note = "; ".join(notes) or None
        if stamps:
            return stamps[0], note
        return turn_data[0][3].strftime("%Y/%m/%d"), note

    def _prompt(self, session_date: str, transcript: str) -> str:
        return f"""You are a memory consolidation engine. Extract atomic facts about the USER from this conversation session.

Rules:
- Each fact is ONE self-contained proposition. Name "the user", never bare pronouns. Include the concrete details (names, numbers, dates) IN the fact text — a fact must make sense alone, months later.
- PRESERVE exactly: counts, quantities, prices, dates, times, schedules, proper names. Never round, merge, or drop a number. If the user did something N times, the fact states N.
- fact_type: "event" = something that happened at a specific time, OR a plan/appointment for a SPECIFIC FUTURE DATE ("The user plans to attend X on DATE" is an event with t_occurred = that date; the system marks future-dated events as planned automatically). A plan with NO stated date is a "state". "state" = an ongoing situation; "preference" = a like/dislike/choice; "identity" = who the user is.
- t_occurred: the date the event happened — or, for a planned event, the date it is planned FOR — YYYY/MM/DD (or YYYY/MM if only the month is known) — resolve relative references ("last Tuesday", "two weeks ago") against the session date {session_date}, NEVER against today. null if no date is stated or implied.
- NEVER extract a fact that merely restates what the user ASKED or wants to learn — questions and curiosity are not facts about the user's life.
- ONLY facts stated by the USER about their own life. Never extract the assistant's knowledge, recommendations, availability information, or tool/system output.

Session date: {session_date}
Transcript:
{transcript}"""
