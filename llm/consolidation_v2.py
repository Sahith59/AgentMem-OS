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
  2. VALIDATE — calendar-real dates; future-dated events KEPT as dated
                events with a "pending F7" warning (founder decision open;
                retyping corrupted dedup identity, R3-B3);
                unparseable dates drop the date and keep the fact;
                vague-quantifier-vs-cited-digits WARNS, never rejects.
  3. SUPPORT GATE — a fact is accepted only with USER-turn evidence:
                stemmed-token overlap; ONE numeric-mention parser on
                both sides (digits incl. glued units, decimals, word
                numerals to twelve — ABOVE TWELVE UNCHECKED, disclosed)
                with complete-date-expression exemption and per-turn
                licensing recorded in the report (R2-B1..R5-B3).
                Citations rank ALL roles by evidence strength before the
                cap, and the cap is disclosed per fact.
  4. WRITE    — one atomic caller-owned batch (extraction before the
                transaction); ConsolidationLog rows persist truncation
                and rejection counts (additive columns).

NOT built yet (disclosed, tracked in the build log): Tier 2-3 semantic
dedup (embedding shortlist + batched LLM adjudication); per-event count
fields ("three times" as count=3); deterministic relative-date resolution
(the model resolves ~50% of relative dates wrong — measured, disclosed —
a deterministic resolver is queued). Undated non-event facts store
t_occurred=NULL, diverging from DESIGN §5.1's "else session date" —
escalated to the founder with F7.
"""

import json
import re
import urllib.request
from datetime import datetime

from loguru import logger

from agentmem_os.db.semantic_facts import SemanticFactStore, normalize_date

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:latest"
TRANSCRIPT_CAP = 36000
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


class ConsolidationV2:
    def __init__(self, get_db_session, model: str = DEFAULT_MODEL,
                 timeout: int = 600):
        self.get_db = get_db_session
        self.model = model
        self.timeout = timeout
        self.store = SemanticFactStore(get_db_session)
        self._last_prompt_eval = None

    # ── 1. Extraction ───────────────────────────────────────────────────────

    def _llm(self, prompt: str) -> dict:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps({
                "model": self.model, "prompt": prompt, "stream": False,
                "format": FACTS_SCHEMA,
                "options": {"temperature": 0, "num_ctx": NUM_CTX,
                            "num_predict": 2000},
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.loads(r.read())
            if "response" not in body:
                raise ValueError(f"Ollama reply missing 'response': {body}")
            self._last_prompt_eval = body.get("prompt_eval_count")
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
        # Stage-1 F7 (founder decision open): a future-dated EVENT is a
        # plan. R1 stored it as occurred; R2's fix deleted it; R3's retype
        # corrupted dedup identity. Current policy: KEEP as dated event +
        # warning, pending the founder's call.
        if (fact_type == "event" and t_occ
                and normalize_date(t_occ)[0] > session_date):
            # R3-B3: retyping to state silently merged different plan dates
            # through the text-only state hash (Stage-1 F3 revert). Kept as
            # a dated event + warning; storage policy is FOUNDER DECISION F7.
            warns.append("future-dated event (planned?) — kept pending F7")

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
                accepted.append((f, ftype, t_occ, cited))

        created = reaffirmed = 0
        batch = self.get_db()
        try:
            for f, ftype, t_occ, cited in accepted:
                _, was_created = self.store.add_fact(
                    f["text"], fact_type=ftype,
                    t_mentioned=session_date, t_occurred=t_occ,
                    source_session_id=session_id, source_turn_ids=cited,
                    extraction_model=self.model, lang_source=lang_source,
                    agent_id=agent_id, user_id=user_id, db=batch,
                )
                created += was_created
                reaffirmed += (not was_created)
            batch.add(ConsolidationLog(
                session_id=session_id, turns_processed=len(turn_data),
                summaries_generated=created,
                truncated_chars=truncated_chars,      # persisted (R2-M4)
                rejected_count=len(rejected),          # persisted (R1-M7)
                rejections_json=json.dumps(
                    [(str(f.get("text"))[:120], p) for f, p in rejected]),
                duration_seconds=(datetime.utcnow() - started).total_seconds(),
                triggered_by="consolidation_v2",
            ))
            batch.commit()
        except Exception:
            batch.rollback()
            raise
        finally:
            batch.close()

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
            "numbers_audit": numbers_audit,   # which values licensed by
                                              # which turns; exempted date
                                              # digits (R5-M5)
            "model": self.model, "session_date": session_date,
        }
        logger.info(f"[ConsolidationV2] {report}")
        return report

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
- fact_type: "event" = something that ALREADY HAPPENED at a time; "state" = an ongoing situation, including PLANS ("The user plans to attend X on DATE" is a state, never an event); "preference" = a like/dislike/choice; "identity" = who the user is.
- t_occurred: the date the event happened, YYYY/MM/DD (or YYYY/MM if only the month is known) — resolve relative references ("last Tuesday", "two weeks ago") against the session date {session_date}, NEVER against today. null if no date is stated or implied.
- NEVER extract a fact that merely restates what the user ASKED or wants to learn — questions and curiosity are not facts about the user's life.
- ONLY facts stated by the USER about their own life. Never extract the assistant's knowledge, recommendations, availability information, or tool/system output.

Session date: {session_date}
Transcript:
{transcript}"""
