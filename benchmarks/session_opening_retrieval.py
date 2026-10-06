"""Bounded lexical session expansion; no gold labels, IDs or model outputs used."""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    role: str
    text: str


@dataclass(frozen=True)
class Session:
    observed_at: str
    turns: tuple[Turn, ...]


MAX_SESSIONS = 8
MAX_EXTRA_CHARS = 4000
CHAR_CAP = 40000
HEADER = "\n\n[ADDITIONAL ORIGINAL SESSION OPENINGS: may be irrelevant; not answers]\n"


def expand(question: str, packet: str, sessions: tuple[Session, ...]):
    """Rank sessions by best user-turn TF-IDF score; append missing first user turns.

    Sessions and turns contain only trusted date/role/text, not benchmark labels.
    Ranking does not certify relevance. Preserve originals and report nonfit.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not question.strip() or len(packet) > CHAR_CAP:
        raise ValueError("Invalid question or original packet budget")
    if not isinstance(sessions, tuple) or any(type(s) is not Session for s in sessions):
        raise ValueError("Explicit session projection required")
    flat = []
    for si, session in enumerate(sessions):
        for ti, turn in enumerate(session.turns):
            if type(turn) is not Turn or turn.role not in {"user", "assistant"}:
                raise ValueError("Invalid source attribution")
            if turn.role == "user" and turn.text.strip():
                flat.append((si, ti, turn.text))
    report = dict(
        policy="session-opening-v1",
        receipts=[],
        skipped=[],
        ranked_sessions=[],
        baseline_sha256=hashlib.sha256(packet.encode()).hexdigest(),
    )
    if not flat:
        report.update(status="UNCHANGED_NO_SOURCES", candidate_sha256=report["baseline_sha256"])
        return packet, report
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    analyzer = vectorizer.build_analyzer()
    if not any(analyzer(t[2]) for t in flat):
        report.update(status="UNCHANGED_NO_TERMS", candidate_sha256=report["baseline_sha256"])
        return packet, report
    matrix = vectorizer.fit_transform([t[2] for t in flat])
    scores = cosine_similarity(vectorizer.transform([question]), matrix)[0]
    best = {}
    for (si, ti, _), score in zip(flat, scores, strict=True):
        if score > 0 and (si not in best or score > best[si][0]):
            best[si] = (float(score), ti)
    ranked = sorted(best, key=lambda si: (-best[si][0], si))[:MAX_SESSIONS]
    block = HEADER
    seen = set()
    for si in ranked:
        session = sessions[si]
        ti = next(i for i, t in enumerate(session.turns) if t.role == "user" and t.text.strip())
        turn = session.turns[ti]
        report["ranked_sessions"].append(
            dict(session=si, anchor_turn=best[si][1], score=best[si][0])
        )
        if turn.text in packet or turn.text in seen:
            report["skipped"].append(dict(session=si, reason="already_present"))
            continue
        prefix = f"[user | observed {session.observed_at}]\n"
        addition = prefix + turn.text + "\n"
        if len(block) + len(addition) > min(MAX_EXTRA_CHARS, CHAR_CAP - len(packet)):
            report["skipped"].append(dict(session=si, reason="budget_nonfit"))
            continue
        start = len(packet) + len(block) + len(prefix)
        block += addition
        seen.add(turn.text)
        report["receipts"].append(
            dict(
                session=si,
                turn=ti,
                role=turn.role,
                observed_at=session.observed_at,
                source_sha256=hashlib.sha256(turn.text.encode()).hexdigest(),
                start=start,
                end=start + len(turn.text),
            )
        )
    candidate = packet + block if report["receipts"] else packet
    report.update(
        status="APPLIED" if report["receipts"] else "UNCHANGED",
        candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
    )
    return candidate, report
