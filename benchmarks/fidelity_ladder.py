"""
FIDELITY LADDER v2 ($0, no LLM) — one variable per rung.

Replaces `extraction_fidelity.py`, whose rungs varied FOUR things at once
(see DECISION_AND_FAILURE_LOG §3.1f, CORRECTION 2). That instrument
attributed −11.4 points to "our extraction prompt" when rung 2 was a
Claude-Haiku cache our prompt never touched; it also compared dated text
against undated text, hid superseded facts, and let longer text win a
word-overlap test.

THE QUESTION: at each stage of the real pipeline, is the gold answer
still recoverable — from the representation the PRODUCT ACTUALLY SERVES?

  R0  RAW-FULL     every raw turn of the gold session.
                   UPPER BOUND. Not budget-matched; label, never compare.
  R1  RAW-RETRIEVED  what the product's raw-turn retriever actually
                   surfaces from the FULL haystack, at the real budget.
  R2  FACTS-FULL   every stored fact of the gold session, rendered by the
                   PRODUCT'S renderer (dates + type markers included).
                   Upper bound for the fact representation.
  R3  FACTS-RETRIEVED  what FactRetriever actually surfaces from the FULL
                   haystack, at the real budget. What ships.

CONTROLLED, one variable per comparison:
  R0 -> R2  representation   (same session, same budget-free basis)
  R1 -> R3  what the answerer really sees, RAW vs FACTS, budget-matched
  R0 -> R1  retrieval loss on raw     R2 -> R3  retrieval loss on facts

R1 vs R3 IS THE DECISIVE ROUTING TEST. If R1 is high where R3 is low, the
43.8-point loss (§3.1a) is a ROUTING problem — serve the right
representation per question — not an extraction problem, and the
extraction contract must not be touched.

BUDGET: both retrieved rungs get the FULL semantic allocation the eval
sets (24000 * 0.79 // 4 = 4740 tokens), so neither is handicapped by the
65/35 tier split. That isolates representation from budget policy.

SUPERSESSION: counted BOTH ways and reported separately — a
knowledge-update gold answer is sometimes the OLD value, so filtering
`superseded_by IS NULL` silently marks correct history as lost.

COVERAGE: the live corpus only covers the ms+temporal haystacks (79 of
150). R2/R3 are reported on that subset ONLY, and R0/R1 are ALSO printed
restricted to the same 79 so every comparison is like-for-like.
"""
import re
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

CORPUS = HERE / "extracted_memories" / "gate_c_facts.db"
SEM_BUDGET = int(24000 * 0.79 // 4)          # 4740 — the eval's own value
APPROX_CHUNK_TOKENS = 60                      # context_assembler.py:259

_STOP = {
    "the", "and", "for", "with", "that", "this", "you", "your", "was", "were",
    "have", "has", "had", "are", "not", "but", "from", "they", "their", "them",
    "when", "what", "which", "would", "could", "should", "about", "into",
    "user", "users", "there", "then", "than", "some", "just", "also", "been",
    "will", "can", "may", "his", "her", "its", "our", "out", "one", "all",
}


def _words(t):
    return {w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in _STOP}


def _numbers(t):
    return set(re.findall(r"\d+(?:\.\d+)?", t or ""))


def recoverable(gold: str, hay: str) -> bool:
    """Numeric answers require the NUMBER to survive — the §3.1a failure
    mode. Otherwise >=60% of gold content words must appear."""
    if not hay:
        return False
    gn = _numbers(gold)
    if gn:
        return all(n in _numbers(hay) for n in gn)
    gw = _words(gold)
    if not gw:
        return gold.strip().lower() in hay.lower()
    return len(gw & _words(hay)) / len(gw) >= 0.60


def tfidf_retrieve(turns, query, budget_tokens):
    """The product's raw-turn retrieval path, replicated exactly:
    TfIdfChromaAdapter.search + the assembler's budget-aware top_k
    (real_code_utils.py:20-48, context_assembler.py:259-261). Run over
    in-memory turns so this never touches the shared dev DB (F-15)."""
    contents = [c for c in turns if c]
    if len(contents) < 3:
        return " ".join(contents)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    top_k = max(5, min(200, budget_tokens // APPROX_CHUNK_TOKENS))
    vec = TfidfVectorizer(max_features=512, sublinear_tf=True, min_df=1)
    matrix = vec.fit_transform(contents)
    sims = cosine_similarity(vec.transform([query]), matrix)[0]
    idx = sims.argsort()[-top_k:][::-1]
    picked, used = [], 0
    for i in idx:
        if sims[i] <= 0.01:
            continue
        t = max(1, len(contents[i]) // 4)
        if used + t > budget_tokens:
            break
        picked.append(contents[i])
        used += t
    return " ".join(picked)


def main():
    from corpus_loaders import load_longmemeval
    from agentmem_os.llm.fact_retrieval import FactRetriever
    from agentmem_os.llm.token_counter import TokenCounter
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker

    ds = load_longmemeval(n_queries=150, seed=42, split="s")
    mem = {m.mid: m for m in ds.memories}

    engine = create_engine(f"sqlite:///{CORPUS}",
                           connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _ro(conn, _):
        cur = conn.cursor()
        cur.execute("PRAGMA query_only=ON")
        cur.close()

    Session = sessionmaker(bind=engine, expire_on_commit=False)
    retriever = FactRetriever(Session)
    counter = TokenCounter()

    con = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
    covered = {r[0] for r in con.execute(
        "SELECT DISTINCT source_session_id FROM semantic_facts")}
    con.close()

    rows = []
    for q in ds.queries:
        gold_ids = list(q.gold_keys)
        raw_full = " ".join(
            t.get("content", "") for k in gold_ids
            for t in (mem[k].turns if k in mem else []))
        hay_turns = [t.get("content", "") for k in q.scope_keys
                     if k in mem for t in mem[k].turns]
        r1 = tfidf_retrieve(hay_turns, q.question, SEM_BUDGET)

        has_corpus = any(k in covered for k in gold_ids)
        r2 = r3 = None
        if has_corpus:
            # R2: the gold session's facts, rendered by the PRODUCT'S
            # renderer (dates + type markers), budget deliberately huge.
            r2 = retriever.build_block(q.question, token_budget=100000,
                                       session_ids=gold_ids) or ""
            # R3: what actually ships — full haystack, real budget.
            r3 = retriever.build_block(q.question, token_budget=SEM_BUDGET,
                                       session_ids=list(q.scope_keys)) or ""
        rows.append({
            "type": q.question_type, "corpus": has_corpus,
            "R0": recoverable(q.gold_answer, raw_full),
            "R1": recoverable(q.gold_answer, r1),
            "R2": recoverable(q.gold_answer, r2) if r2 is not None else None,
            "R3": recoverable(q.gold_answer, r3) if r3 is not None else None,
            "r1_tok": counter.count(r1), "r3_tok": counter.count(r3 or ""),
            "numeric": bool(_numbers(q.gold_answer)),
        })

    def rate(key, sub):
        sel = [r for r in sub if r[key] is not None]
        if not sel:
            return "n/a"
        n = sum(1 for r in sel if r[key])
        return f"{n}/{len(sel)} = {n / len(sel):5.1%}"

    slice_rows = [r for r in rows if r["corpus"]]
    print("=" * 70)
    print(f"FIDELITY LADDER v2 — budget {SEM_BUDGET} tokens/rung")
    print("=" * 70)
    print(f"\n--- LIKE-FOR-LIKE, the {len(slice_rows)} questions the live "
          f"corpus covers ---")
    for k, lab in (("R0", "R0 RAW-FULL       (upper bound, unbudgeted)"),
                   ("R1", "R1 RAW-RETRIEVED  (product path, budgeted)"),
                   ("R2", "R2 FACTS-FULL     (upper bound, rendered)"),
                   ("R3", "R3 FACTS-RETRIEVED(what ships, budgeted)")):
        print(f"  {lab:44s} {rate(k, slice_rows)}")

    print("\n  DECISIVE — same budget, same haystack, same questions:")
    print(f"    R1 raw-retrieved  {rate('R1', slice_rows)}")
    print(f"    R3 facts-retrieved{rate('R3', slice_rows)}")

    print(f"\n--- ALL {len(rows)} questions (R0/R1 only; no corpus for 71) ---")
    print(f"  R0 RAW-FULL      {rate('R0', rows)}")
    print(f"  R1 RAW-RETRIEVED {rate('R1', rows)}")

    print("\nBY TYPE on the covered slice (R0 / R1 / R2 / R3):")
    for t in sorted({r["type"] for r in slice_rows}):
        s = [r for r in slice_rows if r["type"] == t]
        print(f"  {t:24s} {rate('R0', s)} | {rate('R1', s)} | "
              f"{rate('R2', s)} | {rate('R3', s)}")

    for lab, sub in (("NUMERIC", [r for r in slice_rows if r["numeric"]]),
                     ("NON-NUMERIC",
                      [r for r in slice_rows if not r["numeric"]])):
        print(f"\n{lab} (n={len(sub)}): R0 {rate('R0', sub)} | "
              f"R1 {rate('R1', sub)} | R2 {rate('R2', sub)} | "
              f"R3 {rate('R3', sub)}")

    n = max(1, len(slice_rows))
    print(f"\nBUDGET USE (mean tokens): raw-retrieved "
          f"{sum(r['r1_tok'] for r in slice_rows) / n:.0f} | "
          f"facts-retrieved {sum(r['r3_tok'] for r in slice_rows) / n:.0f} "
          f"of {SEM_BUDGET}")
    both = [r for r in slice_rows if r["R1"] is not None and r["R3"] is not None]
    only_raw = sum(1 for r in both if r["R1"] and not r["R3"])
    only_fac = sum(1 for r in both if r["R3"] and not r["R1"])
    print(f"ROUTING SIGNAL: raw-only recoverable {only_raw} | "
          f"facts-only recoverable {only_fac} | n={len(both)}")


if __name__ == "__main__":
    main()
