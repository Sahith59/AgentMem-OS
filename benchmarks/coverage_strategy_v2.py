"""
COVERAGE STRATEGY SWEEP v2 ($0) — vocabulary mismatch and the unused date.

WHY (measured, DECISION_AND_FAILURE_LOG §3.1z / §3.1ab)
Gold-session coverage predicts accuracy: ALL gold sessions in context ->
84.5% correct, partial/none -> ~44%. After retuning FACTS_BUDGET_SHARE the
remaining shortfall is concentrated in temporal (25/40 ALL) and
multi-session (25/39 ALL).

Inspecting the temporal failures, EVERY "not mentioned" answer turned out
to be a RETRIEVAL failure, not a reasoning one:
    "sports event two weeks ago"        gold sessions in context 0/3
    "months between the degrees"        0/2
    "music event last Saturday"         2/5
    "life event of a relative"          1/2
A temporal SORTER would fix none of these — the evidence never arrives.
Two causes are visible in the questions themselves:

  1. VOCABULARY MISMATCH. "sports event two weeks ago" must match a
     session about a "charity soccer tournament". TF-IDF cannot bridge it.
  2. AN UNUSED SIGNAL. The question carries `question_date`, and phrases
     like "two weeks ago" / "last Saturday" name a computable TARGET DATE.
     Every turn in the corpus is timestamped, and in any real agent memory
     every turn has created_at — so this is platform-agnostic, not a
     LongMemEval trick. We currently ignore it entirely at retrieval time.

STRATEGIES (same 4,740-token budget, coverage measured identically)
  A CURRENT    TF-IDF, global top-k                      (ships today)
  B DENSE      e5-small embeddings                       (in the refuted
               ledger — see the note below)
  C HYBRID     TF-IDF + dense, score-normalised sum
  D DATEBOOST  TF-IDF + a boost for chunks whose timestamp falls in the
               window named by the question's relative phrase
  E HYBRID+DATE  C and D together

ON RE-TESTING "DENSE", WHICH THE LEDGER ALREADY REFUTES: it was refuted in
the 66%-era raw-turn configuration, measured against ACCURACY with
gpt-4o-mini, on ~79 questions — an instrument far too blunt to see a
coverage effect. We now have a $0 proxy with a MEASURED relationship to
accuracy. Re-testing a refuted idea with a better instrument is legitimate;
re-testing it with the same instrument and hoping would not be. If it does
not move coverage it goes back in the ledger, permanently.
"""
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

SEM_BUDGET = int(24000 * 0.79 // 4)
RAW_SHARE = 0.65                      # what raw turns get at facts=0.35
RAW_BUDGET = int(SEM_BUDGET * RAW_SHARE)

_STAMP = re.compile(r"\[(\d{4})/(\d{2})/(\d{2})")
_REL = [
    (re.compile(r"(\d+|a|one|two|three|four|five|six)\s+(day|week|month)s?\s+ago",
                re.I), None),
    (re.compile(r"\blast\s+(saturday|sunday|monday|tuesday|wednesday|thursday|friday|week|month)\b",
                re.I), None),
    (re.compile(r"\byesterday\b", re.I), None),
]
_WORDNUM = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6}


def target_window(question, qdate):
    """The date window a relative phrase names, or None.

    Generic English relative-time phrases only — no question-specific
    strings. Returns (lo, hi) dates; the window is deliberately WIDE
    (+/- 6 days) because users are imprecise ("two weeks ago" rarely
    means exactly 14 days) and a narrow window would trade a retrieval
    miss for a different retrieval miss."""
    if not qdate:
        return None
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})", qdate)
    if not m:
        return None
    now = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    q = question.lower()
    mm = re.search(r"(\d+|a|one|two|three|four|five|six)\s+(day|week|month)s?\s+ago", q)
    if mm:
        n = _WORDNUM.get(mm.group(1), None)
        if n is None:
            try:
                n = int(mm.group(1))
            except ValueError:
                return None
        unit = mm.group(2)
        days = n * (1 if unit == "day" else 7 if unit == "week" else 30)
        c = now - timedelta(days=days)
        return c - timedelta(days=6), c + timedelta(days=6)
    if re.search(r"\blast\s+(saturday|sunday|monday|tuesday|wednesday|thursday|friday)\b", q):
        return now - timedelta(days=10), now
    if re.search(r"\blast\s+week\b", q):
        return now - timedelta(days=14), now - timedelta(days=3)
    if re.search(r"\blast\s+month\b", q):
        return now - timedelta(days=45), now - timedelta(days=15)
    if re.search(r"\byesterday\b", q):
        return now - timedelta(days=2), now
    return None


def chunk_date(text):
    m = _STAMP.search(text or "")
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _fill(pairs, budget):
    """pairs: [(score, chunk)] already ordered."""
    out, used = [], 0
    for _, c in pairs:
        t = max(1, len(c) // 4)
        if used + t > budget:
            break
        out.append(c)
        used += t
    return out


def main():
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from corpus_loaders import load_longmemeval
    import answer_presence as ap

    ds = load_longmemeval(n_queries=150, seed=42, split="s")
    mem = {m.mid: m for m in ds.memories}
    model = ap._get_model()

    cov = {k: defaultdict(lambda: [0, 0, 0, 0])
           for k in ("A current", "B dense", "C hybrid", "D dateboost",
                     "E hyb+date")}

    for qi, q in enumerate(ds.queries, 1):
        chunks, owner = [], []
        for k in q.scope_keys:
            m = mem.get(k)
            if not m:
                continue
            for t in m.turns:
                c = t.get("content", "")
                if c:
                    chunks.append(c)
                    owner.append(k)
        if len(chunks) < 3:
            continue

        vec = TfidfVectorizer(max_features=512, sublinear_tf=True, min_df=1)
        M = vec.fit_transform(chunks)
        lex = cosine_similarity(vec.transform([q.question]), M)[0]

        emb = model.encode([f"passage: {c[:400]}" for c in chunks],
                           normalize_embeddings=True, batch_size=256,
                           show_progress_bar=False).astype(np.float32)
        qe = model.encode([f"query: {q.question}"],
                          normalize_embeddings=True,
                          show_progress_bar=False).astype(np.float32)
        den = (qe @ emb.T)[0]

        def norm(v):
            v = np.asarray(v, dtype=np.float32)
            r = v.max() - v.min()
            return (v - v.min()) / r if r > 1e-9 else v * 0

        win = target_window(q.question, getattr(q, "question_date", ""))
        boost = np.zeros(len(chunks), dtype=np.float32)
        if win:
            lo, hi = win
            for i, c in enumerate(chunks):
                d = chunk_date(c)
                if d and lo <= d <= hi:
                    boost[i] = 1.0

        scores = {
            "A current": lex,
            "B dense": den,
            "C hybrid": norm(lex) + norm(den),
            "D dateboost": norm(lex) + 0.5 * boost,
            "E hyb+date": norm(lex) + norm(den) + 0.5 * boost,
        }
        gold = set(q.gold_keys)
        for name, sc in scores.items():
            order = sorted(range(len(chunks)), key=lambda i: -sc[i])
            picked = _fill([(sc[i], chunks[i]) for i in order], RAW_BUDGET)
            got = {owner[i] for i in order[:len(picked)]}
            present = {g for g in gold if g in got and g in mem}
            tot = len([g for g in gold if g in mem])
            d = cov[name][q.question_type]
            d[3] += 1
            if tot and len(present) == tot:
                d[0] += 1
            elif present:
                d[1] += 1
            else:
                d[2] += 1
        if qi % 30 == 0:
            print(f"  ...{qi}", flush=True)

    print(f"\n{'=' * 76}\nGOLD-SESSION COVERAGE by strategy "
          f"(raw-turn budget {RAW_BUDGET} tokens)\n{'=' * 76}")
    base = None
    for name in cov:
        t = [sum(v[i] for v in cov[name].values()) for i in range(4)]
        if base is None:
            base = t[0]
        ms = cov[name]["multi-session"]
        tr = cov[name]["temporal-reasoning"]
        print(f"\n{name:12s} ALL={t[0]:3d}  PARTIAL={t[1]:3d}  NONE={t[2]:3d}"
              f"   Δ={t[0] - base:+d}")
        print(f"             multi-session ALL={ms[0]:2d}/{ms[3]:<2d}   "
              f"temporal ALL={tr[0]:2d}/{tr[3]:<2d}")


if __name__ == "__main__":
    main()
