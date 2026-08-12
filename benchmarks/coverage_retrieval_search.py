"""
GOLD-SESSION COVERAGE OPTIMISER ($0, no LLM calls).

WHY THIS IS THE RIGHT TARGET (measured, DECISION_AND_FAILURE_LOG §3.1z)
Accuracy in the banked 72.0% run, bucketed by how many of a question's
GOLD SESSIONS reached the assembled context:

    ALL gold sessions present -> 87/103 = 84.5%   (ceiling is 86.7%)
    PARTIAL                   -> 13/30  = 43.3%
    NONE                      ->  8/17  = 47.1%

**When retrieval is complete, the system already performs AT CEILING.**
The whole remaining gap is incomplete multi-hop retrieval, concentrated
exactly where we are weakest:
    multi-session      ALL 20/39   temporal ALL 17/40
    (single-session categories: 20/20, 20/20, 18/20 — one session needed,
     one session delivered)

WHY WE NEVER SAW IT: the repo's headline retrieval metric is gold recall
0.967 — "did ANY gold evidence arrive". A question needing 4 sessions and
receiving 1 scores as RECALLED and is then answered wrong. Recall@k is
structurally the wrong instrument for multi-hop memory.

WHAT THIS FILE DOES
Coverage is a $0 PROXY for accuracy, so retrieval strategies can be swept
without spending a cent; only a strategy that moves COVERAGE earns a paid
accuracy run. Strategies are evaluated on the SAME budget so the
comparison is like-for-like.

  A CURRENT     global top-k by TF-IDF, filled to budget
                (real_code_utils.TfIdfChromaAdapter + assembler:259-261)
  B SESSION_RR  rank SESSIONS by their best-scoring chunk, then round-robin
                the budget across the top-M sessions
  C HYBRID      half the budget to A (depth where relevance is high), half
                to B (breadth across sessions)

NOT A REPEAT OF THE REFUTED LEDGER. "coverage mode" (refuted, worse) CLIPPED
every keyword-matching turn to 240 chars — it destroyed the detail it
retrieved. "diversity caps" and "session-level" were swept in the 66%
raw-turn era, against ACCURACY, with gpt-4o-mini, and without knowing the
mechanism. This sweeps against a measured mechanism, at $0, on the current
architecture.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

SEM_BUDGET = int(24000 * 0.79 // 4)
APPROX_CHUNK_TOKENS = 60


def _fill(ordered, budget):
    picked, used = [], 0
    for c in ordered:
        t = max(1, len(c) // 4)
        if used + t > budget:
            break
        picked.append(c)
        used += t
    return picked


def strat_current(sess_chunks, sims, budget):
    """Global top-k by score — what ships today."""
    flat = sorted(((s, c, sid) for sid, cs in sess_chunks.items()
                   for c, s in zip(cs, sims[sid])), key=lambda x: -x[0])
    return _fill([c for s, c, _ in flat if s > 0.01], budget)


def strat_session_rr(sess_chunks, sims, budget, top_m=12):
    """Rank SESSIONS by their best chunk, then round-robin across the top
    M so several sessions are represented instead of one dominating."""
    best = sorted(((max(sims[sid]) if len(sims[sid]) else 0.0, sid)
                   for sid in sess_chunks), key=lambda x: -x[0])[:top_m]
    queues = []
    for _, sid in best:
        order = sorted(zip(sims[sid], sess_chunks[sid]), key=lambda x: -x[0])
        queues.append([c for s, c in order if s > 0.01])
    out, i = [], 0
    while any(queues):
        for qd in queues:
            if qd:
                out.append(qd.pop(0))
        i += 1
        if i > 200:
            break
    return _fill(out, budget)


def strat_hybrid(sess_chunks, sims, budget, top_m=12):
    half = budget // 2
    a = strat_current(sess_chunks, sims, half)
    b = strat_session_rr(sess_chunks, sims, budget - half, top_m)
    seen, out = set(), []
    for c in a + b:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return _fill(out, budget)


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from corpus_loaders import load_longmemeval

    ds = load_longmemeval(n_queries=150, seed=42, split="s")
    mem = {m.mid: m for m in ds.memories}
    res = {r["question"].strip(): r for r in json.load(
        open(HERE / "qa_accuracy_longmemeval_TRUE150_4o.json"))["results"]}

    strategies = {
        "A current      ": strat_current,
        "B session_rr   ": strat_session_rr,
        "C hybrid       ": strat_hybrid,
    }
    cov = {k: defaultdict(lambda: [0, 0, 0, 0]) for k in strategies}

    for qi, q in enumerate(ds.queries, 1):
        if q.question.strip() not in res:
            continue
        sess_chunks, sims = {}, {}
        for k in q.scope_keys:
            m = mem.get(k)
            if not m:
                continue
            cs = [t.get("content", "") for t in m.turns if t.get("content")]
            if cs:
                sess_chunks[k] = cs
        if not sess_chunks:
            continue
        allc = [c for cs in sess_chunks.values() for c in cs]
        if len(allc) < 3:
            continue
        vec = TfidfVectorizer(max_features=512, sublinear_tf=True, min_df=1)
        M = vec.fit_transform(allc)
        qv = vec.transform([q.question])
        flat = cosine_similarity(qv, M)[0]
        i = 0
        for sid, cs in sess_chunks.items():
            sims[sid] = list(flat[i:i + len(cs)])
            i += len(cs)

        gold = [k for k in q.gold_keys if k in sess_chunks]
        for name, fn in strategies.items():
            picked = set(fn(sess_chunks, sims, SEM_BUDGET))
            hit = sum(1 for g in gold
                      if any(c in picked for c in sess_chunks[g]))
            d = cov[name][q.question_type]
            d[3] += 1
            if gold and hit == len(gold):
                d[0] += 1
            elif hit:
                d[1] += 1
            else:
                d[2] += 1
        if qi % 40 == 0:
            print(f"  ...{qi}", flush=True)

    print(f"\n{'=' * 74}\nGOLD-SESSION COVERAGE by retrieval strategy "
          f"(same {SEM_BUDGET}-token budget)\n{'=' * 74}")
    for name in strategies:
        t = [sum(v[i] for v in cov[name].values()) for i in range(4)]
        print(f"\n{name}  ALL={t[0]:3d}  PARTIAL={t[1]:3d}  NONE={t[2]:3d}"
              f"   (n={t[3]})")
        for cat in ("multi-session", "temporal-reasoning"):
            d = cov[name][cat]
            print(f"     {cat:22s} ALL={d[0]:3d}/{d[3]:<3d} "
                  f"PARTIAL={d[1]:3d} NONE={d[2]:3d}")

    base = [sum(v[i] for v in cov["A current      "].values())
            for i in range(4)]
    print(f"\nΔ ALL-coverage vs current:")
    for name in strategies:
        t = [sum(v[i] for v in cov[name].values()) for i in range(4)]
        print(f"  {name} {t[0] - base[0]:+d} questions")
    print("\nAt the measured 84.5% conversion for ALL-coverage vs ~44% for "
          "partial/none, each recovered question is worth ~0.4 correct.")


if __name__ == "__main__":
    main()
