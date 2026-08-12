"""
ITERATIVE RETRIEVAL PROTOTYPE ($0) — pseudo-relevance feedback against the
29 systematic failures.

TARGET (DECISION_AND_FAILURE_LOG §3.1ac + pooled 3-run analysis)
29 questions fail in ALL THREE identical runs — the systematic core.
24 of them are two shapes:
  * cross-session COUNTING ("how many rollercoasters across all events")
    — needs EVERY instance session; partial coverage → undercount
  * RELATIVE-DATE recall ("sports event two weeks ago") — the question
    names a TIME, the evidence names a THING; zero vocabulary overlap

MECHANISM UNDER TEST: pseudo-relevance feedback (PRF/Rocchio — standard
IR since the 1970s, not an invention). Pass 1 ranks normally; the top
seed turns then contribute a third ranking signal (dense similarity to
the seeds), fused by the same RRF the product already uses. The idea:
"sports event" weakly matches SOME sports-adjacent turn; that turn's
neighborhood names the soccer tournament; similarity-to-seed retrieves
the naming turn even though it shares no words with the query.

RISK COVERED: drift. The seed signal is ONE additional RRF list among
three — the query keeps two votes, expansion one. Variants sweep seed
count and fusion so the choice is measured, not assumed.

VERDICT RULE (fixed before running):
  adopt IFF (a) ALL-coverage on the full 150 does not drop, AND
           (b) gold-session coverage on the 29 systematic failures
               improves by >= 3 questions.
Otherwise it goes in the refuted ledger next to session-RR and date-boost.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

SEM_BUDGET = int(24000 * 0.79 // 4)
RAW_BUDGET = int(SEM_BUDGET * 0.65)          # raw turns' share at facts=0.35
RUNS = ["qa_accuracy_longmemeval_dense.json",
        "qa_accuracy_longmemeval_dense_repeat.json",
        "qa_accuracy_longmemeval_dense_run3.json"]


def systematic_failures():
    R = [{r["question"].strip(): bool(r["correct"])
          for r in json.load(open(HERE / f))["results"]} for f in RUNS]
    q = set(R[0]) & set(R[1]) & set(R[2])
    return {x for x in q if not any(r[x] for r in R)}


def rrf(lists, n):
    s = [0.0] * n
    for order in lists:
        for rank, idx in enumerate(order):
            s[int(idx)] += 1.0 / (61 + rank)
    return s


def fill_spans(order, turns, budget, ctx=2):
    """The product's span expansion + the assembler's budget fill."""
    covered, out, used = set(), [], 0
    n = len(turns)
    for idx in order:
        i = int(idx)
        if i in covered:
            continue
        lo, hi = max(0, i - ctx), min(n, i + ctx + 1)
        span = "\n".join(turns[lo:hi])
        t = max(1, len(span) // 4)
        if used + t > budget:
            break
        covered |= set(range(lo, hi))
        out.append((lo, hi, span))
        used += t
    return out


def main():
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from corpus_loaders import load_longmemeval
    import answer_presence as ap

    sysfail = systematic_failures()
    print(f"systematic failures loaded: {len(sysfail)}")
    ds = load_longmemeval(n_queries=150, seed=42, split="s")
    mem = {m.mid: m for m in ds.memories}
    model = ap._get_model()

    VARIANTS = ["base", "prf5", "prf10", "prf5x2"]
    cov = {v: defaultdict(lambda: [0, 0]) for v in VARIANTS}   # [ALL, n]
    sys_cov = {v: [0, 0] for v in VARIANTS}                    # on the 29

    for qi, q in enumerate(ds.queries, 1):
        turns, owner = [], []
        for k in q.scope_keys:
            m = mem.get(k)
            if not m:
                continue
            for t in m.turns:
                c = t.get("content", "")
                if c:
                    turns.append(c)
                    owner.append(k)
        if len(turns) < 3:
            continue
        n = len(turns)

        emb = model.encode([f"passage: {c[:400]}" for c in turns],
                           normalize_embeddings=True, batch_size=256,
                           show_progress_bar=False).astype(np.float32)
        qe = model.encode([f"query: {q.question}"],
                          normalize_embeddings=True,
                          show_progress_bar=False).astype(np.float32)[0]
        dense = emb @ qe
        vec = TfidfVectorizer(max_features=512, sublinear_tf=True, min_df=1)
        M = vec.fit_transform(turns)
        lex = cosine_similarity(vec.transform([q.question]), M)[0]
        d_order = list(np.argsort(dense)[::-1])
        l_order = list(np.argsort(lex)[::-1])
        base_scores = rrf([d_order, l_order], n)
        base_order = list(np.argsort(base_scores)[::-1])

        def prf_order(n_seeds, rounds=1):
            order = base_order
            for _ in range(rounds):
                seeds = order[:n_seeds]
                centroid = emb[seeds].mean(axis=0)
                nc = np.linalg.norm(centroid)
                if nc > 1e-9:
                    centroid = centroid / nc
                seed_sims = emb @ centroid
                s_order = list(np.argsort(seed_sims)[::-1])
                scores = rrf([d_order, l_order, s_order], n)
                order = list(np.argsort(scores)[::-1])
            return order

        orders = {
            "base": base_order,
            "prf5": prf_order(5),
            "prf10": prf_order(10),
            "prf5x2": prf_order(5, rounds=2),
        }
        gold = [k for k in q.gold_keys if k in mem]
        for v, order in orders.items():
            spans = fill_spans(order, turns, RAW_BUDGET)
            got = set()
            for lo, hi, _ in spans:
                got |= {owner[i] for i in range(lo, hi)}
            full = bool(gold) and all(g in got for g in gold)
            d = cov[v][q.question_type]
            d[0] += full
            d[1] += 1
            if q.question.strip() in sysfail:
                sys_cov[v][0] += full
                sys_cov[v][1] += 1
        if qi % 30 == 0:
            print(f"  ...{qi}", flush=True)

    print(f"\n{'=' * 70}\nGOLD-SESSION COVERAGE (raw budget {RAW_BUDGET} tok)"
          f"\n{'=' * 70}")
    print(f"{'variant':10s} {'ALL/150':>9s} {'on-29-systematic':>18s} "
          f"{'ms':>7s} {'temporal':>9s}")
    for v in VARIANTS:
        tot = sum(d[0] for d in cov[v].values())
        ms = cov[v]["multi-session"]
        tr = cov[v]["temporal-reasoning"]
        print(f"{v:10s} {tot:9d} {sys_cov[v][0]:8d}/{sys_cov[v][1]:<9d} "
              f"{ms[0]:3d}/{ms[1]:<3d} {tr[0]:4d}/{tr[1]:<3d}")

    base_tot = sum(d[0] for d in cov["base"].values())
    base_sys = sys_cov["base"][0]
    print("\nVERDICT (rule fixed in the docstring):")
    for v in VARIANTS[1:]:
        tot = sum(d[0] for d in cov[v].values())
        ok = tot >= base_tot and sys_cov[v][0] >= base_sys + 3
        print(f"  {v:8s} overall {tot - base_tot:+d}, systematic "
              f"{sys_cov[v][0] - base_sys:+d}  -> "
              f"{'ADOPT-candidate' if ok else 'REJECT'}")


if __name__ == "__main__":
    main()
