"""
RERANKER PRECISION TEST ($0, local) — does two-stage retrieval deliver
cleaner evidence into the same budget?

THE MEASURED PROBLEM (DECISION_AND_FAILURE_LOG §3.1h/§3.1i): handing the
model only the gold sessions scores 82%; the same model over the full
47.8-session haystack scores 66% on the same 150 questions. The evidence
is retrieved and then LOST AMONG DISTRACTORS.

WHY THIS AND NOT SOMETHING FROM THE LEDGER: dense, sliding windows,
hybrid RRF, recency, diversity caps, session-level, session-filter
hybrids, chronological and coverage-mode are all ALREADY REFUTED and are
not re-run here. A cross-encoder reranker is a different mechanism: RRF
FUSES ranked lists produced independently of the question, a cross-encoder
RESCORES each (question, passage) pair jointly. Nothing in the ledger
tests that.

ONE VARIABLE. Both arms use the same questions, the same haystack, the
same 4,740-token budget and the same budget-fill loop. Only the ORDER in
which candidates are offered differs:
  BASELINE  TF-IDF cosine rank                        (what ships today)
  RERANKED  TF-IDF top-N candidates, then cross-encoder rescoring

Success is NOT "the reranker looks smart" — it is: does the gold answer
survive into the budget MORE OFTEN? Measured with the CALIBRATED detector
(answer_presence: lexical, 96.4% balanced accuracy, beat embeddings on
its own validation set), numeric answers judged by the strict rule.

If this does not lift survival it is DEAD and gets recorded as refuted,
like the other ten.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

SEM_BUDGET = int(24000 * 0.79 // 4)      # 4740 — the eval's own value
APPROX_CHUNK_TOKENS = 60
CANDIDATE_POOL = 200                      # stage-1 depth fed to stage 2
_XENC = "cross-encoder/ms-marco-MiniLM-L-6-v2"
N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 150


def _fill(ordered, budget):
    """Identical budget-fill for both arms — the only difference between
    arms must be the ORDER of `ordered`."""
    picked, used = [], 0
    for c in ordered:
        t = max(1, len(c) // 4)
        if used + t > budget:
            break
        picked.append(c)
        used += t
    return " ".join(picked), used


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import huggingface_hub.constants as _hc
    _hc.HF_HUB_OFFLINE = True
    import numpy as np
    from sentence_transformers import CrossEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from corpus_loaders import load_longmemeval
    import answer_presence as ap

    ds = load_longmemeval(n_queries=N_Q, seed=42, split="s")
    mem = {m.mid: m for m in ds.memories}
    xenc = CrossEncoder(_XENC, max_length=384)
    print(f"reranker: {_XENC} | pool={CANDIDATE_POOL} | budget={SEM_BUDGET}\n")

    base_hit = rr_hit = 0
    base_only = rr_only = 0
    n = 0
    per_type = {}
    for qi, q in enumerate(ds.queries, 1):
        chunks = [t.get("content", "") for k in q.scope_keys if k in mem
                  for t in mem[k].turns if t.get("content")]
        if len(chunks) < 3:
            continue
        vec = TfidfVectorizer(max_features=512, sublinear_tf=True, min_df=1)
        M = vec.fit_transform(chunks)
        sims = cosine_similarity(vec.transform([q.question]), M)[0]
        top_k = max(5, min(200, SEM_BUDGET // APPROX_CHUNK_TOKENS))

        # BASELINE — exactly the shipping path (real_code_utils.py:20-48
        # + context_assembler.py:259-261): TF-IDF order, sim>0.01 filter.
        base_order = [chunks[i] for i in sims.argsort()[-top_k:][::-1]
                      if sims[i] > 0.01]
        base_txt, _ = _fill(base_order, SEM_BUDGET)

        # RERANKED — deeper stage-1 pool, then joint (q, passage) scoring.
        pool_idx = [i for i in sims.argsort()[-CANDIDATE_POOL:][::-1]
                    if sims[i] > 0.0]
        pool = [chunks[i] for i in pool_idx]
        if pool:
            scores = xenc.predict([(q.question, c[:1200]) for c in pool],
                                  batch_size=64, show_progress_bar=False)
            rr_order = [pool[i] for i in np.argsort(scores)[::-1]]
        else:
            rr_order = []
        rr_txt, _ = _fill(rr_order, SEM_BUDGET)

        b = ap._lex(q.gold_answer, base_txt) if not ap.numbers(q.gold_answer) \
            else all(x in ap.numbers(base_txt) for x in ap.numbers(q.gold_answer))
        r = ap._lex(q.gold_answer, rr_txt) if not ap.numbers(q.gold_answer) \
            else all(x in ap.numbers(rr_txt) for x in ap.numbers(q.gold_answer))
        n += 1
        base_hit += b
        rr_hit += r
        base_only += (b and not r)
        rr_only += (r and not b)
        d = per_type.setdefault(q.question_type, [0, 0, 0])
        d[0] += b
        d[1] += r
        d[2] += 1
        if qi % 25 == 0:
            print(f"  ...{qi} questions", flush=True)

    print(f"\n{'=' * 62}\nGOLD-ANSWER SURVIVAL INTO A {SEM_BUDGET}-TOKEN "
          f"BUDGET (n={n})\n{'=' * 62}")
    print(f"  BASELINE (TF-IDF order)      : {base_hit}/{n} = "
          f"{base_hit / n:.1%}")
    print(f"  RERANKED (cross-encoder)     : {rr_hit}/{n} = {rr_hit / n:.1%}")
    print(f"  delta                        : {rr_hit - base_hit:+d} questions")
    print(f"  reranker-only wins {rr_only} | baseline-only wins {base_only}")

    # McNemar exact, paired — the same test used for every accuracy claim.
    import math
    nn = rr_only + base_only
    p = (sum(math.comb(nn, i) for i in range(0, min(rr_only, base_only) + 1))
         / 2 ** nn * 2) if nn else 1.0
    print(f"  McNemar exact p = {min(1.0, p):.4f}"
          f"{'  (SIGNIFICANT)' if min(1.0, p) < 0.05 else '  (not significant)'}")

    print("\nBY TYPE (baseline -> reranked):")
    for t, (b, r, tot) in sorted(per_type.items()):
        print(f"  {t:28s} {b:3d}/{tot:<3d} -> {r:3d}/{tot:<3d}  "
              f"({(r - b):+d})")


if __name__ == "__main__":
    main()
