"""
RELEVANCE-THRESHOLD SEARCH ($0) — derive the fact tier's admission bar from
the outcomes it actually produced.

THE MEASURED PROBLEM (DECISION_AND_FAILURE_LOG §3.1q)
Run #1 put our real architecture on the full benchmark and LOST 26 questions
against the raw-turn baseline (73/150 vs 99/150, McNemar p=0.0016). 54% of the
loss is single-session-assistant, 17/20 -> 3/20 — the category the extraction
contract refuses to store by design.

WHY IT HAPPENS
`_LEXICAL_FLOOR = 0.01` (llm/fact_retrieval.py:56). Any fact sharing
essentially any word with the query is admitted. Against a 35,053-fact corpus
every query finds hundreds of weak matches, so the tier fills its whole share
with near-noise — and the assembler debits the budget by tokens ACTUALLY USED
(context_assembler.py:213), so a full block of weak facts genuinely starves
the raw-turn tier that holds the answer verbatim.

WHAT THIS MEASURES
For every question: the lexical similarity profile of the facts the tier would
admit, joined to what actually happened — did run #1 (facts) get it right, did
the banked raw-turn run get it right. Then: is there a similarity threshold
that separates "facts helped" from "facts destroyed the answer"?

DISCIPLINE: the threshold is READ OFF the data, not chosen and then justified.
If no threshold separates them, this reports that and the fix is wrong — the
same standard that killed the lexical gate rules and the local classifier.
"""
import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

CORPUS = HERE / "extracted_memories" / "gate_c_facts.db"
SEM_BUDGET = int(24000 * 0.79 // 4)
FACTS_SHARE = 0.65


def main():
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from corpus_loaders import load_longmemeval
    from agentmem_os.db.semantic_facts import SemanticFactStore

    ds = load_longmemeval(n_queries=150, seed=42, split="s")
    banked = {r["question"].strip(): bool(r["correct"])
              for r in json.load(open(HERE / "qa_accuracy_longmemeval.json"))["results"]}
    run1 = {r["question"].strip(): bool(r["correct"])
            for r in json.load(open(
                HERE / "qa_accuracy_longmemeval_gate_d_full150.json"))["results"]}

    engine = create_engine(f"sqlite:///{CORPUS}",
                           connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _ro(c, _):
        cur = c.cursor()
        cur.execute("PRAGMA query_only=ON")
        cur.close()

    Session = sessionmaker(bind=engine, expire_on_commit=False)
    store = SemanticFactStore(Session)

    rows = []
    for i, q in enumerate(ds.queries, 1):
        qt = q.question.strip()
        if qt not in banked or qt not in run1:
            continue
        facts = store.current_facts(session_ids=list(q.scope_keys), limit=500)
        texts = [f.fact_text for f in facts]
        if not texts:
            continue
        vec = TfidfVectorizer(max_features=512, sublinear_tf=True, min_df=1)
        try:
            M = vec.fit_transform(texts)
            sims = cosine_similarity(vec.transform([q.question]), M)[0]
        except ValueError:
            continue
        sims = sorted(sims, reverse=True)
        rows.append({
            "type": q.question_type,
            "top": float(sims[0]),
            "top5": float(sum(sims[:5]) / min(5, len(sims))),
            "n_above_01": int(sum(1 for s in sims if s > 0.01)),
            "banked": banked[qt], "run1": run1[qt],
        })
        if i % 40 == 0:
            print(f"  ...{i}", flush=True)

    lost = [r for r in rows if r["banked"] and not r["run1"]]
    kept = [r for r in rows if r["banked"] and r["run1"]]
    won = [r for r in rows if not r["banked"] and r["run1"]]
    print(f"\nquestions analysed: {len(rows)}")
    print(f"  facts DESTROYED the answer (banked right, run1 wrong): {len(lost)}")
    print(f"  facts KEPT it        (both right)                    : {len(kept)}")
    print(f"  facts WON it         (banked wrong, run1 right)       : {len(won)}")

    def stat(name, sel, k):
        if not sel:
            return
        v = sorted(r[k] for r in sel)
        med = v[len(v) // 2]
        print(f"    {name:34s} n={len(sel):3d}  median={med:.4f}  "
              f"min={v[0]:.4f}  max={v[-1]:.4f}")

    for k in ("top", "top5", "n_above_01"):
        print(f"\n  --- {k} ---")
        stat("facts DESTROYED the answer", lost, k)
        stat("facts KEPT the answer", kept, k)
        stat("facts WON the answer", won, k)

    print("\nIF WE SKIPPED THE FACT TIER BELOW A TOP-SIMILARITY THRESHOLD:")
    print(f"  {'thr':>6}  {'destroyed-recovered':>20}  {'wins-lost':>10}  {'net':>6}")
    for thr in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30):
        rec = sum(1 for r in lost if r["top"] < thr)
        lostwin = sum(1 for r in won if r["top"] < thr)
        print(f"  {thr:6.2f}  {rec:20d}  {lostwin:10d}  {rec - lostwin:+6d}")

    print("\nBY TYPE — median top similarity (the tier's own confidence):")
    from collections import defaultdict
    bt = defaultdict(list)
    for r in rows:
        bt[r["type"]].append(r["top"])
    for t, v in sorted(bt.items()):
        v = sorted(v)
        print(f"  {t:28s} median={v[len(v) // 2]:.4f}  n={len(v)}")


if __name__ == "__main__":
    main()
