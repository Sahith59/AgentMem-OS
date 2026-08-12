"""
AGGREGATION GATE ($0) — can the fact corpus COUNT?

THE LEVER THIS DECIDES (DECISION_AND_FAILURE_LOG §3.1ad)
~14 of the 29 systematic failures are cross-session counting ("how many
rollercoasters across all the events...", gold 10). They fail with
partial coverage AND mostly with full coverage: a language model asked
to count instances scattered through prose miscounts — even the oracle
does. The proposed lever is aggregation-aware answering: the memory
system counts/orders its own DATED ATOMIC FACTS in code and hands the
model the assembled, cited list.

That only works if the facts EXIST. One dated fact per instance is the
extraction contract; Gate A's demo assembled rollercoasters 3+1+3+3=10.
But the contract was never audited per counting question on the full
corpus. This gate does that, before any code is written.

METHOD (per counting question)
  1. scope: the question's haystack sessions (exactly what the shipped
     FactRetriever reads)
  2. rank facts by dense similarity to the question (the product's own
     encoder), report the matching-instance count at two thresholds
  3. print the top facts with dates for eyeball verification — an
     automatic threshold count alone would be exactly the kind of
     unvalidated instrument that has burned this project 14 times

VERDICT RULE (fixed before running): the lever proceeds IFF the corpus
plausibly reaches the gold count in >= 8 of the counting questions.
Below that, extraction coverage is the real blocker and the next work is
write-time (extraction), not read-time (aggregation).
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

CORPUS = HERE / "extracted_memories" / "gate_c_facts.db"
RUNS = ["qa_accuracy_longmemeval_dense.json",
        "qa_accuracy_longmemeval_dense_repeat.json",
        "qa_accuracy_longmemeval_dense_run3.json"]


def main():
    import numpy as np
    from corpus_loaders import load_longmemeval
    import answer_presence as ap

    R = [{r["question"].strip(): bool(r["correct"])
          for r in json.load(open(HERE / f))["results"]} for f in RUNS]
    qs = set(R[0]) & set(R[1]) & set(R[2])
    sysfail = {x for x in qs if not any(r[x] for r in R)}

    ds = load_longmemeval(n_queries=150, seed=42, split="s")
    model = ap._get_model()

    con = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
    counting = []
    for q in ds.queries:
        qt = q.question.strip()
        if qt not in sysfail:
            continue
        gold = str(q.gold_answer)
        nums = ap.numbers(gold)
        if not re.search(r"\bhow (many|much|old|often)\b", q.question.lower()) \
                or not nums:
            continue
        counting.append((q, int(float(sorted(nums)[0]))))
    print(f"counting questions among the 29 systematic: {len(counting)}\n")

    reach = 0
    for q, gold_n in counting:
        ph = ",".join("?" * len(q.scope_keys))
        rows = con.execute(
            f"SELECT fact_text, t_occurred, source_session_id FROM "
            f"semantic_facts WHERE superseded_by IS NULL AND "
            f"source_session_id IN ({ph})", list(q.scope_keys)).fetchall()
        if not rows:
            print(f"Q: {q.question[:70]}\n   gold={gold_n}  FACTS IN SCOPE: 0\n")
            continue
        texts = [r[0] for r in rows]
        emb = model.encode([f"passage: {t[:300]}" for t in texts],
                           normalize_embeddings=True, batch_size=256,
                           show_progress_bar=False).astype(np.float32)
        qe = model.encode([f"query: {q.question}"],
                          normalize_embeddings=True,
                          show_progress_bar=False).astype(np.float32)[0]
        sims = emb @ qe
        order = np.argsort(sims)[::-1]
        hi = int((sims >= 0.82).sum())
        lo = int((sims >= 0.78).sum())
        ok = lo >= gold_n
        reach += ok
        print(f"Q: {q.question[:74]}")
        print(f"   gold={gold_n}   facts-in-scope={len(rows)}   "
              f"matches@0.82={hi}  @0.78={lo}   "
              f"{'REACHABLE' if ok else 'SHORT'}")
        for i in order[:min(gold_n + 2, 8)]:
            r = rows[int(i)]
            print(f"     {sims[int(i)]:.3f} [{r[1] or 'undated'}] "
                  f"{r[0][:86]}")
        print()

    print(f"{'=' * 66}\nVERDICT: corpus plausibly reaches the gold count in "
          f"{reach}/{len(counting)} counting questions (bar: >=8)")
    print("PROCEED" if reach >= 8 else
          "STOP — extraction coverage is the blocker, not aggregation")


if __name__ == "__main__":
    main()
