"""Session-level Recall@k for the production retrieval stack. $0.

For each of the 500 LongMemEval _s questions: index the question's
haystack turns with the SAME retriever the product ships
(MultiVectorRetriever: multilingual-e5-small dense + TF-IDF, RRF
fusion), rank turns for the question, and derive a ranked SESSION list
(sessions ordered by their best-ranked turn). Report, at k in
{1,5,10,15}:
  - ANY-gold Recall@k: >=1 gold session in the top k (the metric most
    vendors publish);
  - ALL-gold Recall@k: EVERY gold session in the top k (what multi-hop
    questions actually require).
Both published together because the gap between them IS the story
(see docs/BENCHMARKS.md, the coverage finding).

Snippet/packing config is irrelevant here (ranking only). Output:
benchmarks/recall_at_k_results.json
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))

from corpus_loaders import load_longmemeval  # noqa: E402

KS = (1, 5, 10, 15)


def main():
    from llm.multi_vector_retrieval import MultiVectorRetriever

    ds = load_longmemeval(n_queries=500, seed=42, split="s")
    mem_by_id = {m.mid: m for m in ds.memories}
    items = [q for q in ds.queries if q.gold_answer and q.scope_keys]
    print(f"{len(items)} questions", flush=True)

    any_hit = {k: 0 for k in KS}
    all_hit = {k: 0 for k in KS}
    per_q, t0 = [], time.time()

    for i, q in enumerate(items):
        turns, owner = [], []
        for mkey in q.scope_keys:
            mem = mem_by_id.get(mkey)
            if not mem:
                continue
            for t in mem.turns:
                c = t.get("content", "")
                if c and c.strip():
                    turns.append(c)
                    owner.append(mkey)
        r = MultiVectorRetriever(context_turns=0, snippet_chars=0)
        r.index(turns)
        # rank raw turn indices via the retriever's fused ordering:
        # search returns turn TEXTS in rank order; map back by identity
        ranked_texts = r.search(q.question, top_k=len(turns))
        seen_sessions, ranked_sessions = set(), []
        text_to_idx = {}
        for idx, t in enumerate(turns):
            text_to_idx.setdefault(t, []).append(idx)
        for t in ranked_texts:
            idxs = text_to_idx.get(t.split("\n")[0]) or text_to_idx.get(t)
            if not idxs:
                continue
            s = owner[idxs[0]]
            if s not in seen_sessions:
                seen_sessions.add(s)
                ranked_sessions.append(s)
        golds = set(g for g in q.gold_keys if g in mem_by_id)
        if not golds:
            continue
        row = {"question": q.question[:60], "type": q.question_type,
               "n_gold": len(golds)}
        for k in KS:
            top = set(ranked_sessions[:k])
            a = bool(golds & top)
            al = golds <= top
            any_hit[k] += a
            all_hit[k] += al
            row[f"any@{k}"] = a
            row[f"all@{k}"] = al
        per_q.append(row)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(items)} | {(time.time()-t0)/60:.1f} min",
                  flush=True)

    n = len(per_q)
    out = {
        "n_questions": n,
        "retriever": "MultiVectorRetriever (multilingual-e5-small dense "
                     "+ TF-IDF, RRF k=60), session ranked by best turn",
        "any_gold_recall": {f"@{k}": round(any_hit[k] / n, 4) for k in KS},
        "all_gold_recall": {f"@{k}": round(all_hit[k] / n, 4) for k in KS},
        "per_question": per_q,
    }
    path = HERE / "recall_at_k_results.json"
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print("\nANY-gold:", out["any_gold_recall"])
    print("ALL-gold:", out["all_gold_recall"])
    print("wrote", path)


if __name__ == "__main__":
    main()
