"""$0 prototype: question-blind cross-session ledgers (F-22 candidate).

For each scope behind the 43 chronic/failed counting questions, a LOCAL
model reads the scope's FACTS (never the questions, never raw sessions)
and authors recurring-topic ledgers: topic, count, dated instances.
Then, per failed question, we check whether a blind ledger exists whose
count equals the gold answer — the packet-level proof (or refutation)
that write-time synthesis would put the correct count in front of the
answerer as one line.

Local ollama llama3.1, $0. Output: benchmarks/ledger_prototype_results.json
"""
import json
import os
import re
import sqlite3
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "extracted_memories" / "gate_c_facts_luna.db"
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
sys.path.insert(0, str(HERE))

PROMPT = """You are a memory consolidation engine building ACTIVITY \
LEDGERS from a user's fact history. Group the dated facts below into \
recurring topics (the same kind of event/activity/item happening or \
being mentioned multiple times). For each topic with 2 or more \
distinct instances, output the topic, the count of DISTINCT instances \
(merge duplicates describing the same event), and the dates.

Rules:
- Count DISTINCT real instances; two facts about the same single event \
count once.
- Only concrete countable things (events attended, items bought, \
sessions done, visits, runs, purchases). Not preferences or traits.
- If unsure whether two facts are the same instance, count them once.

FACTS:
{facts}

Return JSON: {{"ledgers": [{{"topic": ..., "count": N, \
"dates": [...]}}]}}"""


def llm(prompt):
    req = urllib.request.Request(
        f"{OLLAMA}/api/generate",
        data=json.dumps({
            "model": "llama3.1:latest", "prompt": prompt, "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_ctx": 16384,
                        "num_predict": 2000},
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(json.loads(r.read())["response"])


def main():
    autopsy = json.load(open(HERE / "autopsy_luna_p1_87.json"))
    counting = [r for r in autopsy if r["is_counting"]]
    print(f"{len(counting)} counting failures", flush=True)

    from corpus_loaders import load_longmemeval
    ds = load_longmemeval(n_queries=500, seed=42, split="s")
    by_q = {q.question: q for q in ds.queries}
    con = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)

    NUM_RE = re.compile(r"\d+(?:\.\d+)?")
    results = []
    for i, r in enumerate(counting):
        q = by_q.get(r["question"])
        if q is None:
            continue
        rows = []
        for sid in q.scope_keys:
            rows += [f"[{d or 'undated'}] {t}" for t, d in con.execute(
                "SELECT fact_text, t_occurred FROM semantic_facts "
                "WHERE source_session_id=? AND fact_type IN "
                "('event','state')", (sid,)).fetchall()]
        if not rows:
            results.append({"question": r["question"], "skip": "no facts"})
            continue
        # scopes can hold hundreds of facts; ledger in chunks then merge
        ledgers = []
        for start in range(0, len(rows), 120):
            chunk = rows[start:start + 120]
            try:
                out = llm(PROMPT.format(facts="\n".join(chunk)))
                ledgers += out.get("ledgers", [])
            except Exception as e:
                print(f"  chunk err: {e}", flush=True)
        gold_nums = NUM_RE.findall(str(r["gold"]))
        gold_n = float(gold_nums[0]) if gold_nums else None
        counts = [l.get("count") for l in ledgers
                  if isinstance(l.get("count"), (int, float))]
        exact = gold_n is not None and gold_n in counts
        near = gold_n is not None and any(
            abs(c - gold_n) <= 1 for c in counts)
        results.append({
            "question": r["question"], "type": r["type"],
            "gold": r["gold"], "stability": r["stability"],
            "n_scope_facts": len(rows), "n_ledgers": len(ledgers),
            "ledger_topics": [str(l.get("topic"))[:40] for l in ledgers][:12],
            "ledger_counts": counts[:12],
            "gold_count": gold_n,
            "exact_count_in_ledgers": exact,
            "within_1": near,
        })
        print(f"  {i+1}/{len(counting)} ledgers={len(ledgers)} "
              f"exact={exact} within1={near} :: {r['question'][:46]}",
              flush=True)

    ok = sum(1 for x in results if x.get("exact_count_in_ledgers"))
    near = sum(1 for x in results if x.get("within_1"))
    n = sum(1 for x in results if "skip" not in x)
    json.dump(results, open(HERE / "ledger_prototype_results.json", "w"),
              indent=1, ensure_ascii=False)
    print(f"\nVERDICT: exact gold count present in blind ledgers: "
          f"{ok}/{n} | within-1: {near}/{n}")
    print("wrote ledger_prototype_results.json")


if __name__ == "__main__":
    main()
