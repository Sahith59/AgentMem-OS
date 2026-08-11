"""
*** DEPRECATED 2026-08-10 — DO NOT USE. Superseded by fidelity_ladder.py. ***

An independent review established that this instrument's rungs vary FOUR
things at once, so its causal attribution ("-11.4 pts = our extraction
prompt") is wrong BY CONSTRUCTION:
  1. Rung 2 is a Claude-Haiku cache (corpus_loaders.py:66-69), NOT our
     prompt — rung 2->3 changes model AND prompt AND pipeline together.
  2. Rung 2 carries [date] prefixes; rung 3 reads bare fact_text, while
     the product renders dates separately (fact_retrieval.py:344-354).
  3. Rung 3 filters `superseded_by IS NULL`, marking knowledge-update
     gold answers that ARE the older value as lost by design.
  4. Rung 3 groups by primary source_session_id, so re-affirmed facts are
     invisible on a system whose thesis is re-affirmation.
The docstring below claims "all three share the same bias". That is FALSE
and is the specific error this header exists to stop being repeated.

KEPT, not deleted, because DECISION_AND_FAILURE_LOG §3.1c cites its
numbers and history must stay auditable. See §3.1f for the corrections.

EXTRACTION FIDELITY LADDER ($0, no LLM calls).

The question N8 forced (DECISION_AND_FAILURE_LOG §3.1a): replacing raw
turns with our extracted facts cost 43.8 accuracy points. WHERE does the
answer-bearing detail disappear?

Three representations of the SAME gold session, checked for whether the
gold answer is still recoverable:

  1. RAW TURNS        — the ceiling. If the answer isn't here, the
                        dataset never contained it and nothing downstream
                        can be blamed.
  2. CACHED FACTS     — benchmark_cache extraction (what N8 actually
                        ingested, via facts_as_turns()).
  3. LIVE PIPELINE    — benchmarks/extracted_memories/gate_c_facts.db,
                        the local llama3.1 run through the REAL
                        consolidation_v2 engine (prompt + validator +
                        support gate + storage).

Reading the ladder:
  raw high, cached low   -> the EXTRACTION PROMPT drops the detail
  cached high, live low  -> the VALIDATOR / SUPPORT GATE / storage drops it
  all three high         -> extraction is fine and the loss is in
                            RETRIEVAL or budget, not representation

Built through the SAME code path the eval uses (attach_extracted_memories
+ facts_as_turns) — F-16: a diagnostic that constructs its own expectation
measures its own bug, not the system's.

METHOD LIMIT, stated up front: presence is checked lexically (numbers must
appear; otherwise >=60% of the gold answer's content words must appear).
It cannot see a correct paraphrase that shares no tokens, so every number
here is a LOWER BOUND on fidelity. It is used only to COMPARE the three
representations, which share the same bias.
"""
import re
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

CORPUS = HERE / "extracted_memories" / "gate_c_facts.db"

_STOP = {
    "the", "and", "for", "with", "that", "this", "you", "your", "was", "were",
    "have", "has", "had", "are", "not", "but", "from", "they", "their", "them",
    "when", "what", "which", "would", "could", "should", "about", "into",
    "user", "users", "there", "then", "than", "some", "just", "also", "been",
    "will", "can", "may", "his", "her", "its", "our", "out", "one", "all",
}


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP}


def _numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def recoverable(gold: str, haystack: str) -> bool:
    """Is the gold answer still present in this representation?"""
    if not haystack:
        return False
    hay_low = haystack.lower()
    g_nums = _numbers(gold)
    if g_nums:
        # A numeric answer is only recoverable if the NUMBER survives —
        # this is the exact failure N8 exposed ("20 playlists" -> "talked
        # about playlists").
        return all(n in _numbers(haystack) for n in g_nums)
    gw = _words(gold)
    if not gw:
        return gold.strip().lower() in hay_low
    hw = _words(hay_low)
    return len(gw & hw) / len(gw) >= 0.60


def main():
    from corpus_loaders import (load_longmemeval, facts_as_turns,
                                attach_extracted_memories)

    ds = load_longmemeval(n_queries=150, seed=42, split="s")
    hits = attach_extracted_memories(ds)
    print(f"extraction cache attached to {hits} sessions\n")
    mem = {m.mid: m for m in ds.memories}

    live = {}
    if CORPUS.exists():
        con = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
        try:
            for sid, txt in con.execute(
                    "SELECT source_session_id, GROUP_CONCAT(fact_text, ' ') "
                    "FROM semantic_facts WHERE superseded_by IS NULL "
                    "GROUP BY source_session_id"):
                live[sid] = txt or ""
        finally:
            con.close()
    print(f"live-pipeline corpus: {len(live)} sessions with facts\n")

    rows = []
    for q in ds.queries:
        raw = cached = livetxt = ""
        has_cache = has_live = False
        for k in q.gold_keys:
            m = mem.get(k)
            if not m:
                continue
            raw += " " + " ".join(t.get("content", "") for t in m.turns)
            if m.facts:
                has_cache = True
                cached += " " + " ".join(t["content"]
                                         for t in facts_as_turns(m))
            if k in live:
                has_live = True
                livetxt += " " + live[k]
        rows.append({
            "type": q.question_type,
            "raw": recoverable(q.gold_answer, raw),
            "cached": recoverable(q.gold_answer, cached) if has_cache else None,
            "live": recoverable(q.gold_answer, livetxt) if has_live else None,
            "numeric": bool(_numbers(q.gold_answer)),
        })

    def rate(key, subset=None):
        sel = [r for r in (subset or rows) if r[key] is not None]
        if not sel:
            return "n/a", 0
        n = sum(1 for r in sel if r[key])
        return f"{n}/{len(sel)} = {n / len(sel):.1%}", len(sel)

    print("=" * 62)
    print("FIDELITY LADDER — is the gold answer still recoverable?")
    print("=" * 62)
    for key, label in (("raw", "1. RAW TURNS      (ceiling)"),
                       ("cached", "2. CACHED FACTS   (what N8 ingested)"),
                       ("live", "3. LIVE PIPELINE  (gate_c corpus)")):
        r, n = rate(key)
        print(f"  {label:36s} {r:>18s}   (n={n})")

    print("\nBY QUESTION TYPE (raw / cached / live):")
    for t in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == t]
        print(f"  {t:28s} {rate('raw', sub)[0]:>16s} | "
              f"{rate('cached', sub)[0]:>16s} | {rate('live', sub)[0]:>16s}")

    num = [r for r in rows if r["numeric"]]
    non = [r for r in rows if not r["numeric"]]
    print(f"\nNUMERIC answers (n={len(num)}) — the failure N8 showed:")
    print(f"  raw {rate('raw', num)[0]} | cached {rate('cached', num)[0]} "
          f"| live {rate('live', num)[0]}")
    print(f"NON-NUMERIC answers (n={len(non)}):")
    print(f"  raw {rate('raw', non)[0]} | cached {rate('cached', non)[0]} "
          f"| live {rate('live', non)[0]}")

    lost = [r for r in rows if r["raw"] and r["cached"] is False]
    print(f"\nLOST BY EXTRACTION (present in raw, gone in cached): "
          f"{len(lost)} questions")


if __name__ == "__main__":
    main()
