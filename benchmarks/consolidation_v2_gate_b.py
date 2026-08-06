"""
Consolidation v2 — Gate B: can a LOCAL model extract facts well enough?

$0 quality check before any paid extraction (see CONSOLIDATION_V2_DESIGN.md §6).
Takes the evidence sessions behind the hard-core aggregation failures, extracts
atomic facts with a local Ollama model using the v2 draft prompt, and scores the
output two ways against ground truth we already trust:

  1. number-preservation — the counts/quantities/dates the paid haiku extraction
     kept (the thing generic summarization measurably drops)
  2. gold-token assemblability — same containment check Gate A ran on haiku facts

Stop rule: if the local model misses >30% of the countable events the haiku
extraction kept, local extraction is not good enough and Gate C needs API money.
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:14b"
MAX_SESSIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 16

EXTRACTION_PROMPT = """You are a memory consolidation engine. Extract atomic facts about the USER from this conversation session.

Rules:
- One fact per line, prefixed exactly "FACT: "
- Each fact must be self-contained: name "the user", never bare pronouns
- PRESERVE exactly: counts, quantities, prices, dates, times, schedules, proper names. Never round, merge, or drop a number.
- If the user did something N times, the fact states N. If an event has its own date in the text, the fact includes that date.
- End every line with " [mentioned: {date}]"
- Only facts about the user and their life. Skip assistant knowledge, generic advice, pleasantries.
- If the session contains no user facts, output exactly "NONE".

Session date: {date}
Transcript:
{transcript}"""


def ollama_generate(prompt: str) -> str:
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 10240, "num_predict": 1200},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"]


def main():
    ds = json.load(open(HERE / "benchmark_cache/longmemeval_s.json"))
    haiku = json.load(open(HERE / "extracted_memories/rich_memories_haiku_LongMemEval.json"))
    slice_res = json.load(open(HERE / "qa_accuracy_longmemeval_answerer54mini.json"))["results"]
    old = json.load(open(HERE / "qa_accuracy_longmemeval.json"))["results"]
    oldmap = {r["question"]: r for r in old
              if r.get("question_type") in ("temporal-reasoning", "multi-session")}
    hardcore = [r for r in slice_res if r["question"] in oldmap
                and not r["correct"] and not oldmap[r["question"]]["correct"]]

    qmap = {q["question"]: q for q in ds["queries"]}
    mems = {m["mid"]: m for m in ds["memories"]}

    # Aggregation hard-core first (numeric golds) — the measured failure mode
    agg = re.compile(r"how (many|much|often)|total|average", re.I)
    picked_q, sessions = [], []
    for r in hardcore:
        if not agg.search(r["question"]):
            continue
        q = qmap[r["question"]]
        new = [k for k in q["gold_keys"] if k in haiku and k not in sessions]
        if len(sessions) + len(new) > MAX_SESSIONS:
            continue
        picked_q.append(r["question"])
        sessions.extend(new)
        if len(sessions) >= MAX_SESSIONS:
            break

    print(f"Gate B: {MODEL} | {len(sessions)} sessions from {len(picked_q)} hard-core aggregation questions", flush=True)

    date_re = re.compile(r"Session dated ([0-9/]+)")
    out = {}
    for i, sid in enumerate(sessions):
        content = mems[sid]["content"]
        m = date_re.search(content)
        date = m.group(1) if m else "unknown"
        raw = ollama_generate(EXTRACTION_PROMPT.format(date=date, transcript=content[:36000]))
        facts = [ln[len("FACT: "):].strip() for ln in raw.splitlines() if ln.strip().startswith("FACT:")]
        out[sid] = {"session_date": date, "facts": facts}
        print(f"  {i+1}/{len(sessions)} {sid}: {len(facts)} facts", flush=True)

    (HERE / "gate_b_local_extraction.json").write_text(json.dumps(
        {"model": MODEL, "sessions": out, "questions": picked_q}, indent=1))

    # ── Scoring ──────────────────────────────────────────────────────────────
    def norm(s):
        return re.sub(r"[^a-z0-9 ]", " ", s.lower())

    NUM = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+(?:\.\d+)?)\b")
    STOP = {"the", "a", "an", "i", "my", "of", "to", "in", "and",
            "times", "time", "days", "day", "weeks", "week", "months", "month"}

    kept = missed = 0
    for sid in sessions:
        local_text = norm(" ".join(out[sid]["facts"]))
        for hm in haiku.get(sid, []):
            for num in set(NUM.findall(norm(hm["memory"]))):
                if num in local_text:
                    kept += 1
                else:
                    missed += 1
    print(f"\nnumber-preservation vs haiku: kept {kept}, missed {missed} "
          f"({kept / max(1, kept + missed):.1%})")

    hits = 0
    for qtext in picked_q:
        q = qmap[qtext]
        local_all = norm(" ".join(f for k in q["gold_keys"] for f in out.get(k, {}).get("facts", [])))
        gold = norm(str(next(r["gold_answer"] for r in hardcore if r["question"] == qtext)))
        toks = [t for t in gold.split() if t not in STOP]
        if toks and all(t in local_all for t in toks):
            hits += 1
    print(f"gold-token assemblability (weak proxy, cf. Gate A haiku 17/34): {hits}/{len(picked_q)}")
    print(f"\nVERDICT: {'LOCAL OK' if kept / max(1, kept + missed) >= 0.70 else 'LOCAL INSUFFICIENT — Gate C needs API extraction'}")


if __name__ == "__main__":
    main()
