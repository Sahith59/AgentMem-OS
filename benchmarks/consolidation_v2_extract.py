"""
Consolidation v2 — Gate C extraction: distill EVERY session in the slice
questions' haystacks (noise included — no oracle shortcut) into dated atomic
facts with a local model. $0. See CONSOLIDATION_V2_DESIGN.md §6.

Parallelized (worker threads against Ollama) and checkpointed (JSONL append,
restart-safe) — the Graphiti lesson: never serial, never checkpoint-free.

Usage:
    python3 benchmarks/consolidation_v2_extract.py [model] [workers]

Sessions come from the union of scope_keys of the 79 questions actually run
in the slice artifact — exactly the haystacks the Gate C eval will query.
Resume: sessions already in the output JSONL are skipped, so kill/restart is
always safe.
"""

import json
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:14b"
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT = HERE / "extracted_memories" / f"facts_local_slice.jsonl"

from consolidation_v2_gate_b import EXTRACTION_PROMPT  # single prompt source


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
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["response"]


def main():
    ds = json.load(open(HERE / "benchmark_cache/longmemeval_s.json"))
    slice_qs = {r["question"] for r in json.load(
        open(HERE / "qa_accuracy_longmemeval_answerer54mini.json"))["results"]}
    qmap = {q["question"]: q for q in ds["queries"]}
    mems = {m["mid"]: m for m in ds["memories"]}

    sids = []
    seen = set()
    for qtext in slice_qs:
        for k in qmap[qtext]["scope_keys"]:
            if k not in seen and k in mems:
                seen.add(k)
                sids.append(k)

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                done.add(json.loads(line)["session_id"])
            except Exception:
                pass
    todo = [s for s in sids if s not in done]
    print(f"Gate C extraction: {MODEL}, {WORKERS} workers | "
          f"{len(sids)} sessions total, {len(done)} done, {len(todo)} to go", flush=True)

    # Refuse to start against a dead server — a dead Ollama once turned this
    # run into 3,600 instant connection-refused "completions".
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
    except Exception as e:
        sys.exit(f"ABORT: Ollama not reachable ({e}). Start it, then relaunch.")

    date_re = re.compile(r"Session dated ([0-9/]+)")
    lock = threading.Lock()
    t0 = time.time()
    n_done = [0]
    consecutive_errors = [0]

    def work(sid):
        content = mems[sid]["content"]
        m = date_re.search(content)
        date = m.group(1) if m else "unknown"
        try:
            raw = ollama_generate(EXTRACTION_PROMPT.format(date=date, transcript=content[:36000]))
        except Exception as e:
            # Do NOT checkpoint failures — the session must retry on relaunch.
            with lock:
                consecutive_errors[0] += 1
                print(f"  [error] {sid}: {e}", flush=True)
                if consecutive_errors[0] >= 15:
                    print("ABORT: 15 consecutive errors — server is down.", flush=True)
                    import os
                    os._exit(2)
            return
        facts = [ln[len("FACT: "):].strip() for ln in raw.splitlines()
                 if ln.strip().startswith("FACT:")]
        rec = {"session_id": sid, "session_date": date, "facts": facts,
               "extraction_model": MODEL}
        with lock:
            consecutive_errors[0] = 0
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n_done[0] += 1
            if n_done[0] % 25 == 0:
                rate = n_done[0] / (time.time() - t0)
                eta_h = (len(todo) - n_done[0]) / max(rate, 1e-9) / 3600
                print(f"  {n_done[0]}/{len(todo)} | {rate*3600:.0f}/hr | ETA {eta_h:.1f}h", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, todo))

    print(f"DONE: {len(todo)} sessions in {(time.time()-t0)/3600:.2f}h -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
