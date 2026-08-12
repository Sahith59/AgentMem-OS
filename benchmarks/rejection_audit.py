"""
SUPPORT-GATE REJECTION AUDIT — is the gate throwing away things that matter?

WHY THIS EXISTS
The "no supporting USER turn" rule is the single largest rejection cause in
the corpus (1,001 of 1,628 = 61%). It was added in G3 round 1 because ~21% of
pre-gate output was assistant knowledge (CONSOLIDATION_V2_BUILD_LOG.md:71,
the Mem0 #4573 bloat class). That measured how much junk EXISTED. **Nobody
ever measured what fraction of what it REJECTS is actually junk.**

Run #1 made the cost concrete: single-session-assistant collapsed 17/20 ->
3/20 (54% of a statistically significant 26-question regression,
DECISION_AND_FAILURE_LOG §3.1q).

The numbers gate got this treatment and came out 68.2% correct
(benchmarks/gate_replay.py). This gate has never been checked at all.

TAXONOMY — every rejected fact lands in exactly one bucket:
  A WORLD_KNOWLEDGE       general/world fact the assistant taught, wrongly
                          phrased as a user fact ("The user's ISS orbits at
                          28000 km/h") -> gate CORRECT
  B USER_FACT_VIA_ASSISTANT genuinely about this user's life, plans,
                          preferences or state, but the wording traces to an
                          assistant turn ("The user plans to try the smoothie
                          recipe") -> gate WRONG, real memory destroyed
  C ASSISTANT_CONTENT     what the assistant said/recommended/offered — a
                          legitimate CONVERSATION fact, not a user fact
                          -> out of the current scope by design; needs a
                          second fact class, not a looser gate
  D UNSUPPORTED           not supported by the conversation at all
                          (extraction hallucination) -> gate CORRECT

Judged by an LLM reading the REAL session turns, not by heuristics — the
distinction is semantic and a keyword rule would just encode my guess.
Deliberately gpt-4o-mini and a bounded sample: this is a measurement, not a
product path, and it must stay under a dollar.

$ COST: ~$0.05 for the default 150-fact sample. Prints an estimate and
requires --go before spending.
"""
import json
import os
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

for _p in (HERE.parent, HERE.parent.parent):
    _env = _p / ".env"
    if _env.exists():
        for line in _env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
        break

CORPUS = HERE / "extracted_memories" / "gate_c_facts.db"
CAUSE = "no supporting USER turn"
N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 150
GO = "--go" in sys.argv
MODEL = "gpt-4o-mini"

PROMPT = """You are auditing a memory system's extraction filter.

The system extracts facts about a USER from a conversation, then REJECTS any
fact it cannot trace to something the USER themselves said. This rejected fact
was discarded. Your job is to say whether discarding it was right.

CONVERSATION (U = user, A = assistant):
{transcript}

REJECTED FACT: "{fact}"

Classify into EXACTLY ONE bucket:

A = WORLD_KNOWLEDGE — general/world information the assistant supplied, wrongly
    written as if it were a fact about the user (e.g. "The user's ISS orbits at
    28000 km/h"). Discarding is CORRECT.
B = USER_FACT_VIA_ASSISTANT — genuinely about THIS user's life, plans,
    preferences, possessions or state, and the conversation supports it, but the
    wording traces to an assistant turn (e.g. the assistant proposed a recipe and
    the user agreed to try it). Discarding LOSES REAL USER MEMORY.
C = ASSISTANT_CONTENT — a true record of what the ASSISTANT said, recommended or
    offered. Useful to remember about the conversation, but it is not a fact
    about the user.
D = UNSUPPORTED — the conversation does not support this claim at all.
    Discarding is CORRECT.

Answer with exactly one letter: A, B, C, or D."""


def main():
    con = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT session_id, rejections_json FROM consolidation_log "
        "WHERE rejections_json IS NOT NULL AND rejections_json != '[]'"
    ).fetchall()
    con.close()

    pool = []
    for sid, js in rows:
        try:
            items = json.loads(js)
        except Exception:
            continue
        for text, probs in items:
            probs = probs if isinstance(probs, list) else [probs]
            if any(CAUSE in str(p) for p in probs):
                pool.append((sid, text))
    print(f"'{CAUSE}' rejections in corpus: {len(pool)}")

    from corpus_loaders import load_longmemeval
    ds = load_longmemeval(n_queries=500, seed=42, split="s")
    turns = {m.mid: m.turns for m in ds.memories}
    pool = [(s, t) for s, t in pool if s in turns]
    print(f"replayable (session turns available): {len(pool)}")

    rng = random.Random(42)
    sample = rng.sample(pool, min(N, len(pool)))
    est = len(sample) * 1500 / 1e6 * 0.15
    print(f"sampling {len(sample)} | est. cost ~${est:.3f} on {MODEL}")
    if not GO:
        print("\nDRY RUN — pass --go to spend.")
        return

    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Per-fact labels are PERSISTED so candidate gate rules can be tested
    # against ground truth OFFLINE and for free, as many times as needed.
    # Without this, every rule idea costs another paid pass and gets
    # evaluated by eyeball — which is how the last six hypotheses died.
    labelled = []
    counts, examples = Counter(), {}
    for i, (sid, fact) in enumerate(sample, 1):
        tr = "\n".join(
            f"{'U' if t.get('role') == 'user' else 'A'}: "
            f"{t.get('content', '')[:400]}" for t in turns[sid][:24])[:6000]
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=3,
                messages=[{"role": "user", "content": PROMPT.format(
                    transcript=tr, fact=fact)}])
            v = (r.choices[0].message.content or "").strip().upper()[:1]
        except Exception as e:
            print(f"  [error] {e}")
            continue
        if v not in "ABCD":
            v = "?"
        counts[v] += 1
        labelled.append({"session_id": sid, "fact": fact, "label": v})
        examples.setdefault(v, []).append(fact[:110])
        if i % 25 == 0:
            print(f"  ...{i}/{len(sample)}", flush=True)

    tot = sum(counts.values())
    label = {"A": "WORLD_KNOWLEDGE      (gate CORRECT)",
             "B": "USER_FACT_VIA_ASSISTANT (gate WRONG — memory lost)",
             "C": "ASSISTANT_CONTENT    (out of scope by design)",
             "D": "UNSUPPORTED          (gate CORRECT)",
             "?": "unparsed"}
    print(f"\n{'=' * 66}\nVERDICT on {tot} sampled '{CAUSE}' rejections\n{'=' * 66}")
    for k in "ABCD?":
        if counts[k]:
            print(f"  {k}  {label[k]:44s} {counts[k]:4d} = "
                  f"{counts[k] / tot:5.1%}")
    correct = counts["A"] + counts["D"]
    print(f"\n  gate CORRECT (A+D)            : {correct / tot:.1%}")
    print(f"  REAL USER MEMORY DESTROYED (B): {counts['B'] / tot:.1%}"
          f"  -> ~{round(counts['B'] / tot * len(pool))} of {len(pool)} facts")
    print(f"  conversation facts out of scope (C): {counts['C'] / tot:.1%}"
          f"  -> ~{round(counts['C'] / tot * len(pool))} of {len(pool)} facts")
    for k in "BC":
        if examples.get(k):
            print(f"\n  --- {label[k]} examples ---")
            for e in examples[k][:5]:
                print(f"    {e}")

    out = HERE / "rejection_audit_results.json"
    out.write_text(json.dumps({
        "cause": CAUSE, "model": MODEL, "population": len(pool),
        "sampled": tot, "counts": dict(counts),
        "gate_correct_rate": round(correct / tot, 4),
        "user_memory_destroyed_rate": round(counts["B"] / tot, 4),
        "assistant_content_rate": round(counts["C"] / tot, 4),
        "labelled": labelled,
    }, indent=2))
    print(f"\nartifact -> {out}")


if __name__ == "__main__":
    main()
