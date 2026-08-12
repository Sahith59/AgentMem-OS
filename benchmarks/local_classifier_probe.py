"""
CAN THE LOCAL MODEL DECLARE PROVENANCE? ($0, local llama3.1)

THE DESIGN THIS DECIDES
Every lexical gate has now failed to separate a genuine user fact from
world knowledge or a hallucination (benchmarks/gate_rule_search.py: best
precision 34% against a 25.5% base rate; and D-labelled hallucinations
carry assistant-turn support 83% of the time — indistinguishable from
real facts). The distinction is semantic.

So the fix must follow the pattern that already works in this codebase
(supersession, llm/supersession.py): **the LLM proposes, deterministic
gates decide.** The extractor knows which turn it drew a fact from; the
gate does not. Have the extractor DECLARE it.

That whole design rests on one assumption nobody has tested:
**can llama3.1 — the model that does our extraction — actually make this
call?** If it cannot, the design is dead and no amount of plumbing saves
it. This probe answers that BEFORE any prompt change, schema migration,
or 3-hour cluster re-extraction.

METHOD
Replay the 250 audit-labelled rejected facts (labels from gpt-4o-mini
reading the real session) through llama3.1 with the same taxonomy, and
measure agreement. Reported as:
  * agreement on the decision that matters: B (store) vs A/D (reject)
  * per-class recall
  * the confusion that would hurt most: D judged B (hallucination stored)

HONEST FRAMING: this measures AGREEMENT WITH gpt-4o-mini, not truth.
gpt-4o-mini is itself a judge, not ground truth. High agreement means the
local model is not the bottleneck; it does not prove either is right.
"""
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "gemma4:26b"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120

PROMPT = """You extract memories from a conversation between a USER and an ASSISTANT.

CONVERSATION (U = user, A = assistant):
{transcript}

CANDIDATE MEMORY: "{fact}"

Which of these is it? Answer with ONE letter only.

A = general/world information the assistant supplied, written as if it were a fact about the user. Not a fact about this user's life.
B = genuinely about THIS user's life, plans, preferences, possessions or state, and the conversation supports it — even if the assistant was the one who put it into words.
C = a record of what the ASSISTANT said, recommended, offered or produced. True about the conversation, but not a fact about the user.
D = the conversation does not support this claim at all.

Answer:"""


def ask(prompt):
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_predict": 4},
    }).encode()
    req = urllib.request.Request(
        OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["response"]


def main():
    from corpus_loaders import load_longmemeval
    rows = json.loads((HERE / "rejection_audit_results.json").read_text())
    rows = rows["labelled"][:N]
    ds = load_longmemeval(n_queries=500, seed=42, split="s")
    turns = {m.mid: m.turns for m in ds.memories}

    pairs = []
    for i, r in enumerate(rows, 1):
        t = turns.get(r["session_id"])
        if not t:
            continue
        tr = "\n".join(
            f"{'U' if x.get('role') == 'user' else 'A'}: "
            f"{x.get('content', '')[:400]}" for x in t[:24])[:6000]
        try:
            out = ask(PROMPT.format(transcript=tr, fact=r["fact"]))
        except Exception as e:
            print(f"  [error] {e}")
            continue
        v = "".join(c for c in out.upper() if c in "ABCD")[:1] or "?"
        pairs.append((r["label"], v))
        if i % 20 == 0:
            print(f"  ...{i}/{len(rows)}", flush=True)

    print(f"\n{'=' * 64}\nlocal {MODEL} vs gpt-4o-mini labels (n={len(pairs)})"
          f"\n{'=' * 64}")
    exact = sum(1 for a, b in pairs if a == b)
    print(f"  exact 4-way agreement : {exact}/{len(pairs)} = "
          f"{exact / max(1, len(pairs)):.1%}")

    # The decision the gate actually makes.
    store = {"B"}
    tp = sum(1 for a, b in pairs if a in store and b in store)
    fn = sum(1 for a, b in pairs if a in store and b not in store)
    fp = sum(1 for a, b in pairs if a not in store and b in store)
    tn = sum(1 for a, b in pairs if a not in store and b not in store)
    print(f"\n  DECISION THAT MATTERS — store (B) vs reject (A/C/D):")
    print(f"    correctly stored   (B->B) : {tp}")
    print(f"    missed             (B->x) : {fn}")
    print(f"    wrongly stored     (x->B) : {fp}")
    print(f"    correctly rejected        : {tn}")
    if tp + fp:
        print(f"    precision={tp / (tp + fp):.1%}  "
              f"recall={tp / max(1, tp + fn):.1%}")

    print("\n  CONFUSION (gpt-4o-mini label -> llama3.1 label):")
    c = Counter(pairs)
    for a in "ABCD":
        row = {b: c[(a, b)] for b in "ABCD?" if c[(a, b)]}
        if row:
            print(f"    {a} -> {row}")
    dangerous = c[("D", "B")]
    print(f"\n  MOST DANGEROUS CONFUSION — hallucination stored as user fact "
          f"(D->B): {dangerous}")


if __name__ == "__main__":
    main()
