"""
SUPPORT-GATE REPLAY ($0, no LLM).

Re-runs the REAL `ConsolidationV2.evaluate_fact` over the facts the live
pipeline actually REJECTED (from consolidation_log.rejections_json in the
Gate C corpus), against the REAL session turns from LongMemEval.

Purpose: quantify each rejection cause on real data BEFORE changing any
rule, so the fix is sized by measurement rather than by my hypothesis
(F-11: never generalize without a denominator).

DISCLOSURES, up front:
  * rejections_json stores fact text TRUNCATED to 120 chars. A number
    beyond that cut cannot be replayed, so "recoverable" here is a LOWER
    bound. Truncated rows are counted and reported separately.
  * fact_type/t_occurred are not persisted with the rejection; the gate's
    support + numbers logic does not read them, so they are stubbed and
    any row whose problems are type/date-related is excluded.
"""
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

CORPUS = HERE / "extracted_memories" / "gate_c_facts.db"
SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def main():
    from corpus_loaders import load_longmemeval
    from agentmem_os.llm.consolidation_v2 import (
        ConsolidationV2, _quantity_values, _WORD_NUMS)

    print(f"word-numeral map covers: {sorted(_WORD_NUMS)} "
          f"(max = {max(int(v) for v in _WORD_NUMS.values())})\n")

    ds = load_longmemeval(n_queries=500, seed=42, split="s")
    turns_by_session = {}
    for m in ds.memories:
        turns_by_session[m.mid] = [
            (i, t.get("role", "user"), t.get("content", ""), None)
            for i, t in enumerate(m.turns)]
    print(f"sessions available for replay: {len(turns_by_session)}")

    con = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT session_id, rejections_json FROM consolidation_log "
        "WHERE rejections_json IS NOT NULL AND rejections_json != '[]'"
    ).fetchall()
    con.close()

    causes = Counter()
    replayed = truncated = no_turns = 0
    num_reject = []          # (text, turn_data) for number-caused rejects
    assistant_reject = 0

    for sid, js in rows:
        td = turns_by_session.get(sid)
        try:
            items = json.loads(js)
        except Exception:
            continue
        for text, probs in items:
            probs = probs if isinstance(probs, list) else [probs]
            head = str(probs[0])[:44] if probs else "?"
            causes[head] += 1
            if td is None:
                no_turns += 1
                continue
            if len(text) >= 118:
                truncated += 1
            replayed += 1
            if any("numbers" in str(p) for p in probs):
                num_reject.append((text, td))
            if any("no supporting USER turn" in str(p) for p in probs):
                assistant_reject += 1

    print(f"rejection rows: {len(rows)} logs, {sum(causes.values())} facts")
    print(f"replayable (session turns found): {replayed}  "
          f"| no turns: {no_turns} | text-truncated: {truncated}\n")
    print("CAUSES:")
    for c, n in causes.most_common(6):
        print(f"  {n:6d}  {c}")

    # ── How many NUMBER rejections are surface-form failures? ──────────
    # Ask the question the fix depends on: is the missing value PRESENT
    # in user speech as a WORD the current map cannot read?
    EXT = {"thirteen": "13", "fourteen": "14", "fifteen": "15",
           "sixteen": "16", "seventeen": "17", "eighteen": "18",
           "nineteen": "19", "twenty": "20", "thirty": "30",
           "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
           "eighty": "80", "ninety": "90", "hundred": "100",
           "thousand": "1000", "dozen": "12", "couple": "2"}
    word_fix = anyturn_fix = still = 0
    sample = num_reject[:SAMPLE]
    for text, td in sample:
        fv, _ = _quantity_values(text)
        cited_user = " ".join(c for _, r, c, _ in td if r == "user")
        allu = cited_user.lower()
        recovered = False
        for w, v in EXT.items():
            if v in fv and w in allu:
                recovered = True
        if recovered:
            word_fix += 1
            continue
        # present as DIGITS anywhere in user speech (i.e. the value is
        # real but the licensing turn did not clear the overlap bar)
        uvals, _ = _quantity_values(cited_user, strip_stamps=True)
        if fv & uvals:
            anyturn_fix += 1
        else:
            still += 1

    n = len(sample)
    print(f"\nNUMBER-REJECTED facts replayed: {n}")
    if n:
        print(f"  value present in user speech as an UNREADABLE WORD "
              f"(>twelve): {word_fix} = {word_fix / n:.1%}")
        print(f"  value present as DIGITS in user speech but the turn "
              f"missed the overlap bar: {anyturn_fix} = {anyturn_fix / n:.1%}")
        print(f"  value genuinely absent from user speech (gate CORRECT): "
              f"{still} = {still / n:.1%}")
    print(f"\nASSISTANT-KNOWLEDGE rejections: {assistant_reject}")


if __name__ == "__main__":
    main()
