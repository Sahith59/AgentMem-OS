"""
SUPPORT-GATE RULE SEARCH ($0) — derive the replacement rule from LABELLED
DATA, using the PRODUCTION tokenizer, before touching the gate.

INPUT: benchmarks/rejection_audit_results.json — 250 facts the live gate
rejected for "no supporting USER turn", each labelled by an LLM that read the
real session:
    A world knowledge mis-attributed to the user  -> must stay REJECTED
    B genuine user fact, assistant-worded          -> must be STORED (~466
                                                      facts destroyed today)
    C assistant content                            -> separate class, not here
    D unsupported / hallucinated                   -> must stay REJECTED

TARGET: a rule that ACCEPTS B while still rejecting A and D. Scored as:
    recall(B)      = share of destroyed user memory recovered
    precision      = share of accepted facts that are genuinely B
    junk admitted  = A or D facts the rule would now store  <- the real risk

Every candidate is evaluated with `consolidation_v2._tokens` and the same
`need` threshold the gate uses, so a rule that wins here is implementable
verbatim — no reimplementation gap between the experiment and the fix.

NOTE ON HONESTY: the labels come from gpt-4o-mini, not a human. They are good
enough to RANK candidate rules and to size an effect; they are not ground
truth in the strong sense. Any rule chosen here still has to prove itself
end-to-end on the benchmark.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

AUDIT = HERE / "rejection_audit_results.json"


def main():
    from agentmem_os.llm.consolidation_v2 import _tokens, _quantity_values
    from corpus_loaders import load_longmemeval

    data = json.loads(AUDIT.read_text())
    rows = data["labelled"]
    print(f"labelled facts: {len(rows)}  "
          f"({sum(1 for r in rows if r['label'] == 'B')} are B)")

    ds = load_longmemeval(n_queries=500, seed=42, split="s")
    turns = {m.mid: m.turns for m in ds.memories}

    feats = []
    for r in rows:
        t = turns.get(r["session_id"])
        if not t:
            continue
        ftoks = _tokens(r["fact"])
        need = 1 if len(ftoks) <= 4 else 2
        user_tx = [x.get("content", "") for x in t
                   if x.get("role") == "user"]
        asst_tx = [x.get("content", "") for x in t
                   if x.get("role") != "user"]
        utoks = set()
        for c in user_tx:
            utoks |= _tokens(c)
        atoks = set()
        for c in asst_tx:
            atoks |= _tokens(c)
        # best per-TURN overlap, the shape the gate already computes
        best_user = max((len(ftoks & _tokens(c)) for c in user_tx), default=0)
        best_asst = max((len(ftoks & _tokens(c)) for c in asst_tx), default=0)
        fvals, _ = _quantity_values(r["fact"])
        uvals = set()
        for c in user_tx:
            v, _ = _quantity_values(c, strip_stamps=True)
            uvals |= v
        feats.append({
            "label": r["label"], "fact": r["fact"],
            "n_ftoks": len(ftoks), "need": need,
            "best_user": best_user, "best_asst": best_asst,
            # POOLED user overlap: does the fact's vocabulary appear in the
            # user's own speech ANYWHERE in the session, ignoring the
            # per-turn bar? This is the signal the current gate cannot see.
            "pool_user": len(ftoks & utoks),
            "pool_user_frac": len(ftoks & utoks) / max(1, len(ftoks)),
            "pool_asst": len(ftoks & atoks),
            "nums_unlicensed": bool(fvals - uvals),
        })

    B = [f for f in feats if f["label"] == "B"]
    AD = [f for f in feats if f["label"] in ("A", "D")]
    C = [f for f in feats if f["label"] == "C"]
    print(f"usable: B={len(B)}  A+D={len(AD)}  C={len(C)}\n")

    print("FEATURE SEPARATION (mean values):")
    for k in ("best_user", "pool_user", "pool_user_frac", "best_asst",
              "n_ftoks"):
        b = sum(f[k] for f in B) / max(1, len(B))
        a = sum(f[k] for f in AD) / max(1, len(AD))
        print(f"  {k:16s} B={b:6.2f}   A+D={a:6.2f}   sep={b - a:+6.2f}")

    def score(name, rule):
        tp = sum(1 for f in B if rule(f))
        fp = sum(1 for f in AD if rule(f))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, len(B))
        print(f"  {name:44s} recall(B)={rec:5.1%}  prec={prec:5.1%}  "
              f"junk admitted={fp:3d}/{len(AD)}")
        return rec, prec, fp

    print("\nCANDIDATE RULES (accept the fact if the rule is TRUE):")
    print("  --- baselines ---")
    score("accept everything (today's rejections)", lambda f: True)
    print("  --- pooled user-vocabulary grounding ---")
    for thr in (1, 2, 3, 4):
        score(f"pool_user >= {thr}", lambda f, t=thr: f["pool_user"] >= t)
    for frac in (0.30, 0.40, 0.50, 0.60):
        score(f"pool_user_frac >= {frac:.2f}",
              lambda f, t=frac: f["pool_user_frac"] >= t)
    print("  --- combined with numeric safety ---")
    for frac in (0.40, 0.50):
        score(f"pool_user_frac >= {frac:.2f} AND numbers licensed",
              lambda f, t=frac: f["pool_user_frac"] >= t
              and not f["nums_unlicensed"])
    print("  --- require SOME assistant support (excludes D-style junk) ---")
    for frac in (0.30, 0.40, 0.50):
        score(f"pool_user_frac >= {frac:.2f} AND best_asst >= need",
              lambda f, t=frac: f["pool_user_frac"] >= t
              and f["best_asst"] >= f["need"])


if __name__ == "__main__":
    main()
