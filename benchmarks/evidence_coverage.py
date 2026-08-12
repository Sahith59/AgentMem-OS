"""
EVIDENCE-TURN COVERAGE ($0) — a sharper retrieval instrument, and a
validation that it is actually sharper.

WHY THE OLD PROXY WAS NOT ENOUGH
Session-level coverage ("did ANY turn from the gold session arrive?")
found the mechanism behind the 72.0% run — ALL gold sessions present ->
84.5% correct, partial/none -> ~44% (§3.1z). But it then MISLED by 32
questions: it predicted dense retrieval would reach 140/150 ALL and the
real assembler delivered 108, while accuracy nonetheless ROSE 110 -> 114.
The reason is structural: dense retrieves BETTER TURNS FROM FEWER
SESSIONS; lexical retrieves SOME turn from more sessions. For answering,
WHICH turn arrives matters more than how many sessions were touched.

WHAT THIS MEASURES INSTEAD
An EVIDENCE TURN is a turn inside a gold session from which the gold
answer is actually recoverable, using the CALIBRATED detector
(answer_presence: lexical, 96.4% balanced accuracy on a 42/42 ground-truth
set, beat embeddings on its own validation). Coverage = what fraction of a
question's evidence turns reach the assembled context.

TWO HONEST LIMITATIONS, HANDLED RATHER THAN HIDDEN
1. DERIVED answers (counting, ordering, durations) appear in NO single
   turn — "how many festivals" is not written anywhere. Such questions
   have ZERO evidence turns by construction. They are counted and
   REPORTED SEPARATELY, never silently scored as 0% coverage, and the
   session-level metric remains the right one for them.
2. The numeric rule can false-positive: a gold answer of "2" matches any
   turn containing a 2. Numeric-answer questions are flagged so the
   split is visible.

VALIDATION — the point of this file
A new proxy is worthless unless it PREDICTS BETTER than the old one. Both
are scored against the same outcomes (the dense 76.0% run) and compared on
separation: the accuracy gap between full-coverage and not-full-coverage
questions. Bigger separation = sharper instrument. If evidence-turn
coverage does not separate better than session coverage, it is discarded
and this file says so.
"""
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

EVAL_DB = (HERE / "eval_dbs" / "longmemeval-s-raw-4b4cc846c7.db")
RUN = HERE / "qa_accuracy_longmemeval_dense.json"


def main():
    os.environ["AGENTMEM_OS_DB_PATH"] = str(EVAL_DB)
    os.environ["AGENTMEM_OS_DISABLE_REDIS"] = "1"
    from corpus_loaders import load_longmemeval
    from agentmem_os.llm.context_assembler import ContextAssembler
    from real_code_utils import install_dense_chroma
    import gate_c_facts_source as gc
    import gate_d_profile_source as gd
    import answer_presence as ap

    def recoverable(gold, hay):
        n = ap.numbers(gold)
        return all(x in ap.numbers(hay) for x in n) if n else ap._lex(gold, hay)

    ds = load_longmemeval(n_queries=150, seed=42, split="s")
    mem = {m.mid: m for m in ds.memories}
    backend = install_dense_chroma(ContextAssembler)
    A = ContextAssembler()
    A.allocations["semantic"] = int(24000 * 0.79 // 4)
    scope = {q.question: list(q.scope_keys) for q in ds.queries}
    gc.install(A, scope)
    gd.install(A, scope)
    res = {r["question"].strip(): bool(r["correct"])
           for r in json.load(open(RUN))["results"]}
    print(f"backend={backend}  run={RUN.name}\n")

    rows = []
    for q in ds.queries:
        qt = q.question.strip()
        if qt not in res:
            continue
        gold = str(q.gold_answer)
        # Evidence turns: turns inside GOLD sessions that actually carry
        # the answer. Derived, because the dataset annotates sessions only.
        ev = []
        for k in q.gold_keys:
            m = mem.get(k)
            if not m:
                continue
            for t in m.turns:
                c = t.get("content", "")
                if c and recoverable(gold, c):
                    ev.append(c)
        sid = ("longmemeval-scope-"
               + hashlib.sha1("|".join(sorted(q.scope_keys)).encode()
                              ).hexdigest()[:12])
        A.profile_session_ids = list(q.scope_keys)
        A._profile.current_question = q.question
        ctx = A.assemble(sid, q.question, agent_id=sid)

        ev_hit = sum(1 for c in ev if c[:60] in ctx)
        s_tot = s_hit = 0
        for k in q.gold_keys:
            if k not in mem:
                continue
            s_tot += 1
            if any(t.get("content", "")[:60] in ctx
                   for t in mem[k].turns if t.get("content")):
                s_hit += 1
        rows.append({
            "type": q.question_type, "correct": res[qt],
            "n_ev": len(ev), "ev_hit": ev_hit,
            "sess_all": bool(s_tot and s_hit == s_tot),
            "ev_all": bool(ev and ev_hit == len(ev)),
            "numeric": bool(ap.numbers(gold)),
        })

    derived = [r for r in rows if r["n_ev"] == 0]
    direct = [r for r in rows if r["n_ev"] > 0]
    print(f"questions with >=1 evidence turn (DIRECT) : {len(direct)}")
    print(f"questions with NO evidence turn (DERIVED) : {len(derived)}"
          f"  <- answer appears in no single turn; session metric applies")
    print(f"   of the derived, numeric answers: "
          f"{sum(1 for r in derived if r['numeric'])}\n")

    def sep(rows_, key, label):
        yes = [r for r in rows_ if r[key]]
        no = [r for r in rows_ if not r[key]]
        if not yes or not no:
            print(f"  {label:34s} (degenerate split)")
            return None
        ay = sum(r["correct"] for r in yes) / len(yes)
        an = sum(r["correct"] for r in no) / len(no)
        print(f"  {label:34s} full={ay:6.1%} (n={len(yes):3d})   "
              f"not-full={an:6.1%} (n={len(no):3d})   "
              f"SEPARATION={ay - an:+6.1%}")
        return ay - an

    print("PROXY VALIDATION — which instrument separates correct from "
          "incorrect better?")
    print("\n ON ALL QUESTIONS:")
    s1 = sep(rows, "sess_all", "session-level coverage (old)")
    print("\n ON DIRECT QUESTIONS ONLY (where evidence turns exist):")
    s2 = sep(direct, "sess_all", "session-level coverage (old)")
    s3 = sep(direct, "ev_all", "EVIDENCE-TURN coverage (new)")

    print()
    if s2 is not None and s3 is not None:
        if s3 > s2:
            print(f"  -> EVIDENCE-TURN coverage separates better "
                  f"({s3:+.1%} vs {s2:+.1%}). Adopt it for direct "
                  f"questions; keep session coverage for derived ones.")
        else:
            print(f"  -> NO improvement ({s3:+.1%} vs {s2:+.1%}). The new "
                  f"proxy is DISCARDED; session coverage stays.")

    print("\nBY CATEGORY (direct questions): evidence-turn coverage")
    per = defaultdict(lambda: [0, 0])
    for r in direct:
        per[r["type"]][0] += r["ev_all"]
        per[r["type"]][1] += 1
    for t in sorted(per):
        a, n = per[t]
        print(f"  {t:26s} {a:3d}/{n:<3d} = {a / max(1, n):5.1%}")


if __name__ == "__main__":
    main()
