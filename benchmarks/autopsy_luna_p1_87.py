"""$0 deep autopsy of the 87 LUNA-P1 failures (413/500 run).

Per failed question, four layers of evidence:
  1. STABILITY: correctness across the three llama-corpus Luna runs
     and the GPT-4o run — chronic (wrong everywhere) vs flip.
  2. PACKET: rebuilt through the exact eval path on the Luna corpus —
     gold-session coverage, gold-answer-string presence, sections,
     elisions, [UPDATED] annotations.
  3. CORPUS (counting questions): are the countable INSTANCES present
     as facts in the Luna corpus at all (write-time delivery), and how
     many made the packet (read-time delivery)?
  4. ANSWER ANATOMY: predicted-vs-gold shape (number-off-by-one?
     unit? abstention? stale value?).
Output: benchmarks/autopsy_luna_p1_87.json + printed decomposition.
"""
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DB = str(HERE / "eval_dbs" / "longmemeval-s-raw-fullturns.db")
CORPUS = HERE / "extracted_memories" / "gate_c_facts_luna.db"
os.environ["AGENTMEM_OS_DB_PATH"] = DB
os.environ["AGENTMEM_OS_DISABLE_REDIS"] = "1"
os.environ["AGENTMEM_OS_GATE_C_CORPUS"] = str(CORPUS)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))

from agentmem_os.llm.context_assembler import ContextAssembler     # noqa: E402
from real_code_utils import install_dense_chroma                   # noqa: E402
assert install_dense_chroma(ContextAssembler) == "dense"
from corpus_loaders import load_longmemeval                        # noqa: E402

ds = load_longmemeval(n_queries=500, seed=42, split="s")
mem_by_id = {m.mid: m for m in ds.memories}
items = [q for q in ds.queries if q.gold_answer and q.scope_keys]
by_q = {q.question: q for q in items}

assembler = ContextAssembler()
assembler.allocations["semantic"] = int(40000 * 0.79 // 4)
assembler.allocations["recent"] = 1200
_scope_by_q = {it.question: list(it.scope_keys) for it in items}
import gate_c_facts_source as _gc                                  # noqa: E402
assert _gc.preflight(_scope_by_q, corpus=CORPUS)
_gc.install(assembler, _scope_by_q, corpus=CORPUS)
import gate_d_profile_source as _gd                                # noqa: E402
assert _gd.preflight(_scope_by_q, corpus=CORPUS)
_gd.install(assembler, _scope_by_q, corpus=CORPUS)

run = json.load(open(HERE / "qa_accuracy_longmemeval_500q_lunacorpus_p1.json"))
prior = {}
for suf, tag in (("_500q_40k_fullturns_luna", "l1"),
                 ("_500q_40k_fullturns_luna_r2", "l2"),
                 ("_500q_40k_fullturns_luna_r3", "l3"),
                 ("_500q_40k_fullturns_r1", "g4o")):
    art = json.load(open(HERE / f"qa_accuracy_longmemeval{suf}.json"))
    prior[tag] = {r["question"]: r["correct"] for r in art["results"]}

failures = [r for r in run["results"] if not r["correct"]]
print(f"{len(failures)} failures to autopsy", flush=True)

COUNT_RE = re.compile(r"\bhow (many|much|often)\b|\btotal\b|\bcount\b",
                      re.I)
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def sid_for(sk):
    return ("longmemeval-scope-"
            + hashlib.sha1("|".join(sorted(sk)).encode()).hexdigest()[:12])


import sqlite3                                                     # noqa: E402
corpus_con = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)


def corpus_instances(q):
    """Facts in the Luna corpus from this scope's gold sessions."""
    golds = [g for g in q.gold_keys if g in mem_by_id]
    n = 0
    for g in golds:
        n += corpus_con.execute(
            "SELECT COUNT(*) FROM semantic_facts WHERE "
            "source_session_id=?", (g,)).fetchone()[0]
    return len(golds), n


report = []
for i, r in enumerate(failures):
    q = by_q.get(r["question"])
    if q is None:
        continue
    sid = sid_for(q.scope_keys)
    assembler.profile_session_ids = list(q.scope_keys)
    if getattr(assembler._profile, "_scope_map", None) is not None:
        assembler._profile.current_question = q.question
    packet = assembler.assemble(sid, q.question, agent_id=sid)[:40000]
    np_ = norm(packet)

    golds = [g for g in q.gold_keys if g in mem_by_id]
    cov = 0
    for g in golds:
        for t in sorted((t.get("content", "") for t in mem_by_id[g].turns),
                        key=len, reverse=True)[:5]:
            nt = norm(t)
            if (len(nt) >= 40 and nt[:80] in np_) or \
               (len(nt) > 160 and nt[40:120] in np_):
                cov += 1
                break
    gold = norm(q.gold_answer)
    gp = gold[:40] if len(gold) > 40 else gold
    ans_in = gp in np_ if len(gp) >= 6 else None

    hist = [prior[t].get(r["question"]) for t in ("l1", "l2", "l3", "g4o")]
    n_right_prior = sum(1 for h in hist if h)
    stability = ("chronic" if n_right_prior == 0 else
                 "mostly-wrong" if n_right_prior == 1 else "flip")

    is_count = bool(COUNT_RE.search(q.question))
    g_sessions, g_facts = corpus_instances(q) if is_count else (None, None)

    pred_nums = NUM_RE.findall(str(r["predicted"]))
    gold_nums = NUM_RE.findall(str(q.gold_answer))
    num_shape = None
    if gold_nums:
        if not pred_nums:
            num_shape = "no-number-answered"
        elif pred_nums[0] == gold_nums[0]:
            num_shape = "same-number-judged-wrong"
        else:
            try:
                d = abs(float(pred_nums[0]) - float(gold_nums[0]))
                num_shape = ("off-by-1" if d == 1 else
                             "off-by-2" if d == 2 else "off-by-more")
            except ValueError:
                num_shape = "unparsed"

    report.append({
        "type": r["question_type"], "question": q.question,
        "gold": str(q.gold_answer), "predicted": str(r["predicted"])[:200],
        "stability": stability, "prior_runs_right": n_right_prior,
        "gold_cov": f"{cov}/{len(golds)}",
        "full_cov": cov == len(golds),
        "ans_in_packet": ans_in,
        "is_counting": is_count,
        "gold_sessions_facts_in_corpus": g_facts,
        "num_shape": num_shape,
        "elisions": packet.count("[...]"),
        "updated_anns": packet.count("[UPDATED"),
        "abstained": "not mentioned" in norm(r["predicted"])[:80]
                     or "no information" in norm(r["predicted"])[:80],
    })
    if (i + 1) % 15 == 0:
        print(f"  {i+1}/{len(failures)}", flush=True)

json.dump(report, open(HERE / "autopsy_luna_p1_87.json", "w"),
          indent=1, ensure_ascii=False)

# ── decomposition ────────────────────────────────────────────────────
print("\n=== DECOMPOSITION ===")
by_type = Counter(r["type"] for r in report)
print("by category:", dict(by_type))
by_stab = Counter(r["stability"] for r in report)
print("by stability:", dict(by_stab))
cnt = [r for r in report if r["is_counting"]]
print(f"counting questions: {len(cnt)} | full gold coverage: "
      f"{sum(1 for r in cnt if r['full_cov'])} | num shapes: "
      f"{dict(Counter(r['num_shape'] for r in cnt))}")
chronic = [r for r in report if r["stability"] == "chronic"]
print(f"CHRONIC ({len(chronic)}): by type "
      f"{dict(Counter(r['type'] for r in chronic))}")
print(f"  chronic w/ full coverage: "
      f"{sum(1 for r in chronic if r['full_cov'])} "
      f"(evidence present, still always wrong)")
flips = [r for r in report if r["stability"] == "flip"]
print(f"FLIPS ({len(flips)}): by type "
      f"{dict(Counter(r['type'] for r in flips))}")
same_num = [r for r in report if r["num_shape"] == "same-number-judged-wrong"]
print(f"same-number-judged-wrong (judge/format): {len(same_num)}")
abst = [r for r in report if r["abstained"]]
print(f"abstentions: {len(abst)} (of which full-coverage: "
      f"{sum(1 for r in abst if r['full_cov'])})")
print("wrote autopsy_luna_p1_87.json")
