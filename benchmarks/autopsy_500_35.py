"""$0 autopsy of the 35 wrong answers (ku/ssa/pref) in the n=500 run.

Rebuilds each question's context packet through the SAME code path the
eval used (same DB, same assembler config, same gate installs), then
records per question: gold-session presence in the packet, gold-answer
string presence, sections present, and the 150-sample cross-reference.
No API calls anywhere.
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
ART = json.load(open(HERE / "qa_accuracy_longmemeval_500q_40k_r1.json"))
DB = ART["db_path"]
CONTEXT_CHARS = ART["context_chars"]
assert Path(DB).exists(), DB

os.environ["AGENTMEM_OS_DB_PATH"] = DB
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))

from agentmem_os.storage.store import ConversationStore            # noqa: E402
from agentmem_os.llm.context_assembler import ContextAssembler     # noqa: E402
from agentmem_os.db import engine as _engine_mod                   # noqa: E402
assert str(_engine_mod.DB_PATH) == DB, (_engine_mod.DB_PATH, DB)

from real_code_utils import install_dense_chroma                   # noqa: E402
assert install_dense_chroma(ContextAssembler) == "dense"

from corpus_loaders import load_longmemeval                        # noqa: E402
ds = load_longmemeval(n_queries=500, seed=ART["seed"], split="s")
mem_by_id = {m.mid: m for m in ds.memories}
items = [q for q in ds.queries if q.gold_answer and q.scope_keys]
by_q = {q.question: q for q in items}

store = ConversationStore()
assembler = ContextAssembler()
assembler.allocations["semantic"] = int(CONTEXT_CHARS * 0.79 // 4)
assembler.allocations["recent"] = 1200

wrong = [r for r in ART["results"] if not r["correct"] and r["question_type"] in
         ("knowledge-update", "single-session-assistant",
          "single-session-preference")]
print(f"{len(wrong)} wrong answers to autopsy")

_scope_by_q = {it.question: list(it.scope_keys) for it in items}
import gate_c_facts_source as _gc                                  # noqa: E402
assert _gc.preflight(_scope_by_q), "gate C preflight failed"
print(_gc.install(assembler, _scope_by_q))
import gate_d_profile_source as _gd                                # noqa: E402
assert _gd.preflight(_scope_by_q), "gate D preflight failed"
print(_gd.install(assembler, _scope_by_q))


def sid_for(scope_keys):
    key = "|".join(sorted(scope_keys))
    return f"longmemeval-scope-{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


# 150-sample cross-reference: was this question sampled there, and correct?
cross = {}
for suf in ("_ctx40k", "_ctx40k_r2", "_ctx40k_r3"):
    art = json.load(open(HERE / f"qa_accuracy_longmemeval{suf}.json"))
    for r in art["results"]:
        cross.setdefault(r["question"], []).append(r["correct"])

report = []
for i, r in enumerate(wrong):
    q = by_q.get(r["question"])
    if q is None:
        report.append({"question": r["question"], "error": "not matched"})
        continue
    sid = sid_for(q.scope_keys)
    assembler.profile_session_ids = list(q.scope_keys)
    if getattr(assembler._profile, "_scope_map", None) is not None:
        assembler._profile.current_question = q.question
    packet = assembler.assemble(sid, q.question, agent_id=sid)
    packet = packet[:CONTEXT_CHARS]
    np = norm(packet)

    gold_sessions = [g for g in q.gold_keys if g in mem_by_id]
    presence = {}
    for g in gold_sessions:
        turns = sorted((t.get("content", "") for t in mem_by_id[g].turns),
                       key=len, reverse=True)[:5]
        found = False
        for t in turns:
            nt = norm(t)
            probe = nt[:80] if len(nt) > 80 else nt
            if len(probe) >= 40 and probe in np:
                found = True
                break
            # verbatim turns may be truncated mid-turn; try a mid-slice
            if len(nt) > 160 and nt[40:120] in np:
                found = True
                break
        presence[g] = found

    gold_ans = str(q.gold_answer)
    ans_in_packet = norm(gold_ans) in np if len(norm(gold_ans)) >= 4 else None

    report.append({
        "type": r["question_type"],
        "question": q.question,
        "question_date": getattr(q, "question_date", ""),
        "gold": gold_ans,
        "predicted": r["predicted"],
        "gold_sessions": len(gold_sessions),
        "gold_sessions_in_packet": sum(presence.values()),
        "presence": presence,
        "gold_answer_string_in_packet": ans_in_packet,
        "packet_chars": len(packet),
        "sections": [s for s in ("[USER PROFILE]", "[SEMANTIC FACTS]",
                                 "[SEMANTIC MEMORY]", "[RECENT TURNS]")
                     if s in packet],
        "in_150_sample": cross.get(q.question),
        "packet_tail_hint": packet[-400:],
    })
    print(f"  {i+1}/{len(wrong)} {r['question_type'][:12]:14s} "
          f"gold-cov {sum(presence.values())}/{len(gold_sessions)} "
          f"ans-in-packet={ans_in_packet}")

out = HERE / "autopsy_500_35.json"
json.dump(report, open(out, "w"), indent=1, ensure_ascii=False)
print("wrote", out)

# summary
from collections import Counter, defaultdict
agg = defaultdict(Counter)
for r in report:
    if "error" in r:
        continue
    cov = "FULL" if r["gold_sessions_in_packet"] == r["gold_sessions"] else (
        "PARTIAL" if r["gold_sessions_in_packet"] else "NONE")
    agg[r["type"]][cov] += 1
    agg[r["type"]]["ans_str_present" if r["gold_answer_string_in_packet"]
                   else "ans_str_absent"] += 0 if r["gold_answer_string_in_packet"] is None else 1
print("\nGOLD-SESSION COVERAGE of the wrong answers:")
for t, c in agg.items():
    print(f"  {t}: {dict(c)}")
