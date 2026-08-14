"""$0 pre-spend check for the KU smoke: rebuild the packets of run1's
10 wrong knowledge-update answers against the NEW full-turns corpus +
full-turns eval DB, through the exact eval assembler path. Verifies,
before any API dollar: (a) gold sessions present, (b) gold answer
string present, (c) the [UPDATED ...] read-time annotation appears
where an update pair exists. No API calls anywhere.
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
ART = json.load(open(HERE / "qa_accuracy_longmemeval_500q_40k_r1.json"))
DB = str(HERE / "eval_dbs" / "longmemeval-s-raw-fullturns.db")
CONTEXT_CHARS = ART["context_chars"]
assert Path(DB).exists(), DB

os.environ["AGENTMEM_OS_DB_PATH"] = DB
os.environ["AGENTMEM_OS_DISABLE_REDIS"] = "1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))

from agentmem_os.storage.store import ConversationStore            # noqa: E402
from agentmem_os.llm.context_assembler import ContextAssembler     # noqa: E402
from agentmem_os.db import engine as _engine_mod                   # noqa: E402
from agentmem_os.db.models import Turn                             # noqa: E402
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

wrong = [r for r in ART["results"] if not r["correct"]
         and r["question_type"] == "knowledge-update"]
print(f"{len(wrong)} wrong KU answers to check against full-turns stack")

_scope_by_q = {it.question: list(it.scope_keys) for it in items}
import gate_c_facts_source as _gc                                  # noqa: E402
assert _gc.preflight(_scope_by_q), "gate C preflight failed"
print(_gc.install(assembler, _scope_by_q))
import gate_d_profile_source as _gd                                # noqa: E402
assert _gd.preflight(_scope_by_q), "gate D preflight failed"
print(_gd.install(assembler, _scope_by_q))

from agentmem_os.db.engine import get_session as _gs               # noqa: E402
try:
    from agentmem_os.agents.namespace import NamespaceManager
    _ns = NamespaceManager(_gs)
except Exception:
    _ns = None


def sid_for(scope_keys):
    key = "|".join(sorted(scope_keys))
    return f"longmemeval-scope-{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def ingest_scope(q):
    """Mirror of qa_accuracy_eval.ensure_scope_ingested (raw source),
    minus KG (not consumed by this eval's packet sections)."""
    sid = sid_for(q.scope_keys)
    db = _gs()
    try:
        if db.query(Turn).filter(Turn.session_id == sid).count() > 0:
            return sid, "preexisting"
    finally:
        db.close()
    store.get_or_create_session(sid, name="longmemeval-scope")
    if _ns is not None:
        _ns.ensure_agent_exists(sid)
    n = 0
    for mkey in q.scope_keys:
        mem = mem_by_id.get(mkey)
        if not mem or not mem.turns:
            continue
        for turn in mem.turns:
            content = turn.get("content", "")
            if not content:
                continue
            store.db.add(Turn(
                session_id=sid, role=turn.get("role", "user"),
                content=content,
                token_count=store.token_counter.count(content)))
            n += 1
        store.db.commit()
    return sid, f"ingested {n} turns"


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


report = []
for i, r in enumerate(wrong):
    q = by_q.get(r["question"])
    if q is None:
        report.append({"question": r["question"], "error": "not matched"})
        continue
    sid, how = ingest_scope(q)
    assembler.profile_session_ids = list(q.scope_keys)
    if getattr(assembler._profile, "_scope_map", None) is not None:
        assembler._profile.current_question = q.question
    packet = assembler.assemble(sid, q.question, agent_id=sid)
    packet = packet[:CONTEXT_CHARS]
    np_ = norm(packet)

    gold_sessions = [g for g in q.gold_keys if g in mem_by_id]
    presence = {}
    for g in gold_sessions:
        turns = sorted((t.get("content", "") for t in mem_by_id[g].turns),
                       key=len, reverse=True)[:5]
        found = False
        for t in turns:
            nt = norm(t)
            probe = nt[:80] if len(nt) > 80 else nt
            if len(probe) >= 40 and probe in np_:
                found = True
                break
            if len(nt) > 160 and nt[40:120] in np_:
                found = True
                break
        presence[g] = found

    gold_ans = str(q.gold_answer)
    ans_in = norm(gold_ans) in np_ if len(norm(gold_ans)) >= 4 else None
    n_updated = packet.count("[UPDATED")

    report.append({
        "question": q.question,
        "gold": gold_ans,
        "predicted_run1": r["predicted"],
        "ingest": how,
        "gold_sessions": len(gold_sessions),
        "gold_sessions_in_packet": sum(presence.values()),
        "gold_answer_string_in_packet": ans_in,
        "updated_annotations": n_updated,
        "packet_chars": len(packet),
        "sections": [s for s in ("[USER PROFILE]", "[SEMANTIC FACTS]",
                                 "[SEMANTIC MEMORY]", "[RECENT TURNS]")
                     if s in packet],
    })
    print(f"  {i+1}/{len(wrong)} gold-cov "
          f"{sum(presence.values())}/{len(gold_sessions)} "
          f"ans-in-packet={ans_in} [UPDATED]x{n_updated} ({how})",
          flush=True)

out = HERE / "ku_packet_check_fullturns.json"
json.dump(report, open(out, "w"), indent=1, ensure_ascii=False)
print("wrote", out)

ok_ans = sum(1 for r in report if r.get("gold_answer_string_in_packet"))
ok_cov = sum(1 for r in report
             if r.get("gold_sessions_in_packet") == r.get("gold_sessions"))
ann = sum(1 for r in report if r.get("updated_annotations", 0) > 0)
print(f"\nSUMMARY: {ok_ans}/{len(report)} gold answer string in packet | "
      f"{ok_cov}/{len(report)} full gold-session coverage | "
      f"{ann}/{len(report)} packets carry [UPDATED] annotations")
