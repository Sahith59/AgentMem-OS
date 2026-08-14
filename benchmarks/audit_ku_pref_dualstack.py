"""$0 dual-stack packet audit for the KU + pref regressions.

Run once per stack (STACK=old|new). Builds every KU (78) and pref (30)
question's packet through the exact eval path and records, per question:
  - packet size, per-section sizes
  - BREADTH: how many of the scope's sessions have at least one turn
    represented in the packet (and how many turns each)
  - gold-session coverage and gold-answer-string presence, split by
    which section carries it (facts vs verbatim)
  - [UPDATED] annotation count
Output: benchmarks/audit_dualstack_<stack>.json — diffed by the
companion report step. No API calls.
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

STACK = os.environ.get("STACK", "new")
HERE = Path(__file__).parent

if STACK == "new":
    DB = str(HERE / "eval_dbs" / "longmemeval-s-raw-fullturns.db")
    CORPUS = HERE / "extracted_memories" / "gate_c_facts.db"
else:
    DB = str(HERE / "eval_dbs" / "longmemeval-s-raw-2b3ebe5512.db")
    CORPUS = HERE / "extracted_memories" / "gate_c_facts.TRUNC800.db.bak"
assert Path(DB).exists(), DB
assert CORPUS.exists(), CORPUS

os.environ["AGENTMEM_OS_DB_PATH"] = DB
os.environ["AGENTMEM_OS_DISABLE_REDIS"] = "1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))

from agentmem_os.llm.context_assembler import ContextAssembler     # noqa: E402
from agentmem_os.db import engine as _engine_mod                   # noqa: E402
assert str(_engine_mod.DB_PATH) == DB, (_engine_mod.DB_PATH, DB)
from real_code_utils import install_dense_chroma                   # noqa: E402
assert install_dense_chroma(ContextAssembler) == "dense"
from corpus_loaders import load_longmemeval                        # noqa: E402

# The OLD stack must see the corpus the OLD run saw. The loader cache is
# full-turn now either way; what differed at answer time was (a) the
# eval DB's stored turns (truncated vs full) and (b) the facts corpus.
ds = load_longmemeval(n_queries=500, seed=42, split="s")
mem_by_id = {m.mid: m for m in ds.memories}
items = [q for q in ds.queries if q.gold_answer and q.scope_keys]
by_q = {q.question: q for q in items}

CONTEXT_CHARS = 40000
assembler = ContextAssembler()
assembler.allocations["semantic"] = int(CONTEXT_CHARS * 0.79 // 4)
assembler.allocations["recent"] = 1200

_scope_by_q = {it.question: list(it.scope_keys) for it in items}
import gate_c_facts_source as _gc                                  # noqa: E402
assert _gc.preflight(_scope_by_q, corpus=CORPUS), "gate C preflight failed"
print(_gc.install(assembler, _scope_by_q, corpus=CORPUS))
import gate_d_profile_source as _gd                                # noqa: E402
assert _gd.preflight(_scope_by_q), "gate D preflight failed"
print(_gd.install(assembler, _scope_by_q))


def sid_for(sk):
    return ("longmemeval-scope-"
            + hashlib.sha1("|".join(sorted(sk)).encode()).hexdigest()[:12])


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


SECTIONS = ("[USER PROFILE]", "[SEMANTIC FACTS]", "[SEMANTIC MEMORY]",
            "[RECENT TURNS]")


def split_sections(packet):
    """Return {section: text} by header positions."""
    pos = sorted((packet.find(h), h) for h in SECTIONS if h in packet)
    out = {}
    for i, (p, h) in enumerate(pos):
        end = pos[i + 1][0] if i + 1 < len(pos) else len(packet)
        out[h] = packet[p:end]
    return out


TYPES = tuple((os.environ.get(
    "AUDIT_TYPES", "knowledge-update,single-session-preference")).split(","))
targets = [q for q in items if q.question_type in TYPES]
print(f"stack={STACK}: auditing {len(targets)} questions")

report = []
for i, q in enumerate(targets):
    sid = sid_for(q.scope_keys)
    assembler.profile_session_ids = list(q.scope_keys)
    if getattr(assembler._profile, "_scope_map", None) is not None:
        assembler._profile.current_question = q.question
    packet = assembler.assemble(sid, q.question, agent_id=sid)
    packet = packet[:CONTEXT_CHARS]
    np_ = norm(packet)
    secs = split_sections(packet)
    nsecs = {h: norm(t) for h, t in secs.items()}

    # breadth: sessions represented in the packet at all, via per-turn
    # 80-char probes (cheap, deterministic)
    sess_present = {}
    for mkey in q.scope_keys:
        mem = mem_by_id.get(mkey)
        if not mem:
            continue
        hits = 0
        for t in mem.turns:
            nt = norm(t.get("content", ""))
            if len(nt) >= 40 and nt[:80] in np_:
                hits += 1
        if hits:
            sess_present[mkey] = hits

    golds = [g for g in q.gold_keys if g in mem_by_id]
    gold_cov = sum(1 for g in golds if g in sess_present)
    gold_ans = norm(str(q.gold_answer))
    probe = gold_ans[:40] if len(gold_ans) > 40 else gold_ans
    ans_in = probe in np_ if len(probe) >= 6 else None
    ans_in_facts = (probe in nsecs.get("[SEMANTIC FACTS]", "")
                    if ans_in else False)
    ans_in_verbatim = (probe in nsecs.get("[SEMANTIC MEMORY]", "")
                       if ans_in else False)

    report.append({
        "type": q.question_type,
        "question": q.question,
        "gold": str(q.gold_answer),
        "packet_chars": len(packet),
        "section_chars": {h: len(t) for h, t in secs.items()},
        "n_scope_sessions": len(q.scope_keys),
        "n_sessions_in_packet": len(sess_present),
        "turns_in_packet_total": sum(sess_present.values()),
        "gold_sessions": len(golds),
        "gold_cov": gold_cov,
        "gold_ans_in_packet": ans_in,
        "gold_ans_in_facts": ans_in_facts,
        "gold_ans_in_verbatim": ans_in_verbatim,
        "updated_annotations": packet.count("[UPDATED"),
    })
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(targets)}", flush=True)

out = HERE / (f"audit_dualstack_{STACK}"
              + os.environ.get("AUDIT_SUFFIX", "") + ".json")
json.dump(report, open(out, "w"), indent=1, ensure_ascii=False)
print("wrote", out)
