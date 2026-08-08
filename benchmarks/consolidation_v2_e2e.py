"""
Stage 6 G2: the FULL product loop with REAL models — every drive call
goes through mcp_server.handle_call_tool exactly as an MCP client
would (save_memory → consolidate_session → recall_memory /
get_knowledge_graph), real llama3.1 extraction + supersession judge,
real LongMemEval `_s` sessions, $0.

Cases (design D3):
  A. Rachel knowledge-update pair — continuity with the Stage 4/5
     artifacts (same sessions, now through the product surface).
  B. Cross-session recall + double-consolidation idempotency on A.
  C. The 5K boundary pair (events/identity excluded from judgment —
     both values visible, disclosed).
  D. ONE FRESH knowledge-update case never used in any prior stage's
     artifacts — index FRESH_IDX below, chosen before running,
     disclosed — so the demo is not secretly tuned to two examples.
Whatever the real model does is reported as-is.
"""
import asyncio
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

# FORCED isolation, THREE channels (the standing rules + the breach
# this script's own first run committed): SQLite via env, Redis via
# kill-switch, and the StorageManager tree (persistent Chroma vector
# store!) via a scratch cwd holding a scratch config.yaml —
# StorageManager reads config.yaml RELATIVE TO CWD, and the first run
# of this script created empty collections in the DEV vector store
# (/Volumes/Sahith_SSD/AgentMem-OS/vectors) before this guard
# existed. Disclosed in the stage record; dev-store cleanup listed
# for the founder.
_SCRATCH = Path(tempfile.mkdtemp(prefix="agentmem-s6e2e-"))
os.environ["AGENTMEM_OS_DB_PATH"] = str(_SCRATCH / "e2e.db")
os.environ["AGENTMEM_OS_DISABLE_REDIS"] = "1"

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))

_repo_cfg = (HERE.parent / "config.yaml").read_text()
_scratch_cfg = _repo_cfg.replace(
    'base_path: "/Volumes/Sahith_SSD/AgentMem-OS/"',
    f'base_path: "{_SCRATCH}/"')
assert str(_SCRATCH) in _scratch_cfg, "config base_path rewrite failed"
(_SCRATCH / "config.yaml").write_text(_scratch_cfg)
os.chdir(_SCRATCH)


def _verify_storage_isolated():
    """final-pass m7: the rewrite assert above catches a failed config
    edit but nothing caught a removed chdir — prove the resolved tree
    actually points at scratch before anything writes through it."""
    from agentmem_os.storage.manager import StorageManager

    base = StorageManager().base_path
    assert base.startswith(str(_SCRATCH)), (
        f"storage tree NOT isolated: {base}")

FRESH_IDX = 10  # never appeared in Stage 4/5 artifacts (those used 22 and 0)


async def _call(name, args):
    from mcp_server.server import handle_call_tool

    result = await handle_call_tool(name, args)
    return json.loads(result[0].text)


def _await_background(timeout=240.0):
    """Design D4: bounded, loud quiescence wait for save_turn's
    background threads (first ingests serialize behind the ~87s cold
    alias-model load — measured, disclosed)."""
    deadline = time.monotonic() + timeout
    main = threading.main_thread()
    while time.monotonic() < deadline:
        busy = [t for t in threading.enumerate()
                if t is not main and t.is_alive()
                and not t.name.startswith(("pytest", "asyncio"))
                and not t.__class__.__module__.startswith("tqdm")]
        if not busy:
            return
        time.sleep(0.5)
    raise SystemExit(f"background threads still alive after {timeout}s")


async def _ingest_mcp(mems, sid):
    n = 0
    for line in mems[sid]["content"].split("\n"):
        line = line.strip()
        if not line:
            continue
        role = "user" if line.startswith("User:") else \
               "assistant" if line.startswith("Assistant:") else "system"
        await _call("save_memory", {"session_id": sid, "role": role,
                                    "content": line})
        n += 1
    return n


def _sections(ctx):
    return {m.group(1): m.group(2)
            for m in re.finditer(r"<\[([A-Z ]+)\]>(.*?)</\[\1\]>",
                                 ctx, re.S)}


def _report(ctx, needle, label):
    spans = _sections(ctx)
    hits = [name for name, body in spans.items()
            if needle.lower() in body.lower()]
    print(f"  {label}: {needle!r} in {hits or 'NOWHERE'}")
    for name, body in spans.items():
        print(f"    {name}: {len(body)} chars")
    return hits


async def main():
    ds = json.load(open(HERE / "benchmark_cache/longmemeval_s.json"))
    mems = {m["mid"]: m for m in ds["memories"]}
    qs = [q for q in ds["queries"]
          if q.get("question_type") == "knowledge-update"
          and len(q["gold_keys"]) == 2]

    _verify_storage_isolated()
    print(f"scratch DB: {os.environ['AGENTMEM_OS_DB_PATH']}")
    print(f"scratch storage tree verified: {_SCRATCH}")

    # ── A. Rachel pair, in order, through the product surface ──────────
    q = qs[22]
    sids = q["gold_keys"]
    print(f"\n== A. KNOWLEDGE-UPDATE via MCP (question: "
          f"{q['question'][:60]!r}, gold {q['gold_answer']!r}) ==")
    for sid in sids:
        n = await _ingest_mcp(mems, sid)
        _await_background()
        r = await _call("consolidate_session", {"session_id": sid})
        sup = r.get("supersession") or {}
        print(f"  {sid}: {n} turns -> {r.get('created')} facts, "
              f"{r.get('entities_linked')} links, "
              f"superseded={sup.get('superseded', [])}, "
              f"judge_failure={r.get('judge_failure')}")
    recall = await _call("recall_memory", {
        "session_id": sids[0], "query": q["question"]})
    hits = _report(recall["context"], q["gold_answer"], "recall@gold-1  ")
    a_ok = "SEMANTIC FACTS" in hits

    # ── B. Cross-session + idempotency ─────────────────────────────────
    print("\n== B. CROSS-SESSION + IDEMPOTENCY ==")
    neutral = next(c for c in q.get("scope_keys", [])
                   if c not in sids and c in mems)
    await _ingest_mcp(mems, neutral)
    _await_background()
    recall_far = await _call("recall_memory", {
        "session_id": neutral, "query": q["question"]})
    far_hits = _report(recall_far["context"], q["gold_answer"],
                       f"recall@{neutral[:10]}")
    b_ok = far_hits == ["SEMANTIC FACTS"]
    r2 = await _call("consolidate_session", {"session_id": sids[0]})
    print(f"  re-consolidate {sids[0]}: created={r2.get('created')} "
          f"(0 = idempotent re-affirmation)")
    idem_ok = r2.get("created") == 0

    # ── C. 5K boundary pair (disclosed exclusion) ──────────────────────
    q5 = qs[0]
    print(f"\n== C. BOUNDARY via MCP (gold {q5['gold_answer'][:40]!r}) ==")
    for sid in q5["gold_keys"]:
        await _ingest_mcp(mems, sid)
        _await_background()
        r = await _call("consolidate_session", {"session_id": sid})
        print(f"  {sid}: {r.get('created')} facts, "
              f"judge_failure={r.get('judge_failure')}")
    recall5 = await _call("recall_memory", {
        "session_id": q5["gold_keys"][0], "query": q5["question"]})
    _report(recall5["context"], "25:50", "new value      ")
    _report(recall5["context"], "27:12", "old value      ")

    # ── D. FRESH case (index disclosed above; never in prior artifacts)
    qf = qs[FRESH_IDX]
    print(f"\n== D. FRESH CASE qs[{FRESH_IDX}] (question: "
          f"{qf['question'][:60]!r}, gold {qf['gold_answer'][:40]!r}) ==")
    for sid in qf["gold_keys"]:
        await _ingest_mcp(mems, sid)
        _await_background()
        r = await _call("consolidate_session", {"session_id": sid})
        sup = r.get("supersession") or {}
        print(f"  {sid}: {r.get('created')} facts, "
              f"superseded={sup.get('superseded', [])}, "
              f"judge_failure={r.get('judge_failure')}")
    recallf = await _call("recall_memory", {
        "session_id": qf["gold_keys"][0], "query": qf["question"]})
    f_hits = _report(recallf["context"], qf["gold_answer"],
                     "recall@gold-1  ")
    d_ok = bool(f_hits)

    # ── E. KG surface ──────────────────────────────────────────────────
    kg = await _call("get_knowledge_graph",
                     {"session_id": sids[0], "entity": "Rachel"})
    print(f"\n== E. KG surface == subgraph len="
          f"{len(str(kg.get('subgraph', '')))} chars")

    print(f"\nRESULT: A_answer_as_fact={a_ok} "
          f"B_cross_session_facts_only={b_ok} idempotent={idem_ok} "
          f"D_fresh_case_answer_found={d_ok}")
    if not (a_ok and idem_ok):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
