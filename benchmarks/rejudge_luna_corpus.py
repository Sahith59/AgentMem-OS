"""Re-judgment pass for the Luna corpus (F-20 repair, founder-approved
~$10-12). The original 86,395 judgments ran through the mute API prompt
(schema fields never stated) and applied nothing; they are deleted and
every live state/preference fact is re-judged through the FIXED
gpt-4o-mini path. Threaded; resume-safe (a fact with a judgment row is
never re-judged, same rule as the pipeline).
"""
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).parent
DB = str(HERE / "extracted_memories" / "gate_c_facts_luna.db")
os.environ["AGENTMEM_OS_DB_PATH"] = DB
os.environ["AGENTMEM_OS_DISABLE_REDIS"] = "1"
os.environ.setdefault("AGENTMEM_OS_SUPERSESSION_API_MODEL", "gpt-4o-mini")
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))

for line in (HERE.parent / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
WIPE = "--wipe-mute-judgments" in sys.argv


def main():
    from agentmem_os.db.engine import get_session
    from agentmem_os.db.models import SemanticFact, SupersessionJudgment
    from agentmem_os.llm.supersession import SupersessionJudge
    from sqlalchemy import exists

    db = get_session()
    try:
        if WIPE:
            n = db.query(SupersessionJudgment).delete()
            db.commit()
            print(f"wiped {n} mute judgment rows", flush=True)
        todo = [r[0] for r in db.query(SemanticFact.id)
                .filter(SemanticFact.fact_type.in_(("state", "preference")))
                .filter(~exists().where(
                    SupersessionJudgment.fact_id == SemanticFact.id))
                .order_by(SemanticFact.id.asc()).all()]
    finally:
        db.close()
    print(f"{len(todo)} facts to judge | {WORKERS} workers", flush=True)

    judge = SupersessionJudge(get_session)
    lock = threading.Lock()
    state = {"ok": 0, "fail": 0, "applied": 0, "t0": time.time()}

    def work(fid):
        try:
            r = judge.judge_fact(fid)
            with lock:
                state["ok"] += 1
                state["applied"] += len(r.get("superseded", [])) + \
                    len(r.get("cancelled", []))
        except Exception as e:
            with lock:
                state["fail"] += 1
            if state["fail"] < 10:
                print(f"  FAIL fact {fid}: {type(e).__name__}: "
                      f"{str(e)[:120]}", flush=True)
        n = state["ok"] + state["fail"]
        if n % 500 == 0:
            rate = n / max(1e-9, time.time() - state["t0"])
            print(f"  {n}/{len(todo)} | applied {state['applied']} | "
                  f"{rate*3600:.0f}/h | ETA "
                  f"{(len(todo)-n)/max(1e-9,rate)/3600:.1f}h", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, todo))

    print(f"DONE: ok={state['ok']} fail={state['fail']} "
          f"applied={state['applied']} in "
          f"{(time.time()-state['t0'])/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
