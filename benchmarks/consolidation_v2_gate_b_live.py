"""
Gate B re-run for the plans-as-events flip (founder-approved 2026-08-08)
— measured on the LIVE ENGINE, A/B against the pre-flip prompt.

Why not just re-run consolidation_v2_gate_b.py: that script embeds the
pre-engine DRAFT prompt ("FACT:" lines, no fact_type at all) — it is
structurally INSENSITIVE to the flip and re-running it would be gate
theater. The original 91.2% artifact measured that draft prompt,
pre-validation; the number that matters for Gate C is what the LIVE
pipeline STORES. So this harness runs the real ConsolidationV2 (real
llama3.1, real validator, real linking + judging) over the same Gate-B
hard-core sessions TWICE in one process — pass A with the flipped
prompt (HEAD), pass B with the pre-flip fact_type/t_occurred lines
monkeypatched back — and scores BOTH with Gate B's own
number-preservation metric against the trusted haiku extraction. The
A-minus-B diff isolates the flip; the absolute numbers are the live
pipeline's own floor.

Also measured: PLANNED-MARKER REACHABILITY — the whole point of the
flip. Stage 3 measured 0/23 (dated plans extracted as states, marker
prompt-unreachable); pass A counts event_status='planned' rows.

STOP RULE (recorded before running): if pass A's number-preservation
drops more than 2 points below pass B's, the flip regresses extraction
quality — REVERT the prompt and report. Reachability must be > 0 or
the flip failed at its own purpose.
"""
import json
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ["AGENTMEM_OS_DB_PATH"] = str(
    Path(tempfile.mkdtemp(prefix="agentmem-gatebl-")) / "gb.db")
os.environ["AGENTMEM_OS_DISABLE_REDIS"] = "1"

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))

MAX_SESSIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 16

# The exact pre-flip lines (from git history of llm/consolidation_v2.py)
OLD_FACT_TYPE = (
    '- fact_type: "event" = something that ALREADY HAPPENED at a time; '
    '"state" = an ongoing situation, including PLANS ("The user plans to '
    'attend X on DATE" is a state, never an event); "preference" = a '
    'like/dislike/choice; "identity" = who the user is.')
OLD_T_OCC = (
    "- t_occurred: the date the event happened, YYYY/MM/DD (or YYYY/MM "
    "if only the month is known)")


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


_NUM = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+(?:\.\d+)?)\b")


def main():
    ds = json.load(open(HERE / "benchmark_cache/longmemeval_s.json"))
    mems = {m["mid"]: m for m in ds["memories"]}
    haiku = json.load(open(
        HERE / "extracted_memories/rich_memories_haiku_LongMemEval.json"))

    # EXACT same session selection as consolidation_v2_gate_b.py
    # (comparability): evidence sessions behind the both-models-failed
    # aggregation questions.
    slice_res = json.load(open(
        HERE / "qa_accuracy_longmemeval_answerer54mini.json"))["results"]
    old = json.load(open(
        HERE / "qa_accuracy_longmemeval.json"))["results"]
    oldmap = {r["question"]: r for r in old
              if r.get("question_type") in ("temporal-reasoning",
                                            "multi-session")}
    hardcore = [r for r in slice_res if r["question"] in oldmap
                and not r["correct"]
                and not oldmap[r["question"]]["correct"]]
    qmap = {q["question"]: q for q in ds["queries"]}
    agg = re.compile(r"how (many|much|often)|total|average", re.I)
    sessions = []
    for r in hardcore:
        if not agg.search(r["question"]):
            continue
        q = qmap[r["question"]]
        new = [k for k in q["gold_keys"] if k in haiku
               and k not in sessions]
        if len(sessions) + len(new) > MAX_SESSIONS:
            continue
        sessions.extend(new)
        if len(sessions) >= MAX_SESSIONS:
            break

    print(f"Gate B LIVE re-run: llama3.1 | {len(sessions)} sessions | "
          f"A=flipped prompt (HEAD) vs B=pre-flip prompt")
    print(f"scratch DB: {os.environ['AGENTMEM_OS_DB_PATH']}")

    from agentmem_os.db.engine import get_session
    from agentmem_os.db.models import (
        SemanticFact, Session as SessionRow, Turn,
    )
    from agentmem_os.llm.consolidation_v2 import ConsolidationV2

    db = get_session()
    try:
        for sid in sessions:
            if db.query(SessionRow).filter(
                    SessionRow.session_id == sid).first():
                continue
            db.add(SessionRow(session_id=sid))
            for line in mems[sid]["content"].split("\n"):
                line = line.strip()
                if not line:
                    continue
                role = "user" if line.startswith("User:") else \
                    "assistant" if line.startswith("Assistant:") else \
                    "system"
                db.add(Turn(session_id=sid, role=role, content=line))
        db.commit()
    finally:
        db.close()

    def run_pass(label, agent_scope, patch_old):
        cv2 = ConsolidationV2(get_session)
        if patch_old:
            orig_prompt = cv2._prompt

            def old_prompt(session_date, transcript):
                p = orig_prompt(session_date, transcript)
                # restore the two pre-flip lines
                p = re.sub(r"- fact_type:.*who the user is\.",
                           OLD_FACT_TYPE, p, count=1, flags=re.S)
                p = p.replace(
                    "- t_occurred: the date the event happened — or, "
                    "for a planned event, the date it is planned FOR — "
                    "YYYY/MM/DD (or YYYY/MM if only the month is known)",
                    OLD_T_OCC)
                assert "never an event" in p, "pre-flip patch failed"
                return p

            cv2._prompt = old_prompt
        # AgentNamespace FK: create the scope row first
        from agentmem_os.db.models import AgentNamespace
        db = get_session()
        try:
            if not db.query(AgentNamespace).filter(
                    AgentNamespace.agent_id == agent_scope).first():
                db.add(AgentNamespace(agent_id=agent_scope,
                                      name=agent_scope))
                db.commit()
        finally:
            db.close()
        for i, sid in enumerate(sessions):
            r = cv2.consolidate_session(sid, agent_id=agent_scope)
            print(f"  [{label}] {i + 1}/{len(sessions)} {sid}: "
                  f"{r.get('created')} facts, "
                  f"judge_failure={r.get('judge_failure')}", flush=True)

        db = get_session()
        try:
            rows = (db.query(SemanticFact)
                    .filter(SemanticFact.agent_id == agent_scope).all())
            by_sid = {}
            planned = []
            for f in rows:
                by_sid.setdefault(f.source_session_id, []).append(
                    f.fact_text)
                if f.event_status == "planned":
                    planned.append((f.source_session_id,
                                    f.t_occurred, f.fact_text[:90]))
        finally:
            db.close()

        kept = missed = 0
        for sid in sessions:
            local_text = _norm(" ".join(by_sid.get(sid, [])))
            for hm in haiku.get(sid, []):
                for num in set(_NUM.findall(_norm(hm["memory"]))):
                    if num in local_text:
                        kept += 1
                    else:
                        missed += 1
        pct = kept / max(1, kept + missed)
        print(f"  [{label}] number-preservation vs haiku: kept {kept}, "
              f"missed {missed} ({pct:.1%}); planned rows: "
              f"{len(planned)}")
        for p in planned:
            print(f"    planned: {p}")
        return pct, len(planned)

    pct_b, planned_b = run_pass("B pre-flip", "gateb-preflip",
                                patch_old=True)
    pct_a, planned_a = run_pass("A flipped ", "gateb-flip",
                                patch_old=False)

    diff = (pct_a - pct_b) * 100
    print(f"\nRESULT: flipped={pct_a:.1%} pre-flip={pct_b:.1%} "
          f"diff={diff:+.1f}pts | planned reachability: "
          f"{planned_a} (was {planned_b} pre-flip; Stage-3 era: 0)")
    print("STOP RULE: revert if diff < -2.0pts or planned_a == 0")
    if diff < -2.0 or planned_a == 0:
        print("VERDICT: STOP-RULE HIT — revert the prompt flip")
        sys.exit(1)
    print("VERDICT: FLIP HOLDS — quality within noise, marker reachable")


if __name__ == "__main__":
    main()
