"""
Stage 5 G2 / $0 diagnostics: facts-first retrieval measured on REAL
LongMemEval knowledge-update sessions through the REAL product path —
engine-backed store, real llama3.1 extraction + supersession judgment
(local Ollama, $0), the real ContextAssembler with the benchmark's
champion TF-IDF chunk backend installed (install_tfidf_chroma — the
exact stand-in the banked numbers used, and it keeps a diagnostic run
out of the live ~/.agentmem_os vector store).

Honest scope labels:
  A. NO-REGRESS (no LLM): real ingested turns, EMPTY fact store —
     assembled context with the facts tier enabled must be
     BYTE-IDENTICAL to the facts-disabled assembly. Exits non-zero on
     failure — this artifact is treated as a gate.
  B. KNOWLEDGE-UPDATE, in order (real extraction + judge): does the
     assembled context surface the CURRENT answer as a dated fact,
     and at what char cost? NOTE (G3 R1 M4): the gold string lives
     ONLY in the second gold session, so the raw-turn path missing it
     when recalling against the first is STRUCTURAL (session-scoped
     retrieval cannot reach another session's turns at any budget) —
     the same axis Part C measures, stated here so nobody reads the
     comparison as a retrieval quality measurement.
  C. CROSS-SESSION REACH: the actual measurement of that axis —
     recall against a session that never discussed the answer;
     per-session chunk retrieval structurally cannot surface it; the
     scope-wide fact tier can.
  D. BOUNDARY, disclosed: the event-typed 5K metric case — the judge
     excludes events and identity facts by design; the diagnostic
     PRINTS the fact lines and probes BOTH the old and new value so
     "both visible, no false supersession" is measured, not asserted.
Whatever the real model does is reported as-is, including extraction
misses (they degrade Part B/C to honest failures, not silent passes).
Uses install_best_chroma (the champion TF-IDF stand-in).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

# FORCED (the R5 lesson): a diagnostic must never touch a live DB,
# inherited env included.
os.environ["AGENTMEM_OS_DB_PATH"] = str(
    Path(tempfile.mkdtemp(prefix="agentmem-s5diag-")) / "s5diag.db")
# Second isolation channel (found BY this diagnostic's Part A): Redis
# hot-cache keys carry no DB identity, so a live localhost Redis serves
# ghost turns from earlier runs — and its 10-turn cap silently halved
# recall depth on warm reads (6264 vs 4960 chars, run 2). Kill it.
os.environ["AGENTMEM_OS_DISABLE_REDIS"] = "1"

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))


def _ingest(get_session, mems, sid):
    from agentmem_os.db.models import Session as SessionRow, Turn
    db = get_session()
    try:
        if db.query(SessionRow).filter(
                SessionRow.session_id == sid).first():
            return
        db.add(SessionRow(session_id=sid))
        for line in mems[sid]["content"].split("\n"):
            line = line.strip()
            if not line:
                continue
            role = "user" if line.startswith("User:") else \
                   "assistant" if line.startswith("Assistant:") else "system"
            db.add(Turn(session_id=sid, role=role, content=line))
        db.commit()
    finally:
        db.close()


def _sections(ctx):
    """Char spans of each labelled section in the assembled context."""
    import re
    spans = {}
    for m in re.finditer(r"<\[([A-Z ]+)\]>(.*?)</\[\1\]>", ctx, re.S):
        spans[m.group(1)] = (m.start(), m.group(2))
    return spans


def _report_presence(ctx, needle, label):
    # Offsets are WITHIN-SECTION and labelled as such (G3 R1 m1: the
    # first version added a tag offset to a body-relative index and
    # reported a hybrid number that was neither).
    spans = _sections(ctx)
    hits = []
    for name, (start, body) in spans.items():
        if needle.lower() in body.lower():
            hits.append((name, body.lower().index(needle.lower())))
    if hits:
        first = min(hits, key=lambda h: h[1])
        where = (f" (first: {first[1]} chars into "
                 f"[{first[0]}])")
    else:
        where = ""
    print(f"  {label}: needle {needle!r} found in "
          f"{[h[0] for h in hits] or 'NOWHERE'}{where}")
    for name, (_, body) in spans.items():
        print(f"    section {name}: {len(body)} chars")
    return hits


def main():
    ds = json.load(open(HERE / "benchmark_cache/longmemeval_s.json"))
    mems = {m["mid"]: m for m in ds["memories"]}
    qs = [q for q in ds["queries"]
          if q.get("question_type") == "knowledge-update"
          and len(q["gold_keys"]) == 2]

    from agentmem_os.db.engine import get_session
    from agentmem_os.llm.context_assembler import ContextAssembler
    from benchmarks.real_code_utils import install_best_chroma

    backend = install_best_chroma(ContextAssembler)
    print(f"chunk backend: {backend!r} (benchmark champion; keeps the "
          f"run out of the live vector store)")
    print(f"scratch DB: {os.environ['AGENTMEM_OS_DB_PATH']}")

    # ── A. NO-REGRESS: empty fact store ⇒ byte-identical assembly ──────
    q = qs[22]  # the Rachel case — same demonstration case as Stage 4 G2
    sids = q["gold_keys"]
    for sid in sids:
        _ingest(get_session, mems, sid)
    assembler = ContextAssembler()
    with_facts = assembler.assemble(sids[0], q["question"])
    without = assembler.assemble(sids[0], q["question"],
                                 disable=frozenset({"facts"}))
    identical = with_facts == without
    print(f"\n== A. NO-REGRESS (empty fact store, real turns) ==\n"
          f"byte-identical: {identical} "
          f"({len(with_facts)} chars vs {len(without)} chars)")
    if not identical:
        import difflib
        print("  !! REGRESSION — facts tier changed output with zero facts")
        print("\n".join(list(difflib.unified_diff(
            without.split("\n"), with_facts.split("\n"),
            "facts-disabled", "facts-enabled", lineterm=""))[:40]))

    # ── B. Knowledge-update through the real pipeline, in order ────────
    from agentmem_os.llm.consolidation_v2 import ConsolidationV2
    print(f"\n== B. KNOWLEDGE-UPDATE (real extraction + judge) ==\n"
          f"question: {q['question'][:80]!r}  gold: {q['gold_answer']!r}")
    cv2 = ConsolidationV2(get_session)
    for sid in sids:
        r = cv2.consolidate_session(sid)
        sup = r["supersession"] or {}
        print(f"  {sid}: {r['created']} facts, "
              f"{r['entities_linked']} links, "
              f"superseded={sup.get('superseded', [])}, "
              f"judge_failure={r['judge_failure']}")

    # Full judgment transparency: what the judge saw, proposed, applied,
    # dropped — a diagnostic that hides the audit trail isn't one.
    from agentmem_os.db.models import SupersessionJudgment
    db = get_session()
    try:
        for j in db.query(SupersessionJudgment).all():
            if j.skipped:
                print(f"  judgment(fact {j.fact_id}): SKIP {j.skipped[:60]}")
                continue
            raw = json.loads(j.raw_output_json) if j.raw_output_json else {}
            print(f"  judgment(fact {j.fact_id}): "
                  f"proposed={raw.get('superseded_ids')}"
                  f"/{raw.get('cancelled_ids')} "
                  f"applied={j.applied_json} dropped="
                  f"{str(j.dropped_json)[:100]}")
            if raw.get("reasoning"):
                print(f"    reasoning: {raw['reasoning'][:140]!r}")
    finally:
        db.close()

    ctx_new = assembler.assemble(sids[0], q["question"])
    hits = _report_presence(ctx_new, q["gold_answer"], "with facts   ")
    ctx_off = assembler.assemble(sids[0], q["question"],
                                 disable=frozenset({"facts"}))
    _report_presence(ctx_off, q["gold_answer"], "facts disabled")
    print("  (the facts-disabled miss is STRUCTURAL — the answer text "
          "exists only in the other gold session; see Part C for the "
          "measurement of that axis)")
    in_facts = any(h[0] == "SEMANTIC FACTS" for h in hits)
    history = "[change history:" in ctx_new
    print(f"  current answer as dated fact: {in_facts}; "
          f"change history visible: {history}")

    # ── C. CROSS-SESSION REACH ─────────────────────────────────────────
    # A session that never discussed Rachel's new employer: per-session
    # chunk search cannot surface the answer there; the fact tier is
    # scope-wide. Use a NEUTRAL third session from the same dataset.
    neutral_sid = None
    for cand in q.get("scope_keys", []):
        if cand not in sids and cand in mems:
            neutral_sid = cand
            break
    print("\n== C. CROSS-SESSION REACH ==")
    if neutral_sid is None:
        print("  no neutral session available in cache — SKIPPED (loud)")
    else:
        _ingest(get_session, mems, neutral_sid)
        ctx_far = assembler.assemble(neutral_sid, q["question"])
        far_hits = _report_presence(ctx_far, q["gold_answer"],
                                    f"recall@{neutral_sid[:12]}")
        only_facts = (bool(far_hits)
                      and all(h[0] == "SEMANTIC FACTS" for h in far_hits))
        print(f"  answer reachable from an unrelated session: "
              f"{bool(far_hits)} (facts-only: {only_facts})")

    # ── D. BOUNDARY (disclosed): event-typed metric, judge excludes ────
    q5k = qs[0]
    print(f"\n== D. BOUNDARY — event-typed metric (disclosed exclusion) ==\n"
          f"question: {q5k['question'][:80]!r}  gold: {q5k['gold_answer']!r}")
    for sid in q5k["gold_keys"]:
        _ingest(get_session, mems, sid)
        r = cv2.consolidate_session(sid)
        sup = r["supersession"] or {}
        print(f"  {sid}: {r['created']} facts, "
              f"superseded={sup.get('superseded', [])}, "
              f"judge_failure={r['judge_failure']}")
    ctx_5k = assembler.assemble(q5k["gold_keys"][0], q5k["question"])
    # The gold string is a full sentence ("25 minutes and 50 seconds
    # (or 25:50)") that never appears verbatim anywhere — probe the
    # canonical token instead, and say so. Probe the OLD value too:
    # "both values stay visible, no false supersession" must be a
    # measurement, not an assertion (G3 R1 M5).
    needle_5k, needle_old = "25:50", "27:12"
    print(f"  (probing canonical tokens {needle_5k!r} new / "
          f"{needle_old!r} old — the gold sentence appears verbatim "
          f"nowhere)")
    _report_presence(ctx_5k, needle_5k, "new value    ")
    _report_presence(ctx_5k, needle_old, "old value    ")
    spans_5k = _sections(ctx_5k)
    facts_body = spans_5k.get("SEMANTIC FACTS", (0, ""))[1]
    metric_lines = [l for l in facts_body.split("\n")
                    if "5K" in l or needle_5k in l or needle_old in l]
    print("  metric-related fact lines, as rendered:")
    for l in metric_lines:
        print(f"    | {l[:150]}")
    print("  NOTE: events and identity facts are excluded from "
          "supersession BY DESIGN — both values may appear; "
          "chronological order puts the newest last.")

    print("\ncaps in play (disclosed): lexical scan 500 newest facts "
          "(undated facts rank via the entity floor only); 24 query "
          "surfaces; 100 facts/entity; TF-IDF floor 0.01; facts share "
          "the 20% semantic budget in BOTH units (tokens and the "
          "caller's 4x char fast path), chunks take the remainder")
    print(f"\nRESULT: no_regress_byte_identical={identical} "
          f"answer_as_fact={in_facts} history_visible={history}")
    if not identical:
        sys.exit(1)


if __name__ == "__main__":
    main()
