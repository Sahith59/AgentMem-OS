#!/usr/bin/env python3
"""Frozen, offline rank-before-chronology packet audit; no provider/model calls.

`generate` loads only question-local runtime material. It first reproduces every
legacy corrected and precision packet, then generates candidate packets. `evaluate`
reads annotations and historical grades only after all 500 contexts are complete.
Run only after the opt-in implementation and its tests are frozen.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
MEMORY = HERE.parents[1]
OUTER = MEMORY.parent
REPO = OUTER / "AgentMem-OS"
TOOLS = MEMORY / "tools"
RUNS = MEMORY / "runs" / "english-improvement-2026-09-11"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dump_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def inputs():
    return {
        "frozen_policy": HERE / "POLICY.md",
        "runner": HERE / "audit.py",
        "source_db": RUNS / "unified-english-all500-001/assembler.db",
        "assembler_policy": RUNS / "unified-english-all500-001/policy.json",
        "cache": RUNS / "integrity-audit-001/corrected-cache-v2/longmemeval_s.json",
        "runtime_projection": HERE / "runtime-projection.json",
        "corrected_package": RUNS / "full500-corrected-measurement-package-001/package.json",
        "precision_package": RUNS / "full500-precision-measurement-package-001/package.json",
        "upstream_evaluator_only": RUNS / "integrity-audit-001/longmemeval_s_cleaned.json",
        "paid_checkpoint_evaluator_only": RUNS / "paid-full500-precision-measurement1-001/checkpoint.json",
        "context_assembler_code": REPO / "llm/context_assembler.py",
        "token_counter_code": REPO / "llm/token_counter.py",
        "fact_retrieval_code": REPO / "llm/fact_retrieval.py",
        "packer_code": REPO / "llm/ranked_chunk_packing.py",
        "precision_supplement_code": REPO / "benchmarks/precise_source_supplement.py",
        "precision_ranking_code": REPO / "benchmarks/precise_lexical_retrieval.py",
        "dated_adapter_code": REPO / "benchmarks/dated_event_adapter.py",
        "recall_adapter_code": REPO / "benchmarks/recall_span_adapter.py",
        "scope_adapter_code": TOOLS / "corrected_question_scope_adapter.py",
        "scope_cache_code": TOOLS / "build_question_local_english_cache.py",
        "tfidf_adapter_code": REPO / "benchmarks/real_code_utils.py",
        "packer_tests": REPO / "tests/test_ranked_chunk_packing.py",
    }


def project():
    """Isolated projection step. It decodes legacy files, emits no label fields."""
    if (HERE / "runtime-projection.json").exists() or (HERE / "freeze.json").exists():
        raise FileExistsError("Refusing to overwrite projection or frozen run")
    cache = json.loads((RUNS / "integrity-audit-001/corrected-cache-v2/longmemeval_s.json").read_text())
    corrected = json.loads((RUNS / "full500-corrected-measurement-package-001/package.json").read_text())
    precision = json.loads((RUNS / "full500-precision-measurement-package-001/package.json").read_text())
    config = json.loads((RUNS / "unified-english-all500-001/policy.json").read_text())["configuration"]
    by_precision = {case["id"]: case for case in precision["cases"]}
    if len(corrected["cases"]) != len(by_precision) or len(by_precision) != 500:
        raise ValueError("Expected 500 matched corrected/precision cases")
    cases = []
    for case in corrected["cases"]:
        other = by_precision[case["id"]]
        if case["question"] != other["question"] or other["baseline_context_sha256"] != case["context_sha256"]:
            raise ValueError("Source package binding mismatch: " + case["id"])
        cases.append({"id": case["id"], "question": case["question"],
            "corrected_context": case["context"], "corrected_context_sha256": case["context_sha256"],
            "precision_context": other["context"], "precision_context_sha256": other["context_sha256"]})
    projection = {"schema": "rank-chronology-label-free-runtime-v1", "config": config,
        "source_sha256": {
            "cache": sha(RUNS / "integrity-audit-001/corrected-cache-v2/longmemeval_s.json"),
            "corrected_package": sha(RUNS / "full500-corrected-measurement-package-001/package.json"),
            "precision_package": sha(RUNS / "full500-precision-measurement-package-001/package.json"),
            "assembler_policy": sha(RUNS / "unified-english-all500-001/policy.json"),
        },
        "cases": cases,
        "queries": [{"question_id": q["question_id"], "question": q["question"],
                     "question_date": q["question_date"], "scope_keys": q["scope_keys"]}
                    for q in cache["queries"]],
        "memories": [{"mid": m["mid"], "turns": m["turns"]} for m in cache["memories"]]}
    if len(projection["queries"]) != 500:
        raise ValueError("Expected 500 query projections")
    dump_new(HERE / "runtime-projection.json", projection)


def freeze_receipt():
    paths = inputs()
    if any(not p.is_file() for p in paths.values()):
        raise FileNotFoundError([str(p) for p in paths.values() if not p.is_file()])
    receipt = {
        "schema": "rank-before-chronology-freeze-v1",
        "status": "FROZEN_BEFORE_CANDIDATE_OUTPUTS",
        "policy": "whole_rank_v1 opt-in; legacy default; reserve fallback",
        "cases": 500,
        "no_paid_or_embedding_calls": True,
        "runtime_labels_loaded": False,
        "inputs": {k: {"path": str(p), "sha256": sha(p)} for k, p in paths.items()},
    }
    dump_new(HERE / "freeze.json", receipt)
    return receipt


def check_freeze(freeze):
    for k, item in freeze["inputs"].items():
        if sha(inputs()[k]) != item["sha256"]:
            raise ValueError("Frozen input changed: " + k)


def assert_source_wal_empty():
    wal = Path(str(inputs()["source_db"]) + "-wal")
    if wal.exists() and wal.stat().st_size != 0:
        raise ValueError("Original source DB has nonempty WAL")


def deny_network():
    def deny(*_args, **_kwargs):
        raise RuntimeError("Network prohibited in offline rank-before-chronology audit")
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny


def load_repo():
    spec = importlib.util.spec_from_file_location(
        "agentmem_os", REPO / "__init__.py", submodule_search_locations=[str(REPO)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["agentmem_os"] = module
    spec.loader.exec_module(module)
    sys.path.insert(0, str(TOOLS))


def check_raw_delivery(assembled, section, report, cap, qid):
    if not section:
        if report.get("policy") == "whole_rank_v1" and report.get("used_chars") != 0:
            raise ValueError("Packing receipt has unrendered section: " + qid)
        return
    start = assembled.find(section)
    if start < 0 or assembled.find(section, start + 1) >= 0:
        raise ValueError("Ambiguous raw section: " + qid)
    if start + len(section) > cap:
        raise ValueError("Final cap truncates raw section: " + qid)
    if report.get("policy") == "whole_rank_v1":
        if report["section_sha256"] != digest(section) or report["used_chars"] != len(section):
            raise ValueError("Raw packing report mismatch: " + qid)
        if report["used_chars"] > report["char_budget"] or report["used_tokens"] > report["token_budget"]:
            raise ValueError("Raw packing exceeds budget: " + qid)
        for receipt in report["receipts"]:
            if digest(section[receipt["start"]:receipt["end"]]) != receipt["sha256"]:
                raise ValueError("Raw receipt mismatch: " + qid)


class TraceAdapter:
    """Record exact ranked chunks from the same search call the assembler used."""

    def __init__(self, inner):
        self.inner = inner
        self.last_receipt = None
        self.chunks = None

    def search(self, session_id, query, top_k=5):
        self.chunks = None
        self.last_receipt = None
        chunks = self.inner.search(session_id, query, top_k=top_k)
        self.chunks = list(chunks)
        self.last_receipt = copy.deepcopy(getattr(self.inner, "last_receipt", None))
        return chunks


def generate():
    if (HERE / "freeze.json").exists():
        raise FileExistsError("Refusing to overwrite prior freeze or run")
    assert_source_wal_empty()
    freeze = freeze_receipt()
    deny_network()
    projection = json.loads(inputs()["runtime_projection"].read_text())
    if projection["schema"] != "rank-chronology-label-free-runtime-v1":
        raise ValueError("Wrong runtime projection")
    for key, expected in projection["source_sha256"].items():
        if expected != freeze["inputs"][key]["sha256"]:
            raise ValueError("Runtime projection source binding differs: " + key)
    config = projection["config"]
    for key, value in config["environment"].items():
        os.environ[key] = value
    dbcopy = HERE / "disposable-assembler.db"
    shutil.copy2(inputs()["source_db"], dbcopy)
    if sha(dbcopy) != freeze["inputs"]["source_db"]["sha256"]:
        raise ValueError("Copied DB hash mismatch")
    os.environ["AGENTMEM_OS_DB_PATH"] = str(dbcopy)
    load_repo()
    from corrected_question_scope_adapter import install
    from agentmem_os.benchmarks.dated_event_adapter import DatedEventTfIdfAdapter
    from agentmem_os.benchmarks.recall_span_adapter import RecallSpanContextAssembler, RecallSpanTfIdfAdapter
    from agentmem_os.benchmarks.real_code_utils import TfIdfChromaAdapter
    from agentmem_os.benchmarks.precise_source_supplement import supplement_packet
    from agentmem_os.storage.store import ConversationStore

    class TrackingAssembler(RecallSpanContextAssembler):
        """Observe the exact returned raw section without reparsing source text."""

        def assemble(self, *args, **kwargs):
            self.audit_raw_section = ""
            self.audit_raw_budget = None
            self.audit_chronological = None
            return super().assemble(*args, **kwargs)

        def _render_raw_evidence(self, chunks, token_budget, chronological=True):
            section = super()._render_raw_evidence(
                chunks, token_budget, chronological=chronological)
            self.audit_raw_section = section
            self.audit_raw_budget = token_budget
            self.audit_chronological = chronological
            return section

    corrected = projection
    precision = projection
    if len(corrected["cases"]) != 500 or len(precision["cases"]) != 500:
        raise ValueError("Expected 500 cases in both frozen packages")
    by_precision = {c["id"]: c for c in precision["cases"]}
    if len(by_precision) != 500 or {c["id"] for c in corrected["cases"]} != set(by_precision):
        raise ValueError("Corrected/precision case identity differs")
    memories = {m["mid"]: m for m in projection["memories"]}
    queries = {q["question_id"]: q for q in projection["queries"]}
    if len(queries) != 500:
        raise ValueError("Expected 500 corrected-cache questions")
    groups = {
        "unified-all500-" + qid: [memories[k]["turns"] for k in q["scope_keys"]]
        for qid, q in queries.items()
    }
    scope = {q["question"]: "lme-question-" + qid for qid, q in queries.items()}
    dates = {q["question"]: q["question_date"] for q in queries.values()}
    assembler = TrackingAssembler()
    assembler.reserve_budget_share = config["recall_reserve_budget_share"]
    assembler.allocations["semantic"] = config["semantic_tokens"]
    assembler.allocations["recent"] = config["recent_tokens"]
    install(assembler, scope, dbcopy, lexical_method="char_wb")
    assembler._store = ConversationStore()
    dated = DatedEventTfIdfAdapter(
        dates, reserve_limit=config["dated_reserve_limit"],
        ordered_music_limit=config["ordered_music_limit"], base=TfIdfChromaAdapter(),
        turn_loader=lambda sid: [t for g in groups[sid] for t in g])
    wrapped = TraceAdapter(RecallSpanTfIdfAdapter(
        groups, session_limit=config["recall_session_limit"],
        min_similarity=config["recall_min_similarity"],
        group_similarity_weight=config["recall_group_similarity_weight"],
        max_chars_per_turn=config["recall_max_chars_per_turn"], base=dated))
    assembler._chroma = wrapped
    context_dir = HERE / "contexts"
    context_dir.mkdir(exist_ok=False)
    (context_dir / "legacy").mkdir()
    (context_dir / "candidate").mkdir()
    legacy_rows = []
    started = time.monotonic()

    # Complete the historical exactness gate before emitting any candidate.
    for index, case in enumerate(corrected["cases"], 1):
        qid = case["id"]
        q = queries[qid]
        expected = by_precision[qid]
        if q["question"] != case["question"] or q["question"] != expected["question"]:
            raise ValueError("Question binding mismatch: " + qid)
        session_id = "unified-all500-" + qid
        wrapped.chunks = None
        wrapped.last_receipt = None
        assembled = assembler.assemble(
            session_id, q["question"], disable=frozenset(config["disabled_tiers"]))
        legacy = assembled[:config["context_char_cap"]]
        if legacy != case["corrected_context"] or digest(legacy) != case["corrected_context_sha256"]:
            raise ValueError("Corrected baseline mismatch: " + qid)
        turns = [t for group in groups[session_id] for t in group]
        final, receipts = supplement_packet(legacy, turns, q["question"])
        if final != expected["precision_context"] or digest(final) != expected["precision_context_sha256"]:
            raise ValueError("Precision baseline mismatch: " + qid)
        legacy_section = assembler.audit_raw_section
        check_raw_delivery(assembled, legacy_section, assembler.last_raw_evidence_packing,
                           config["context_char_cap"], qid)
        (context_dir / "legacy" / (qid + ".txt")).write_text(final)
        legacy_rows.append({
            "id": qid, "question_sha256": digest(q["question"]),
            "raw_ranked_chunks": wrapped.chunks,
            "adapter_receipt": wrapped.last_receipt,
            "raw_section": legacy_section,
            "raw_budget_tokens": assembler.audit_raw_budget,
            "raw_chronological": assembler.audit_chronological,
            "raw_packing_receipt": copy.deepcopy(assembler.last_raw_evidence_packing),
            "assembled_text": assembled,
            "assembled_sha256": digest(assembled),
            "corrected_context_sha256": digest(legacy),
            "final_context_sha256": digest(final),
            "precision_receipts": receipts,
        })
        if index % 50 == 0:
            print(json.dumps({"legacy_exact": index, "seconds": round(time.monotonic()-started, 1)}), flush=True)
    with (HERE / "legacy-traces.jsonl").open("x") as stream:
        for row in legacy_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    dump_new(HERE / "legacy-exact.json", {"cases": len(legacy_rows), "corrected_exact": 500,
        "precision_exact": 500, "trace_sha256": sha(HERE / "legacy-traces.jsonl")})

    candidate_assembler = TrackingAssembler(raw_evidence_policy="whole_rank_v1")
    candidate_assembler.reserve_budget_share = config["recall_reserve_budget_share"]
    candidate_assembler.allocations["semantic"] = config["semantic_tokens"]
    candidate_assembler.allocations["recent"] = config["recent_tokens"]
    install(candidate_assembler, scope, dbcopy, lexical_method="char_wb")
    candidate_assembler._store = ConversationStore()
    candidate_assembler._chroma = wrapped
    candidate_rows = []
    for index, case in enumerate(corrected["cases"], 1):
        qid = case["id"]
        q = queries[qid]
        session_id = "unified-all500-" + qid
        wrapped.chunks = None
        wrapped.last_receipt = None
        assembled = candidate_assembler.assemble(
            session_id, q["question"], disable=frozenset(config["disabled_tiers"]))
        if wrapped.chunks != legacy_rows[index-1]["raw_ranked_chunks"]:
            raise ValueError("Ranked raw inputs changed between arms: " + qid)
        if wrapped.last_receipt != legacy_rows[index-1]["adapter_receipt"]:
            raise ValueError("Adapter reserve receipt changed between arms: " + qid)
        if wrapped.chunks and candidate_assembler.audit_raw_budget is None:
            raise ValueError("Raw packing silently skipped after retrieval: " + qid)
        candidate_section = candidate_assembler.audit_raw_section
        packing = candidate_assembler.last_raw_evidence_packing
        if packing.get("policy") == "whole_rank_v1":
            if packing["token_budget"] != candidate_assembler.audit_raw_budget:
                raise ValueError("Raw token budget receipt mismatch: " + qid)
            if packing["input_sha256"] != [digest(c) for c in wrapped.chunks]:
                raise ValueError("Raw ranked input hash receipt mismatch: " + qid)
        check_raw_delivery(assembled, candidate_section,
                           packing,
                           config["context_char_cap"], qid)
        if wrapped.last_receipt and wrapped.last_receipt.get("reserve"):
            if assembled != legacy_rows[index-1]["assembled_text"] or candidate_assembler.last_raw_evidence_packing.get("reason") != "reserve_fallback":
                raise ValueError("Reserve fallback differs from legacy: " + qid)
        candidate = assembled[:config["context_char_cap"]]
        turns = [t for group in groups[session_id] for t in group]
        final, receipts = supplement_packet(candidate, turns, q["question"])
        if len(final) > 40_000:
            raise ValueError("Invalid candidate packet or missing trace: " + qid)
        (context_dir / "candidate" / (qid + ".txt")).write_text(final)
        candidate_rows.append({
            "id": qid, "question_sha256": digest(q["question"]),
            "raw_ranked_chunks": wrapped.chunks,
            "adapter_receipt": wrapped.last_receipt,
            "raw_section": candidate_section,
            "raw_budget_tokens": candidate_assembler.audit_raw_budget,
            "raw_chronological": candidate_assembler.audit_chronological,
            "raw_packing_receipt": copy.deepcopy(packing),
            "assembled_text": assembled,
            "assembled_sha256": digest(assembled),
            "pre_supplement_context_sha256": digest(candidate),
            "final_context_sha256": digest(final),
            "precision_receipts": receipts,
            "legacy_pre_supplement_sha256": case["corrected_context_sha256"],
            "legacy_final_sha256": by_precision[qid]["precision_context_sha256"],
            "pre_supplement_changed": candidate != case["corrected_context"],
            "final_changed": final != by_precision[qid]["precision_context"],
            "supplement_changed": [(r["role"],r["source_sha256"]) for r in receipts]
                != [(r["role"],r["source_sha256"]) for r in legacy_rows[index-1]["precision_receipts"]],
        })
        if index % 50 == 0:
            print(json.dumps({"candidate_complete": index, "seconds": round(time.monotonic()-started, 1)}), flush=True)
    with (HERE / "candidate-traces.jsonl").open("x") as stream:
        for row in candidate_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    assembler._corrected_fact_engine.dispose()
    candidate_assembler._corrected_fact_engine.dispose()
    assert_source_wal_empty()
    check_freeze(freeze)
    dump_new(HERE / "phase1.json", {
        "status": "PASS_500_CONTEXTS_NO_LABELS_LOADED",
        "legacy_exact": 500,
        "candidate_complete": 500,
        "legacy_trace_sha256": sha(HERE / "legacy-traces.jsonl"),
        "candidate_trace_sha256": sha(HERE / "candidate-traces.jsonl"),
        "pre_supplement_changed": sum(r["pre_supplement_changed"] for r in candidate_rows),
        "final_changed": sum(r["final_changed"] for r in candidate_rows),
        "supplement_changed": sum(r["supplement_changed"] for r in candidate_rows),
        "db_copy_sha256_after_run": sha(dbcopy),
        "source_db_sha256_after_run": sha(inputs()["source_db"]),
        "freeze_sha256": sha(HERE / "freeze.json"),
        "paid_calls": 0, "new_answers": 0, "new_embeddings": 0,
    })


def evaluate():
    phase1 = json.loads((HERE / "phase1.json").read_text())
    freeze = json.loads((HERE / "freeze.json").read_text())
    check_freeze(freeze)
    if phase1["status"] != "PASS_500_CONTEXTS_NO_LABELS_LOADED":
        raise ValueError("Incomplete phase1")
    if sha(HERE / "legacy-traces.jsonl") != phase1["legacy_trace_sha256"] or sha(HERE / "candidate-traces.jsonl") != phase1["candidate_trace_sha256"]:
        raise ValueError("Trace changed after generation")
    sys.path.insert(0, str(TOOLS))
    from build_question_local_english_cache import identity, render
    cache = json.loads(inputs()["cache"].read_text())
    upstream = {q["question_id"]: q for q in json.loads(inputs()["upstream_evaluator_only"].read_text())}
    corrected = {c["id"]: c for c in json.loads(inputs()["corrected_package"].read_text())["cases"]}
    precision = {c["id"]: c for c in json.loads(inputs()["precision_package"].read_text())["cases"]}
    checkpoint = json.loads(inputs()["paid_checkpoint_evaluator_only"].read_text())
    memories = {m["mid"]: m for m in cache["memories"]}
    queries = {q["question_id"]: q for q in cache["queries"]}
    candidate_traces = [json.loads(s) for s in (HERE / "candidate-traces.jsonl").read_text().splitlines()]
    legacy_traces = [json.loads(s) for s in (HERE / "legacy-traces.jsonl").read_text().splitlines()]
    if len(candidate_traces) != 500 or len(legacy_traces) != 500:
        raise ValueError("Trace count mismatch")
    rows = []
    for control, candidate in zip(legacy_traces, candidate_traces, strict=True):
        qid = control["id"]
        if qid != candidate["id"]:
            raise ValueError("Candidate/control order differs")
        original = upstream[qid]
        query = queries[qid]
        baseline = precision[qid]["context"]
        control_text = (HERE / "contexts" / "legacy" / (qid + ".txt")).read_text()
        candidate_text = (HERE / "contexts" / "candidate" / (qid + ".txt")).read_text()
        if control_text != baseline or digest(candidate_text) != candidate["final_context_sha256"]:
            raise ValueError("Final context binding differs: " + qid)
        grade = checkpoint["jobs"][qid + "/judge"]
        if grade["status"] != "complete" or type(grade["correct"]) is not bool:
            raise ValueError("Missing historical grade: " + qid)
        annotations = []
        for sid, date, turns in zip(original["haystack_session_ids"], original["haystack_dates"], original["haystack_sessions"], strict=True):
            key = identity(sid, date, turns)
            _, stamped = render(turns, date)
            if key not in query["scope_keys"] or memories[key]["turns"] != stamped:
                raise ValueError("Source lineage mismatch: " + qid)
            for position, (turn, source) in enumerate(zip(turns, stamped, strict=True)):
                if turn.get("has_answer") is not True:
                    continue
                text = source["content"]
                annotations.append({
                    "session_id": sid, "source_key": key, "position": position,
                    "role": turn["role"], "text_sha256": digest(text),
                    "legacy_full_body": text in control_text,
                    "candidate_full_body": text in candidate_text,
                })
        gains = [a for a in annotations if not a["legacy_full_body"] and a["candidate_full_body"]]
        losses = [a for a in annotations if a["legacy_full_body"] and not a["candidate_full_body"]]
        rows.append({
            "id": qid, "historically_correct": grade["correct"],
            "abstention": precision[qid]["abst"], "question_type": precision[qid]["type"],
            "legacy_context_sha256": digest(control_text),
            "candidate_context_sha256": digest(candidate_text),
            "pre_supplement_changed": candidate["pre_supplement_changed"],
            "supplement_changed": candidate["supplement_changed"],
            "annotations": annotations, "gained_count": len(gains), "lost_count": len(losses),
        })
    def cohort(group):
        return {"cases": len(group),
            "gain_cases": sum(r["gained_count"] > 0 for r in group),
            "loss_cases": sum(r["lost_count"] > 0 for r in group),
            "gained_turns": sum(r["gained_count"] for r in group),
            "lost_turns": sum(r["lost_count"] for r in group)}
    misses = [r for r in rows if not r["historically_correct"]]
    correct = [r for r in rows if r["historically_correct"]]
    allstats = cohort(rows); missstats = cohort(misses)
    gate = {
        "historical_miss_gain_cases_at_least_10": missstats["gain_cases"] >= 10,
        "historical_miss_net_source_cases_at_least_5": missstats["gain_cases"] - missstats["loss_cases"] >= 5,
        "no_annotated_losses_all500": allstats["loss_cases"] == 0,
    }
    dump_new(HERE / "audit.json", {
        "schema": "rank-before-chronology-offline-source-audit-v1",
        "status": "PASS_SOURCE_TRIAGE" if all(gate.values()) else "FAIL_SOURCE_TRIAGE",
        "freeze_sha256": sha(HERE / "freeze.json"), "phase1_sha256": sha(HERE / "phase1.json"),
        "cohorts": {"all500": allstats, "historical_miss": missstats,
            "historical_correct": cohort(correct), "abstention": cohort([r for r in rows if r["abstention"]])},
        "gate": gate, "semantic_completeness": "NOT_CERTIFIED",
        "answer_accuracy": "NOT_MEASURED", "paid_calls": 0, "new_answers": 0,
        "limits": ["Annotated complete-body presence is not semantic sufficiency or source identity.",
            "Precision supplement may change because the assembled baseline and remaining room changed; any apparent gain needs stage-specific tracing and source review.",
            "All 500 questions are development-exposed; no hidden accuracy claim."],
        "rows": rows,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("project", "generate", "evaluate"))
    args = parser.parse_args()
    if args.phase == "project":
        project()
    elif args.phase == "generate":
        generate()
    else:
        evaluate()
