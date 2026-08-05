#!/usr/bin/env python3
"""
Real baseline evaluation — AgentMem OS vs. Mem0 vs. Graphiti vs. Letta vs.
LangMem vs. a trivial recent-only floor, scored with the same QA-accuracy
methodology Mem0/Zep publish (retrieve -> generate -> judge), on real
LoCoMo/LongMemEval data. Supersedes every proxy simulation in
benchmarks/deprecated_proxy_sim/ — every number this script produces comes
from that system's own installed library actually running its own
extraction and retrieval, not a hand-rolled simulation of one.
See LAUNCH_ROADMAP.md Phase 2 for the full design rationale.

Pipeline per (system, question):
  1. INGEST  — each unique haystack scope is ingested exactly once per
     system, via that system's own adapter (real extraction: mem0's LLM
     fact extraction, Graphiti's temporal-KG construction, Letta's
     archival writes, LangMem's memory-store-manager extraction,
     AgentMem OS's own KG/procedural/semantic tiers).
  2. RETRIEVE — adapter.retrieve() over that question's own haystack scope.
  3. GENERATE — an answer from the retrieved context (GPT-4o-mini).
  4. JUDGE — the answer vs. the gold answer (GPT-4o -> correct/incorrect).
  QA accuracy = % judged correct per system, directly comparable to Mem0's
  published 91.6-92.5 (LoCoMo) / 94.4-94.8 (LongMemEval) — same generator
  model, same judge model, same metric family.

Deliberately uses OpenAI (not Claude) for generator + judge, matching
Mem0's own published methodology and avoiding AgentMem OS's own model
(Claude) judging its own answers — see qa_accuracy_eval.py's docstring,
which this script mirrors structurally.

adapter_disclosures: known scoping limitations that travel with every
result artifact instead of being silently absorbed into the headline
number (LAUNCH_ROADMAP.md Phase 2, Definition of Done #5) — e.g. Letta's
archival-only scoping, recent_only's trivial-floor purpose.

Cost: REAL MONEY for every system except recent_only. Ingestion runs each
system's own real extraction (mem0/graphiti/langmem all make LLM calls
per turn; AgentMem OS makes Haiku-based conflict-detection calls; Letta
needs a real embedding provider for archival writes). ALWAYS run
--dry-run-cost first — see LAUNCH_ROADMAP.md Phase 2 §4 for the existing
order-of-magnitude estimates (~$0.35-0.70/system ingestion for LoCoMo,
~$0.001/question/system for generate+judge) and the $200 circuit-breaker.

Usage:
    python3 benchmarks/real_baseline_eval.py --dataset locomo --systems all --dry-run-cost
    python3 benchmarks/real_baseline_eval.py --dataset locomo --systems agentmem_os,mem0 --n 20

Output: benchmarks/reports/real_baseline_results_<dataset>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lme_judge import build_judge_prompt, parse_judge_verdict, is_abstention  # noqa: E402
from corpus_loaders import load_locomo, load_longmemeval  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from benchmarks.adapters.registry import ADAPTERS, get_adapter  # noqa: E402

for _p in [REPO_ROOT, REPO_ROOT.parent]:
    _env = _p / ".env"
    if _env.exists():
        for line in _env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
        break

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["locomo", "longmemeval"], required=True)
ap.add_argument("--systems", default="all",
                 help=f"comma-separated system names ({sorted(ADAPTERS)}) or 'all'")
ap.add_argument("--n", type=int, default=20, help="questions to sample — START SMALL, this costs real money")
ap.add_argument("--gen-model", default="gpt-4o-mini")
ap.add_argument("--judge-model", default="gpt-4o")
ap.add_argument("--top-k", type=int, default=10)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--lme-split", choices=["oracle", "s"], default="oracle",
                 help="LongMemEval haystack: oracle (evidence sessions only) or s "
                      "(~40 sessions/question, the split vendor numbers use)")
ap.add_argument("--answerer", choices=["reasoning", "simple"], default="reasoning",
                 help="answer layer applied identically to EVERY system")
ap.add_argument("--dry-run-cost", action="store_true",
                 help="ingest/retrieve/generate/judge on just 3 questions per system, "
                      "sum actual generate+judge token usage, project full-run cost, "
                      "then exit WITHOUT running the full dataset")
args = ap.parse_args()

try:
    import openai
except ImportError:
    print("openai package not installed. Run: pip install openai")
    sys.exit(1)

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("OPENAI_API_KEY not found in environment or .env. "
          "This harness deliberately uses OpenAI (not Claude) for the "
          "generator+judge, and most adapters need it for their own "
          "extraction too — see module docstring.")
    sys.exit(1)
client = openai.OpenAI(api_key=api_key)

SYSTEMS = sorted(ADAPTERS) if args.systems == "all" else [s.strip() for s in args.systems.split(",")]
for s in SYSTEMS:
    if s not in ADAPTERS:
        print(f"Unknown system '{s}'. Registered: {sorted(ADAPTERS)}")
        sys.exit(1)

# Known scoping limitations that must travel with every result artifact
# instead of being silently absorbed into the headline number (Phase 2 DoD #5).
ADAPTER_DISCLOSURES = {
    "letta": ("Archival-memory-only scoping — ingest_session writes directly into "
              "Letta's archival memory (passages.create), deliberately bypassing "
              "Letta's own conversational agent loop. Not a full-fidelity "
              "Letta agent-loop comparison; isolates archival retrieval quality only."),
    "recent_only": ("Trivial floor baseline — no retrieval logic at all, returns the "
                     "literal most-recent turns regardless of query relevance. Exists "
                     "to sanity-check the harness: if this ever beats a real memory "
                     "system, something in the harness is broken, not the memory system."),
}

SIMPLE_PROMPT = """You answer a question using ONLY the memories provided. Be concise — answer in as few words as possible (a name, date, number, or short phrase). If the memories do not contain the answer, reply "I don't know".

Memories:
{context}

Question: {question}
Answer:"""

# Date-anchored chain-of-thought answer layer, identical to
# qa_accuracy_eval.py's. Measured here: LongMemEval 0.467 -> 0.533 on the
# same 30 questions and the same retrieval. Applied identically to EVERY
# system in the comparison — an answer layer that only one system gets is
# not a memory-system benchmark, it's a prompt benchmark.
REASONING_PROMPT = """You answer a question about a user's own past conversations. You are given memories from those conversations, some tagged with dates.{today_line}

Memories:
{context}

Question: {question}

Reason carefully before answering:
- Find the specific fact(s) in the memories that bear on the question.
- DATES / DURATIONS ("how many days ago", "how long since", "between X and Y", "most recent"): locate the exact date(s) and compute the difference, counting carefully.
- COUNTS / TOTALS ("how many", "total", "in total"): find EVERY relevant item across ALL memories and add them up. Do not stop at the first one.
- UPDATES ("currently", "now", "most recently", "switched"): prefer the latest-dated fact over earlier ones.
- If the memories genuinely do not contain the information, do not guess — say it was not mentioned.

Think step by step, then end with exactly one final line starting with "ANSWER: ".
- For factual questions (who/what/when/where/how many): the ANSWER line is the shortest possible answer — a name, number, date, or short phrase; or "not mentioned" if the memories do not contain it.
- If the question asks for advice, suggestions, recommendations, or ideas: the ANSWER line is one or two sentences that respond helpfully and make specific use of what the memories say about the user — their preferences, past activities, possessions, and plans. Never answer "not mentioned" to an advice question; use whatever relevant user context the memories contain."""

JUDGE_PROMPT = """You are grading whether a predicted answer to a question is correct, given the gold answer. Be lenient about phrasing, formatting, and extra words — judge only whether the predicted answer conveys the same factual information as the gold answer.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Is the predicted answer correct? Reply with exactly one word: CORRECT or INCORRECT."""

_cost_lock = threading.Lock()
_cost_totals = {"prompt_tokens": 0, "completion_tokens": 0}
# Rough $/1K token pricing for cost projection only (dry-run-cost) — not
# used for the real report, which records raw token counts instead.
_PRICE_PER_1K = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
}



# OpenAI TPM limits are per-organization and low on standard accounts (measured
# on this one: gpt-4o = 30,000 TPM, gpt-4o-mini = 200,000 TPM). A 6k-token
# answerer call x 4 workers bursts past 30k instantly. Without backoff, calls
# fail, questions are silently dropped, and the harness reports a confident
# score computed from whatever survived — a 150-question run once finished with
# 4 answers and printed an "oracle ceiling" from them. Retry, and make the
# completion rate impossible to miss.
_MODEL_TPM = {"gpt-4o": 30_000, "gpt-4o-mini": 200_000}


def safe_workers(model: str, requested: int) -> int:
    """Worker count that won't burst past the model's TPM limit."""
    if _MODEL_TPM.get(model, 200_000) <= 30_000:
        return min(requested, 2)
    return requested


def _chat_with_retry(client, model, prompt, max_tokens, tries=8):
    import time as _t
    import openai as _o
    last = None
    for attempt in range(tries):
        try:
            return client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
        except _o.RateLimitError as e:
            last = e
            _t.sleep(min(2 ** attempt, 60))
        except Exception as e:
            last = e
            _t.sleep(2)
    raise RuntimeError(f"{model}: gave up after {tries} attempts ({last})")


def gen_answer(context: str, question: str, today: str = "") -> str:
    reasoning = args.answerer == "reasoning"
    if reasoning:
        prompt = REASONING_PROMPT.format(
            context=context[:24000], question=question,
            today_line=f"\nToday's date is {today}." if today else "")
    else:
        prompt = SIMPLE_PROMPT.format(context=context[:24000], question=question)
    r = _chat_with_retry(client, args.gen_model, prompt, 700 if reasoning else 80)
    with _cost_lock:
        _cost_totals["prompt_tokens"] += r.usage.prompt_tokens
        _cost_totals["completion_tokens"] += r.usage.completion_tokens
    out = (r.choices[0].message.content or "").strip()
    if reasoning:
        # Score the final ANSWER line only — never the chain-of-thought.
        m = re.search(r"ANSWER:\s*(.+)", out, re.IGNORECASE | re.DOTALL)
        out = (m.group(1).strip() if m else out).split("\n")[0].strip()
    return out


def judge(question: str, gold: str, pred: str, qtype: str = "", qid: str = "") -> bool:
    r = _chat_with_retry(client, args.judge_model, build_judge_prompt(
        qtype, question, gold, pred, abstention=is_abstention(qid)), 5)
    with _cost_lock:
        _cost_totals["prompt_tokens"] += r.usage.prompt_tokens
        _cost_totals["completion_tokens"] += r.usage.completion_tokens
    return parse_judge_verdict(r.choices[0].message.content or "")


def _scope_session_id(system: str, scope_keys: list) -> str:
    key = "|".join(sorted(scope_keys))
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    return f"{args.dataset}-{system}-scope-{h}"


def ensure_scope_ingested(adapter, mem_by_id: dict, ingested: set, lock: threading.Lock,
                           system: str, scope_keys: list) -> str:
    sid = _scope_session_id(system, scope_keys)
    with lock:
        if sid in ingested:
            return sid
        ingested.add(sid)
    adapter.reset(sid)
    for mkey in scope_keys:
        mem = mem_by_id.get(mkey)
        if not mem or not mem.turns:
            continue
        try:
            adapter.ingest_session(sid, mkey, mem.turns)
        except Exception as e:
            print(f"  [warn] {system}: ingest failed for scope member {mkey}: {e}")
    return sid


_retrieve_lock = threading.Lock()


def run_one(adapter, mem_by_id: dict, ingested: set, lock: threading.Lock, system: str, it) -> dict:
    sid = ensure_scope_ingested(adapter, mem_by_id, ingested, lock, system, it.scope_keys)
    try:
        # Serialized: adapters holding a single DB session (agentmem_os)
        # are not safe for concurrent retrieval; the slow part — the two
        # OpenAI calls below — stays parallel.
        with _retrieve_lock:
            retrieved = adapter.retrieve(sid, it.question, top_k=args.top_k)
    except Exception as e:
        return {"question": it.question, "gold_answer": it.gold_answer,
                "predicted": "", "correct": False, "error": f"retrieve failed: {e}"}
    context = "\n".join(str(r) for r in retrieved)
    pred = gen_answer(context, it.question, getattr(it, "question_date", "")) if context.strip() else "I don't know"
    ok = judge(it.question, it.gold_answer, pred,
               getattr(it, "question_type", ""), getattr(it, "question_id", ""))
    return {"question": it.question, "gold_answer": it.gold_answer,
            "predicted": pred, "correct": ok}


def run_system(system: str, items: list, mem_by_id: dict, out_path: Path) -> dict:
    print(f"\n{'=' * 60}\n  SYSTEM: {system}\n{'=' * 60}")
    adapter = get_adapter(system)
    adapter.setup()
    ingested: set = set()
    lock = threading.Lock()

    done_results = []
    done_questions = set()
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            done_results = prev.get("results", [])
            done_questions = {r["question"] for r in done_results}
            if done_results:
                print(f"  Resuming: {len(done_results)} questions already judged in {out_path.name}")
        except Exception:
            pass

    todo = [it for it in items if it.question not in done_questions]
    results = list(done_results)
    correct = sum(1 for r in done_results if r["correct"])
    done = len(done_results)
    result_lock = threading.Lock()

    def _checkpoint():
        out_path.write_text(json.dumps({
            "system": system, "dataset": args.dataset,
            "gen_model": args.gen_model, "judge_model": args.judge_model,
            "n_questions": len(items), "answerer": args.answerer,
            "qa_accuracy": round(correct / max(1, done), 4),
            "correct": correct, "total": done,
            "disclosures": ADAPTER_DISCLOSURES.get(system, ""),
            "results": results,
        }, indent=2))

    try:
        if todo:
            # Serial pre-ingestion — same thread-safety bug class caught
            # live in qa_accuracy_eval.py (see its comment): single-session
            # adapters can't be ingested from racing workers, and a worker
            # could otherwise retrieve from a scope another worker was
            # still halfway through ingesting.
            unique_scopes = []
            seen_scopes = set()
            for it in todo:
                s = _scope_session_id(system, it.scope_keys)
                if s not in seen_scopes:
                    seen_scopes.add(s)
                    unique_scopes.append(it.scope_keys)
            print(f"  Ingesting {len(unique_scopes)} unique scopes (serial)...")
            for i, sk in enumerate(unique_scopes, 1):
                ensure_scope_ingested(adapter, mem_by_id, ingested, lock, system, sk)
                if i % 5 == 0 or i == len(unique_scopes):
                    print(f"    {i}/{len(unique_scopes)} scopes", flush=True)
            with ThreadPoolExecutor(max_workers=safe_workers(args.gen_model, args.workers)) as pool:
                futs = {pool.submit(run_one, adapter, mem_by_id, ingested, lock, system, it): it for it in todo}
                for f in as_completed(futs):
                    try:
                        r = f.result()
                    except Exception as e:
                        it = futs[f]
                        print(f"  [error] question failed, skipping: {it.question[:50]}... ({e})")
                        continue
                    with result_lock:
                        results.append(r)
                        correct += int(r["correct"])
                        done += 1
                        if done % 10 == 0 or done == len(items):
                            print(f"  {done}/{len(items)}  running QA acc={correct/done:.3f}", flush=True)
                            _checkpoint()
        _checkpoint()
    finally:
        adapter.teardown()

    print(f"  {system}: QA accuracy = {correct/max(1,done):.3f}  ({correct}/{done} correct)")
    return {"system": system, "qa_accuracy": round(correct / max(1, done), 4),
            "correct": correct, "total": done, "disclosures": ADAPTER_DISCLOSURES.get(system, "")}


def run_dry_run_cost(items: list, mem_by_id: dict):
    print(f"\n{'=' * 60}\n  DRY RUN — cost projection (3 questions/system)\n{'=' * 60}")
    sample = items[:3]
    projections = {}
    for system in SYSTEMS:
        adapter = get_adapter(system)
        t0 = time.time()
        adapter.setup()
        ingested: set = set()
        lock = threading.Lock()
        _cost_totals["prompt_tokens"] = 0
        _cost_totals["completion_tokens"] = 0
        for it in sample:
            run_one(adapter, mem_by_id, ingested, lock, system, it)
        adapter.teardown()
        elapsed = time.time() - t0

        gen_price = _PRICE_PER_1K.get(args.gen_model, {"prompt": 0, "completion": 0})
        judge_price = _PRICE_PER_1K.get(args.judge_model, {"prompt": 0, "completion": 0})
        # Rough split: gen+judge share the same running totals here, so
        # price at gen_model rate for prompt tokens and judge_model rate
        # for completion tokens is not exact — good enough for an
        # order-of-magnitude pre-commit check, not a billing statement.
        genjudge_cost = (
            _cost_totals["prompt_tokens"] / 1000 * gen_price["prompt"]
            + _cost_totals["completion_tokens"] / 1000 * judge_price["completion"]
        )
        per_question = genjudge_cost / max(1, len(sample))
        projected_full = per_question * len(items)
        projections[system] = {
            "sample_questions": len(sample),
            "elapsed_sec": round(elapsed, 1),
            "genjudge_cost_sample": round(genjudge_cost, 4),
            "genjudge_cost_projected_full_run": round(projected_full, 2),
            "note": ("Generate+judge cost only — does NOT include each system's own "
                     "extraction/embedding cost during ingest_session, which this script "
                     "doesn't instrument per-library. See LAUNCH_ROADMAP.md Phase 2 §4 "
                     "for existing order-of-magnitude ingestion estimates "
                     "(~$0.35-0.70/system for LoCoMo)."),
        }
        print(f"  {system}: {len(sample)}q in {elapsed:.1f}s, "
              f"genjudge ${genjudge_cost:.4f} sample -> ${projected_full:.2f} projected full run "
              f"({len(items)}q)")

    print(f"\nProjection covers generate+judge only. Add each system's own ingestion/"
          f"extraction cost (order-of-magnitude estimates in LAUNCH_ROADMAP.md Phase 2 §4) "
          f"before deciding on a full run. $200 total is the documented circuit-breaker.")
    return projections


def main():
    print(f"Loading {args.dataset}...")
    ds = load_locomo(n_queries=args.n, seed=args.seed) if args.dataset == "locomo" \
        else load_longmemeval(n_queries=args.n, seed=args.seed, split=args.lme_split)

    mem_by_id = {m.mid: m for m in ds.memories}
    items = [q for q in ds.queries if q.gold_answer and q.scope_keys]
    print(f"real_baseline_eval: {args.dataset}, {len(items)} questions "
          f"(of {len(ds.queries)} sampled — some lack a scope or answer), "
          f"systems={SYSTEMS}, gen={args.gen_model}, judge={args.judge_model}")

    if args.dry_run_cost:
        run_dry_run_cost(items, mem_by_id)
        return

    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    per_system_reports = {}
    for system in SYSTEMS:
        out_path = reports_dir / f"real_baseline_{args.dataset}_{system}.json"
        per_system_reports[system] = run_system(system, items, mem_by_id, out_path)

    combined_path = reports_dir / f"real_baseline_results_{args.dataset}.json"
    combined_path.write_text(json.dumps({
        "dataset": args.dataset, "n_questions": len(items),
        "gen_model": args.gen_model, "judge_model": args.judge_model,
        "systems": per_system_reports,
        "comparable_to": ("Mem0 LoCoMo 91.6-92.5" if args.dataset == "locomo"
                           else "Mem0 LongMemEval 94.4-94.8"),
    }, indent=2))

    print(f"\n{'=' * 60}\n  REAL BASELINE RESULTS — {args.dataset} ({len(items)} questions)\n{'=' * 60}")
    print(f"  {'System':<15} {'QA Acc':>8} {'Correct/Total':>15}")
    print(f"  {'-'*15} {'-'*8} {'-'*15}")
    for system, r in per_system_reports.items():
        print(f"  {system:<15} {r['qa_accuracy']:>8.3f} {r['correct']:>6}/{r['total']:<8}")
    print(f"\n  Results -> {combined_path}")


if __name__ == "__main__":
    main()
