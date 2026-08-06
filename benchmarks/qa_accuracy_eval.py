#!/usr/bin/env python3
"""
QA-accuracy evaluation — the metric Mem0/Zep/Honcho actually publish.

Adapted from X-MemoryArch (RetrievalEngine/training/qa_accuracy_eval.py), a
separate project by the same author with an already-working version of this
harness. See LAUNCH_ROADMAP.md Phase 1 Group E and
agentmem_os_xmemoryarch_reuse.md for the reuse rationale, and
benchmarks/eval_metrics.py / corpus_loaders.py for the two other ported
pieces this depends on.

Pipeline per question (standard LongMemEval / LoCoMo methodology):
  1. RETRIEVE — real ContextAssembler.assemble() over a real, ingested
     AgentMem OS session scoped to this question's haystack.
  2. GENERATE — an answer from the retrieved context (GPT-4o-mini).
  3. JUDGE — the answer vs. the gold answer (GPT-4o -> correct/incorrect).
  QA accuracy = % judged correct -> directly comparable to Mem0's published
  91.6-92.5 (LoCoMo) / 94.4-94.8 (LongMemEval).

This is the metric family Mem0/Zep/Honcho publish. It is NOT the same as
retrieval recall (R@5, see eval_metrics.py) — recall is a different, easier
metric that typically runs 20-30 points higher on the same data. Do not
compare this script's numbers against a retrieval-recall number as if they
were the same thing, and do not compare against a competitor's QA-accuracy
number unless generator model, judge model, and dataset split all match.

Deliberately uses OpenAI (not Claude) for generator + judge, matching
Mem0's own published methodology — and avoiding AgentMem OS's own model
(Claude) judging its own answers, a confound a reviewer would flag
immediately.

Cost: real money. ALWAYS start with a small --n (e.g. 20-30, roughly
$0.30-0.50) before committing to a full run — see LAUNCH_ROADMAP.md Phase 1
Group E/F for cost estimates and the reasoning behind starting small.

Usage:
    python3 benchmarks/qa_accuracy_eval.py --dataset locomo --n 20
    python3 benchmarks/qa_accuracy_eval.py --dataset longmemeval --n 20

Output: benchmarks/qa_accuracy_<dataset>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_loaders import (  # noqa: E402
    load_locomo, load_longmemeval, attach_extracted_memories, facts_as_turns,
)
from lme_judge import build_judge_prompt, parse_judge_verdict, is_abstention  # noqa: E402
from real_code_utils import install_best_chroma  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

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
ap.add_argument("--n", type=int, default=20, help="questions to sample — START SMALL, this costs real money")
ap.add_argument("--gen-model", default="gpt-4o-mini")
ap.add_argument("--judge-model", default="gpt-4o")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--lme-split", choices=["oracle", "s"], default="oracle",
                 help="LongMemEval haystack: oracle (evidence sessions only) or s "
                      "(~40 sessions/question, the split vendor numbers use)")
ap.add_argument("--memory-source", choices=["extracted", "raw"], default="extracted",
                 help="what gets stored: LLM-extracted atomic memories (default, "
                      "matches what every competitor stores) or raw dialogue turns")
ap.add_argument("--context-chars", type=int, default=24000,
                 help="chars of assembled context fed to the answerer. 24k (~6k tokens) "
                      "was chosen for cost parity with compressed-memory systems, but "
                      "raw-turn memory carries less information per char — measured: at "
                      "24k, multi-session questions see only 77% of their gold sessions; "
                      "at 40k, 90%+. Always DISCLOSE this with results; it is part of "
                      "the operating point, like Mem0's top-200 or Zep's 1.6k tokens.")
ap.add_argument("--types", default="",
                 help="comma-separated question_type filter — validate a fix on the "
                      "affected slice for cents instead of re-running everything")
ap.add_argument("--out-suffix", default="",
                 help="output-file suffix so a slice run never overwrites (or resumes "
                      "from) the canonical full-run artifact")
ap.add_argument("--answerer", choices=["reasoning", "simple"], default="reasoning",
                 help="answer layer: reasoning (date-anchored CoT + aggregation + "
                      "calibrated abstention) or simple (naive one-shot)")
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
          "generator+judge — see module docstring for why.")
    sys.exit(1)
client = openai.OpenAI(api_key=api_key)

from agentmem_os.storage.store import ConversationStore  # noqa: E402
from agentmem_os.llm.context_assembler import ContextAssembler  # noqa: E402
from agentmem_os.agents.namespace_manager import AgentNamespaceManager  # noqa: E402
from agentmem_os.db.engine import get_session as get_db  # noqa: E402
from agentmem_os.db.models import Turn  # noqa: E402

RETRIEVAL_BACKEND = install_best_chroma(ContextAssembler)
print(f"Semantic retrieval backend: {RETRIEVAL_BACKEND}")

SIMPLE_PROMPT = """You answer a question using ONLY the memories provided. Be concise — answer in as few words as possible (a name, date, number, or short phrase). If the memories do not contain the answer, reply "I don't know".

Memories:
{context}

Question: {question}
Answer:"""

# The answer layer. A naive one-shot answerer is where this benchmark's
# scores actually die: retrieval can surface the right session and the
# model still fails "how many days between X and Y", "how many N in total",
# and "you never told me that". Ported from X-MemoryArch's measured result
# (same author): identical retrieval, naive answerer 0.334 -> reasoning
# answerer 0.630 on LongMemEval-S. Applied to EVERY system in the
# head-to-head, never only to ours.
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
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        # These models reject max_tokens and non-default temperature, and
        # spend completion tokens on internal reasoning before any visible
        # output — a tight cap yields empty answers, so give headroom.
        kwargs["max_completion_tokens"] = max(max_tokens * 6, 4000)
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = 0
    last = None
    for attempt in range(tries):
        try:
            return client.chat.completions.create(**kwargs)
        except _o.RateLimitError as e:
            last = e
            _t.sleep(min(2 ** attempt, 60))
        except _o.BadRequestError:
            raise  # malformed request — retrying can't fix it
        except Exception as e:
            last = e
            _t.sleep(2)
    raise RuntimeError(f"{model}: gave up after {tries} attempts ({last})")


def gen_answer(context: str, question: str, today: str = "") -> str:
    reasoning = args.answerer == "reasoning"
    if reasoning:
        prompt = REASONING_PROMPT.format(
            context=context[:args.context_chars], question=question,
            today_line=f"\nToday's date is {today}." if today else "")
    else:
        prompt = SIMPLE_PROMPT.format(context=context[:args.context_chars], question=question)
    r = _chat_with_retry(client, args.gen_model, prompt, 700 if reasoning else 80)
    out = (r.choices[0].message.content or "").strip()
    if reasoning:
        # Everything before "ANSWER:" is chain-of-thought — the judge must
        # score the final line only, or verbose reasoning gets graded.
        m = re.search(r"ANSWER:\s*(.+)", out, re.IGNORECASE | re.DOTALL)
        out = (m.group(1).strip() if m else out).split("\n")[0].strip()
    return out


def judge(question: str, gold: str, pred: str, qtype: str = "", qid: str = "") -> bool:
    r = _chat_with_retry(client, args.judge_model, build_judge_prompt(
        qtype, question, gold, pred, abstention=is_abstention(qid)), 5)
    return parse_judge_verdict(r.choices[0].message.content or "")


# ── Load dataset via the ported, bug-fixed loaders (raw turns, not
#    precomputed extraction — see module docstring) ──────────────────────

print(f"Loading {args.dataset}...")
if args.dataset == "locomo":
    ds = load_locomo(n_queries=args.n, seed=args.seed)
else:
    ds = load_longmemeval(n_queries=args.n, seed=args.seed, split=args.lme_split)

MEMORY_SOURCE = args.memory_source
if MEMORY_SOURCE == "extracted":
    hits = attach_extracted_memories(ds)
    if hits:
        print(f"  extracted memories attached to {hits}/{len(ds.memories)} sessions")
    else:
        print("  no extraction cache found — falling back to raw turns")
        MEMORY_SOURCE = "raw"

mem_by_id = {m.mid: m for m in ds.memories}
items = [q for q in ds.queries if q.gold_answer and q.scope_keys]
if args.types:
    _wanted = {t.strip() for t in args.types.split(",")}
    items = [q for q in items if getattr(q, "question_type", "") in _wanted]
    print(f"  type filter {sorted(_wanted)} -> {len(items)} questions")
print(f"QA-accuracy eval: {args.dataset}, {len(items)} questions "
      f"(of {len(ds.queries)} sampled — some lack a scope or answer), "
      f"gen={args.gen_model}, judge={args.judge_model}")


# ── Ingest each unique haystack scope into a real AgentMem OS session ────
# (synchronous ingestion, same pattern as ablation_study_real.py — no
# daemon-thread timing dependency). Deduped by the exact scope-key set so
# questions sharing a haystack reuse the same ingested session instead of
# re-ingesting it.

store = ConversationStore()
assembler = ContextAssembler()
# The gen prompt hard-caps context at 24,000 chars (~6k tokens) for cost
# parity with what competitors feed their answerers (Mem0 ~7k tokens, Zep
# 1.6k). The assembler's DEFAULT allocations assume a 128k window and build
# a ~69k-char context — so the cap used to do the real selection, keeping
# whatever happened to be first. Rank order made that accidentally safe;
# chronological presentation made it catastrophic (earliest-dated survived,
# gold evidence was cut — measured: temporal/multi-session collapsed to
# 0.13 on the slice gate). Declare the true budget so RANK selects within
# ~18k chars of semantic evidence and chronology only orders what the
# model actually sees.
assembler.allocations["semantic"] = int(args.context_chars * 0.79 // 4)
assembler.allocations["recent"] = 1200     # tokens ≈ last-20-turns tail
_namespace_mgr = AgentNamespaceManager(get_db)
_ingested = set()
_ingest_lock = threading.Lock()


def _scope_session_id(scope_keys: list) -> str:
    key = "|".join(sorted(scope_keys))
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    return f"{args.dataset}-scope-{h}"


def ensure_scope_ingested(scope_keys: list) -> str:
    sid = _scope_session_id(scope_keys)
    with _ingest_lock:
        if sid in _ingested:
            return sid
        _ingested.add(sid)
    # Idempotent across PROCESSES, not just this run: if the scope's session
    # already holds turns in the DB (e.g. a slice re-run reusing the full
    # run's scratch DB), re-ingesting would DOUBLE every turn and poison
    # retrieval. Skip instead — same data, 39 minutes saved.
    _db = get_db()
    try:
        if _db.query(Turn).filter(Turn.session_id == sid).count() > 0:
            return sid
    finally:
        _db.close()
    store.get_or_create_session(sid, name=f"{args.dataset}-scope")
    # Required before any KG ingestion — kg_nodes.agent_id has a FK to
    # agent_namespaces.agent_id, and agent_id=sid (not None) needs a
    # matching row or every insert fails with IntegrityError. Also keeps
    # each haystack scope's KG/procedural pool isolated from every other
    # scope's — see benchmarks/adapters/agentmem_adapter.py's module
    # docstring for the two bugs this sidesteps.
    _namespace_mgr.ensure_agent_exists(sid)
    for mkey in scope_keys:
        mem = mem_by_id.get(mkey)
        if not mem:
            continue
        units = facts_as_turns(mem) if (MEMORY_SOURCE == "extracted" and mem.facts) else mem.turns
        if not units:
            continue
        for turn in units:
            content = turn.get("content", "")
            if not content:
                continue
            tokens = store.token_counter.count(content)
            t = Turn(session_id=sid, role=turn.get("role", "user"),
                      content=content, token_count=tokens)
            store.db.add(t)
        store.db.commit()
        try:
            for turn in units:
                if turn.get("content"):
                    store._ingest_kg(sid, sid, turn["content"])
        except Exception as e:
            print(f"  [warn] KG ingestion failed for {mkey}: {e}")
    return sid


def retrieve_context(scope_keys: list, question: str) -> str:
    sid = ensure_scope_ingested(scope_keys)
    return assembler.assemble(sid, question, agent_id=sid)


_retrieve_lock = threading.Lock()


def run_one(it) -> dict:
    # Retrieval is serialized: ContextAssembler reads through shared
    # single-session stores, and SQLAlchemy sessions are not thread-safe.
    # The slow part — the two OpenAI network calls — stays parallel.
    with _retrieve_lock:
        ctx = retrieve_context(it.scope_keys, it.question)
    pred = gen_answer(ctx, it.question, getattr(it, "question_date", "")) if ctx.strip() else "I don't know"
    ok = judge(it.question, it.gold_answer, pred,
               getattr(it, "question_type", ""), getattr(it, "question_id", ""))
    return {"question": it.question, "gold_answer": it.gold_answer,
            "predicted": pred, "correct": ok,
            "question_type": getattr(it, "question_type", "")}


def main():
    out_path = Path(__file__).parent / f"qa_accuracy_{args.dataset}{args.out_suffix}.json"

    # Resume support: skip questions already judged in a prior (possibly
    # interrupted) run against the same output file.
    done_results = []
    done_questions = set()
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            done_results = prev.get("results", [])
            done_questions = {r["question"] for r in done_results}
            if done_results:
                print(f"Resuming: {len(done_results)} questions already judged in {out_path}")
        except Exception:
            pass

    todo = [it for it in items if it.question not in done_questions]

    lock = threading.Lock()
    results = list(done_results)
    correct = sum(1 for r in done_results if r["correct"])
    done = len(done_results)

    def _checkpoint():
        out_path.write_text(json.dumps({
            "dataset": args.dataset, "gen_model": args.gen_model,
            "judge_model": args.judge_model, "n_questions": len(items),
            "retrieval_backend": RETRIEVAL_BACKEND,
            "context_chars": args.context_chars,
            "memory_source": MEMORY_SOURCE,
            "answerer": args.answerer,
            "qa_accuracy": round(correct / max(1, done), 4),
            "correct": correct, "total": done,
            "results": results,
        }, indent=2))

    if not todo:
        print("Nothing to do — all sampled questions already judged.")
        _checkpoint()
        print(f"QA accuracy = {correct/max(1,done):.3f} ({correct}/{done})")
        return

    # Ingest every unique scope serially BEFORE the thread pool. The shared
    # ConversationStore holds ONE SQLAlchemy session and sessions are not
    # thread-safe — with ingestion inside the workers, racing threads
    # interleaved duplicate INSERTs into a single flush, hit IntegrityError,
    # and the poisoned session then failed every later question
    # (PendingRollbackError cascade — observed live: 26/30 questions lost,
    # and the 4 survivors answered from a broken store). Ingestion is local
    # and $0, so serializing it costs seconds, not money.
    unique_scopes = []
    seen_scopes = set()
    for it in todo:
        sid = _scope_session_id(it.scope_keys)
        if sid not in seen_scopes:
            seen_scopes.add(sid)
            unique_scopes.append(it.scope_keys)
    print(f"Ingesting {len(unique_scopes)} unique haystack scopes (serial, $0)...")
    for i, sk in enumerate(unique_scopes, 1):
        ensure_scope_ingested(sk)
        if i % 5 == 0 or i == len(unique_scopes):
            print(f"  {i}/{len(unique_scopes)} scopes ingested", flush=True)

    with ThreadPoolExecutor(max_workers=safe_workers(args.gen_model, args.workers)) as pool:
        futs = {pool.submit(run_one, it): it for it in todo}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:
                it = futs[f]
                print(f"  [error] question failed, skipping: {it.question[:50]}... ({e})")
                continue
            with lock:
                results.append(r)
                correct += int(r["correct"])
                done += 1
                if done % 10 == 0 or done == len(items):
                    print(f"  {done}/{len(items)}  running QA acc={correct/done:.3f}", flush=True)
                    _checkpoint()

    _checkpoint()
    print("\n" + "=" * 60)
    print(f"QA ACCURACY — {args.dataset} ({len(items)} questions)")
    print("=" * 60)
    print(f"  QA accuracy = {correct/max(1,done):.3f}  ({correct}/{done} correct)")
    print(f"  Generator: {args.gen_model} | Judge: {args.judge_model}")
    if args.dataset == "locomo":
        print("  Comparable to: Mem0 LoCoMo 91.6-92.5 (QA accuracy)")
    else:
        print("  Comparable to: Mem0 LongMemEval 94.4-94.8 (QA accuracy)")
    print(f"  Results -> {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
