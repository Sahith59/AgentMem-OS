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
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_loaders import load_locomo, load_longmemeval  # noqa: E402
from real_code_utils import install_tfidf_chroma  # noqa: E402

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
from agentmem_os.db.models import Turn  # noqa: E402

install_tfidf_chroma(ContextAssembler)

GEN_PROMPT = """You answer a question using ONLY the memories provided. Be concise — answer in as few words as possible (a name, date, number, or short phrase). If the memories do not contain the answer, reply "I don't know".

Memories:
{context}

Question: {question}
Answer:"""

JUDGE_PROMPT = """You are grading whether a predicted answer to a question is correct, given the gold answer. Be lenient about phrasing, formatting, and extra words — judge only whether the predicted answer conveys the same factual information as the gold answer.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Is the predicted answer correct? Reply with exactly one word: CORRECT or INCORRECT."""


def gen_answer(context: str, question: str) -> str:
    r = client.chat.completions.create(
        model=args.gen_model, max_tokens=80, temperature=0,
        messages=[{"role": "user", "content": GEN_PROMPT.format(context=context[:8000], question=question)}],
    )
    return (r.choices[0].message.content or "").strip()


def judge(question: str, gold: str, pred: str) -> bool:
    r = client.chat.completions.create(
        model=args.judge_model, max_tokens=5, temperature=0,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)}],
    )
    out = (r.choices[0].message.content or "").strip().upper()
    # "INCORRECT" contains "CORRECT" as a substring — must check INCORRECT first.
    if "INCORRECT" in out:
        return False
    return "CORRECT" in out


# ── Load dataset via the ported, bug-fixed loaders (raw turns, not
#    precomputed extraction — see module docstring) ──────────────────────

print(f"Loading {args.dataset}...")
if args.dataset == "locomo":
    ds = load_locomo(n_queries=args.n, seed=args.seed)
else:
    ds = load_longmemeval(n_queries=args.n, seed=args.seed)

mem_by_id = {m.mid: m for m in ds.memories}
items = [q for q in ds.queries if q.gold_answer and q.scope_keys]
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
    store.get_or_create_session(sid, name=f"{args.dataset}-scope")
    for mkey in scope_keys:
        mem = mem_by_id.get(mkey)
        if not mem or not mem.turns:
            continue
        for turn in mem.turns:
            content = turn.get("content", "")
            if not content:
                continue
            tokens = store.token_counter.count(content)
            t = Turn(session_id=sid, role=turn.get("role", "user"),
                      content=content, token_count=tokens)
            store.db.add(t)
        store.db.commit()
        try:
            for turn in mem.turns:
                if turn.get("content"):
                    store._ingest_kg(sid, None, turn["content"])
        except Exception as e:
            print(f"  [warn] KG ingestion failed for {mkey}: {e}")
    return sid


def retrieve_context(scope_keys: list, question: str) -> str:
    sid = ensure_scope_ingested(scope_keys)
    return assembler.assemble(sid, question)


def run_one(it) -> dict:
    ctx = retrieve_context(it.scope_keys, it.question)
    pred = gen_answer(ctx, it.question) if ctx.strip() else "I don't know"
    ok = judge(it.question, it.gold_answer, pred)
    return {"question": it.question, "gold_answer": it.gold_answer,
            "predicted": pred, "correct": ok}


def main():
    out_path = Path(__file__).parent / f"qa_accuracy_{args.dataset}.json"

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
            "qa_accuracy": round(correct / max(1, done), 4),
            "correct": correct, "total": done,
            "results": results,
        }, indent=2))

    if not todo:
        print("Nothing to do — all sampled questions already judged.")
        _checkpoint()
        print(f"QA accuracy = {correct/max(1,done):.3f} ({correct}/{done})")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
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
