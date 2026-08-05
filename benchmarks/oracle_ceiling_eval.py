#!/usr/bin/env python3
"""
Oracle-ceiling diagnostic — the experiment that tells you whether to invest
in RETRIEVAL or in ANSWERING, instead of guessing.

Skips retrieval entirely and hands the answerer the raw text of every
session in the question's own haystack (the gold sessions are guaranteed to
be in there). Whatever it scores is the ceiling any retrieval system can
reach with this answerer, judge, and dataset:

  • ceiling >> our score  -> retrieval is the bottleneck; invest there.
  • ceiling ~= our score  -> retrieval is already finding what matters;
                            the remaining gap is answering/judging, and
                            better retrieval CANNOT help.

This is the same class of check that found X-MemoryArch's 1,200-char
truncation bug in 10 minutes after weeks of blind prompt tuning
(PHASE5_NOTES.md, Lesson 43: "ALWAYS run failure analysis before
optimizing").

Usage:
    python3 benchmarks/oracle_ceiling_eval.py --dataset longmemeval --n 30
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lme_judge import build_judge_prompt, parse_judge_verdict, is_abstention  # noqa: E402
from corpus_loaders import load_locomo, load_longmemeval  # noqa: E402

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
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--gen-model", default="gpt-4o-mini")
ap.add_argument("--judge-model", default="gpt-4o")
ap.add_argument("--cap", type=int, default=40000,
                 help="chars of oracle context. Default 40k, NOT the 24k the retrieval "
                      "harnesses use: measured on LongMemEval _s, a question's gold "
                      "sessions total a mean 12.7k chars but reach 39.4k, and at a 24k "
                      "cap 12/150 questions lost their own gold evidence — understating "
                      "the ceiling by up to 8 points. A ceiling must be generous by "
                      "construction or it is not a ceiling. Retrieval systems are "
                      "deliberately NOT given this budget; they keep the realistic 24k "
                      "(~6k tokens, in line with Mem0's ~7k and Zep's 1.6k).")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--types", default="",
                 help="comma-separated question_type filter (e.g. single-session-preference) "
                      "— cheap targeted validation of a fix instead of a full rerun")
ap.add_argument("--out-suffix", default="",
                 help="suffix for the output file so a targeted run never overwrites the "
                      "canonical full-run artifact")
ap.add_argument("--lme-split", choices=["oracle", "s"], default="oracle",
                 help="LongMemEval haystack: oracle (evidence sessions only) or s "
                      "(~40 sessions/question, the split vendor numbers use)")
args = ap.parse_args()

import openai  # noqa: E402

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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


def _chat(model, prompt, max_tokens):
    r = _chat_with_retry(client, model, prompt, max_tokens)
    return (r.choices[0].message.content or "").strip()


def run_one(it, mem_by_id):
    # Gold sessions first, then the rest of the haystack — with a char cap,
    # ordering by relevance-we-already-know keeps the ceiling a true ceiling
    # instead of an artifact of which sessions the cap happened to keep.
    gold = [k for k in it.scope_keys if k in it.gold_keys]
    rest = [k for k in it.scope_keys if k not in it.gold_keys]
    parts = []
    for key in gold + rest:
        mem = mem_by_id.get(key)
        if mem and mem.content:
            parts.append(f"--- {mem.title} ---\n{mem.content}")
    context = "\n\n".join(parts)[:args.cap]

    today = getattr(it, "question_date", "") or ""
    out = _chat(args.gen_model, REASONING_PROMPT.format(
        context=context, question=it.question,
        today_line=f"\nToday's date is {today}." if today else ""), 700)
    m = re.search(r"ANSWER:\s*(.+)", out, re.IGNORECASE | re.DOTALL)
    pred = (m.group(1).strip() if m else out).split("\n")[0].strip()

    # Official per-question-type judge (benchmarks/lme_judge.py). A single
    # generic prompt scored preference/abstention/temporal categories wrongly.
    verdict = _chat(args.judge_model, build_judge_prompt(
        getattr(it, "question_type", ""), it.question, it.gold_answer, pred,
        abstention=is_abstention(getattr(it, "question_id", ""))), 5)
    ok = parse_judge_verdict(verdict)
    # Whether this question's own gold sessions survived the cap. A "ceiling"
    # computed over questions whose evidence was truncated away is not a
    # ceiling — it must be reported, not averaged in silently.
    gold_chars = sum(len(mem_by_id[k].content) + len(mem_by_id[k].title) + 10
                     for k in gold if k in mem_by_id)
    return {"question": it.question, "gold_answer": it.gold_answer,
            "predicted": pred, "correct": ok,
            "question_type": getattr(it, "question_type", ""),
            "gold_truncated": gold_chars > args.cap}


def main():
    ds = load_locomo(n_queries=args.n, seed=args.seed) if args.dataset == "locomo" \
        else load_longmemeval(n_queries=args.n, seed=args.seed, split=args.lme_split)
    mem_by_id = {m.mid: m for m in ds.memories}
    items = [q for q in ds.queries if q.gold_answer and q.scope_keys]
    if args.types:
        wanted = {t.strip() for t in args.types.split(",")}
        items = [q for q in items if getattr(q, "question_type", "") in wanted]
        print(f"  type filter {sorted(wanted)} -> {len(items)} questions")
    print(f"Oracle ceiling: {args.dataset}, {len(items)} questions, "
          f"answerer={args.gen_model}, judge={args.judge_model}, cap={args.cap} chars")

    results, correct = [], 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=safe_workers(args.gen_model, args.workers)) as pool:
        futs = [pool.submit(run_one, it, mem_by_id) for it in items]
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:
                print(f"  [error] {e}")
                continue
            with lock:
                results.append(r)
                correct += int(r["correct"])

    n = len(results)
    attempted = len(items)
    if n < attempted:
        print(f"\n!! INCOMPLETE RUN: {n}/{attempted} questions returned "
              f"({attempted - n} failed). Scores below are NOT valid — "
              f"re-run before using them.")
    by_type = {}
    for r in results:
        t = r.get("question_type") or "unknown"
        b = by_type.setdefault(t, {"correct": 0, "total": 0})
        b["total"] += 1
        b["correct"] += int(r["correct"])
    for t, b in sorted(by_type.items()):
        b["accuracy"] = round(b["correct"] / max(1, b["total"]), 4)
        print(f"  {t:32s} {b['accuracy']:.3f}  ({b['correct']}/{b['total']})")

    trunc = [r for r in results if r.get("gold_truncated")]
    clean = [r for r in results if not r.get("gold_truncated")]
    if trunc:
        c_acc = sum(r["correct"] for r in clean) / max(1, len(clean))
        print(f"\n  {len(trunc)}/{n} questions had gold evidence exceed the "
              f"{args.cap:,}-char cap.")
        print(f"  Ceiling over the {len(clean)} untruncated questions only: {c_acc:.3f} "
              f"<- the honest upper bound")

    out_path = Path(__file__).parent / f"oracle_ceiling_{args.dataset}{args.out_suffix}.json"
    out_path.write_text(json.dumps({
        "dataset": args.dataset, "gen_model": args.gen_model,
        "judge_model": args.judge_model, "cap_chars": args.cap,
        "oracle_ceiling": round(correct / max(1, n), 4),
        "correct": correct, "total": n, "attempted": len(items),
        "gold_truncated_count": sum(1 for r in results if r.get("gold_truncated")),
        "ceiling_untruncated_only": round(
            sum(r["correct"] for r in results if not r.get("gold_truncated"))
            / max(1, sum(1 for r in results if not r.get("gold_truncated"))), 4),
        "complete": n == len(items), "by_question_type": by_type,
        "results": results,
    }, indent=2))

    print(f"\n{'=' * 60}")
    print(f"ORACLE CEILING — {args.dataset}: {correct/max(1,n):.3f}  ({correct}/{n})")
    print("  No retrieval — gold sessions handed directly to the answerer.")
    print("  Any retrieval system's score on this setup is bounded by this.")
    print(f"  Results -> {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
