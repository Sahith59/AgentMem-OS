#!/usr/bin/env python3
"""
$0 retrieval ablation — measure retrieval quality WITHOUT calling any LLM.

Retrieval either puts the gold evidence in front of the answerer or it
doesn't. That is measurable for free: run the retriever over a question's
haystack and check whether turns from the gold session(s) survive into the
context budget. No generator, no judge, no API cost, seconds per variant
instead of an hour and $4.

This exists because every paid run so far has been most valuable when it was
preceded by a free check, and because two "improvements" (dense embeddings,
window chunking) measured WORSE and were caught before anyone paid for them.

Metrics:
  gold_recall  — fraction of questions with >=1 gold-session turn retrieved
  gold_density — mean fraction of retrieved turns that are from gold sessions
  session_spread — mean distinct source sessions represented in the results
                   (multi-session questions need evidence from several; a
                   retriever that returns 10 turns from one session scores
                   well on recall and still fails the question)

Usage:
    python3 benchmarks/retrieval_ablation.py --lme-split s --n 60
    python3 benchmarks/retrieval_ablation.py --lme-split s --n 60 --variants baseline,recency,diverse,all
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_loaders import load_locomo, load_longmemeval  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["locomo", "longmemeval"], default="longmemeval")
ap.add_argument("--lme-split", choices=["oracle", "s"], default="s")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--top-k", type=int, default=40, help="turns retrieved per question")
ap.add_argument("--variants", default="baseline,session,hybrid3,hybrid5,hybrid8,hyb5+rec")
args = ap.parse_args()

_DATE_RE = re.compile(r"^\[([^\]]+)\]\s*")
# Queries that ask for the CURRENT state — these are the knowledge-update and
# "most recent" questions where the correct evidence is the LATEST mention,
# and a purely lexical ranker has no way to know that.
_RECENCY_MARKERS = re.compile(
    r"\b(current|currently|now|nowadays|these days|latest|most recent|recently|"
    r"still|today|at present|updated|switch(?:ed)?|new(?:est)?|last time)\b", re.I)


def turn_date(text: str) -> str:
    m = _DATE_RE.match(text or "")
    return m.group(1) if m else ""


def build_corpus(mem_by_id, scope_keys):
    """[(turn_text, source_session_id)] for a question's whole haystack."""
    out = []
    for key in scope_keys:
        mem = mem_by_id.get(key)
        if not mem:
            continue
        for t in mem.turns:
            c = t.get("content", "")
            if c:
                out.append((c, key))
    return out


def retrieve(query, corpus, top_k, recency=False, diverse=False, session_level=False, session_top_n=0):
    """
    TF-IDF over turns (the measured champion), with two optional signals:

      recency — when the query asks for the *current* state, break ties toward
        later-dated turns. Our Temporal KG only supersedes three hardcoded
        relation types (WORKS_AT/LIVES_AT/STUDIES_AT), so it cannot help with
        "what car do I drive now" or "which shampoo do I use". Recency is the
        general form of the same idea and needs no relation vocabulary.

      diverse — cap how many turns any single source session may contribute,
        so a multi-session question can actually see multiple sessions. Pure
        top-k routinely returns every slot from one session: high similarity,
        unanswerable question.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [c for c, _ in corpus]
    if len(texts) < 3:
        return list(range(len(texts)))

    vec = TfidfVectorizer(max_features=512, sublinear_tf=True, min_df=1)
    matrix = vec.fit_transform(texts)
    sims = cosine_similarity(vec.transform([query]), matrix)[0]

    if recency and _RECENCY_MARKERS.search(query):
        dates = [turn_date(t) for t in texts]
        ordered = sorted({d for d in dates if d})
        if ordered:
            rank = {d: i / max(1, len(ordered) - 1) for i, d in enumerate(ordered)}
            # Small additive nudge: never override a strong lexical match,
            # only order among comparably-scored candidates.
            sims = sims + 0.05 * np.array([rank.get(d, 0.0) for d in dates])

    if session_level:
        # Score SESSIONS, not turns, then return the winning sessions' turns.
        # LongMemEval's own authors measured session-valued retrieval beating
        # round-valued (R@5 0.732 vs 0.644). The mechanism matters here: turn
        # retrieval finds the gold evidence (measured 96.7% recall) but buries
        # it — only 19.6% of returned turns are gold. Selecting whole sessions
        # keeps the evidence WITH its surrounding context and drops 20
        # unrelated sessions instead of interleaving them.
        by_sess = defaultdict(list)
        for i, (_, sid) in enumerate(corpus):
            by_sess[sid].append(i)
        # A session scores as the mean of its top-3 turns: robust to session
        # length, and rewards a session with several relevant turns over one
        # with a single lucky keyword hit.
        scored = []
        for sid, idxs in by_sess.items():
            top = sorted((sims[i] for i in idxs), reverse=True)[:3]
            scored.append((sum(top) / len(top), sid))
        scored.sort(reverse=True)

        if session_top_n:
            # HYBRID: use session scores only to FILTER the candidate pool,
            # then rank turns within the surviving sessions. Whole-session
            # selection is too coarse — an LME session runs to 40 turns, so a
            # 40-turn budget buys 1-3 sessions and recall collapses (measured
            # 0.650). Filtering to the best few sessions and then picking the
            # best turns inside them keeps turn-level recall while discarding
            # the ~45 irrelevant sessions that were diluting the context.
            keep = {sid for _, sid in scored[:session_top_n]}
            cand = [i for i in range(len(texts)) if corpus[i][1] in keep]
            cand.sort(key=lambda i: -sims[i])
            return cand[:top_k]

        picked, used = [], 0
        for _, sid in scored:
            idxs = sorted(by_sess[sid])
            if used + len(idxs) > top_k and picked:
                break
            picked.extend(idxs)
            used += len(idxs)
            if used >= top_k:
                break
        return picked

    order = sims.argsort()[::-1]
    if not diverse:
        return [int(i) for i in order[:top_k] if sims[int(i)] > 0.01]

    per_session_cap = max(2, top_k // 6)
    counts = defaultdict(int)
    picked = []
    for i in order:
        i = int(i)
        if sims[i] <= 0.01:
            break
        sid = corpus[i][1]
        if counts[sid] >= per_session_cap:
            continue
        counts[sid] += 1
        picked.append(i)
        if len(picked) >= top_k:
            break
    return picked



def main():
    ds = (load_locomo(n_queries=args.n, seed=args.seed) if args.dataset == "locomo"
          else load_longmemeval(n_queries=args.n, seed=args.seed, split=args.lme_split))
    mem_by_id = {m.mid: m for m in ds.memories}
    items = [q for q in ds.queries if q.gold_answer and q.scope_keys]

    variants = {
        "baseline": dict(recency=False, diverse=False),
        "recency": dict(recency=True, diverse=False),
        "diverse": dict(recency=False, diverse=True),
        "all": dict(recency=True, diverse=True),
        "session": dict(session_level=True),
        "hybrid3": dict(session_level=True, session_top_n=3),
        "hybrid5": dict(session_level=True, session_top_n=5),
        "hybrid8": dict(session_level=True, session_top_n=8),
        "hyb5+rec": dict(session_level=True, session_top_n=5, recency=True),
    }
    wanted = [v.strip() for v in args.variants.split(",") if v.strip() in variants]

    print(f"{args.dataset}/{args.lme_split}  n={len(items)}  top_k={args.top_k}   (no LLM calls, $0)\n")
    print(f"{'variant':10s} {'gold_recall':>12s} {'gold_density':>13s} {'session_spread':>15s}")
    results = {}
    for name in wanted:
        cfg = variants[name]
        hits = dens = spread = 0
        by_type = defaultdict(lambda: [0, 0])
        for it in items:
            corpus = build_corpus(mem_by_id, it.scope_keys)
            idxs = retrieve(it.question, corpus, args.top_k, **cfg)
            got = [corpus[i][1] for i in idxs]
            gold = set(it.gold_keys)
            n_gold = sum(1 for g in got if g in gold)
            ok = n_gold > 0
            hits += int(ok)
            dens += (n_gold / max(1, len(got)))
            spread += len(set(got))
            t = getattr(it, "question_type", "") or "unknown"
            by_type[t][1] += 1
            by_type[t][0] += int(ok)
        n = len(items)
        results[name] = {
            "gold_recall": round(hits / n, 4),
            "gold_density": round(dens / n, 4),
            "session_spread": round(spread / n, 2),
            "by_type": {k: round(v[0] / v[1], 3) for k, v in sorted(by_type.items())},
        }
        r = results[name]
        print(f"{name:10s} {r['gold_recall']:>12.3f} {r['gold_density']:>13.3f} {r['session_spread']:>15.2f}")

    print("\nper-category gold_recall:")
    types = sorted({t for r in results.values() for t in r["by_type"]})
    print(f"  {'category':32s} " + "  ".join(f"{n:>9s}" for n in wanted))
    for t in types:
        print(f"  {t:32s} " + "  ".join(f"{results[n]['by_type'].get(t, 0):>9.3f}" for n in wanted))

    out = Path(__file__).parent / "retrieval_ablation_results.json"
    out.write_text(json.dumps({"dataset": args.dataset, "split": args.lme_split,
                               "n": len(items), "top_k": args.top_k,
                               "variants": results}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
