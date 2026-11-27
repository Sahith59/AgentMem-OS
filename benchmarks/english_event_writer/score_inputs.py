"""Score frozen detector JSONL against local author anchors and prior-chain links."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.open()]


def matching(pred, marks, overlap):
    """Maximum cardinality one-to-one span matching. Stable order resolves ties."""
    linked = {}

    def assign(gold_index, visited):
        g = marks[gold_index]["span"]
        choices = [
            (p != g, j) for j, p in enumerate(pred) if (set(p) & set(g) if overlap else p == g)
        ]
        for _, j in sorted(choices):
            if j in visited:
                continue
            visited.add(j)
            if j not in linked or assign(linked[j], visited):
                linked[j] = gold_index
                return True
        return False

    for i in range(len(marks)):
        assign(i, set())
    return {i: j for j, i in linked.items()}


def contributions(runtime, gold, predictions, overlap):
    counts = defaultdict(Counter)
    prior = []
    prior_gold = []
    prevdoc = None
    for row, pred in zip(runtime, predictions, strict=True):
        doc, sid = row["doc"], row["sentence"]
        if prevdoc != doc:
            prior, prior_gold = [], []
            prevdoc = doc
        if isinstance(pred, dict):
            assert (pred["doc"], pred["sentence"]) == (doc, sid)
            spans = pred["spans"]
        else:
            spans = [[i] for i in pred]
        assert all(
            isinstance(s, list)
            and s
            and s == sorted(set(s))
            and all(type(i) is int and 0 <= i < len(row["tokens"]) for i in s)
            for s in spans
        )
        assert len({tuple(s) for s in spans}) == len(spans)
        ref = gold.get((doc, sid))
        marks = ref["mentions"] if ref else []
        mapped = matching(spans, marks, overlap)

        def label(span):
            return " ".join(row["tokens"][i].lower() for i in span)

        local = [((sid, s[0]), (sid, s[-1]), label(s), (sid, j)) for j, s in enumerate(spans)]
        past_and_current = prior + local
        if ref:
            c = counts[ref["topic"]]
            c.update(
                {
                    "sentences": 1,
                    "predictions": len(spans),
                    "mentions": len(marks),
                    "matched": len(mapped),
                    "multitoken_gold": sum(len(m["span"]) > 1 for m in marks),
                }
            )
            current_gold = [
                (
                    (sid, m["span"][0]),
                    (sid, m["span"][-1]),
                    m["chain"],
                    (sid, mapped[i]) if i in mapped else None,
                )
                for i, m in enumerate(marks)
            ]
            for i, (start, _, chain, pointer) in enumerate(current_gold):
                valid = [
                    p for _, end, cl, p in prior_gold + current_gold if end < start and cl == chain
                ]
                if not valid:
                    continue
                c["applicable"] += 1
                if pointer is None:
                    continue
                c["current_detected"] += 1
                pred_start, _, surface, _ = local[mapped[i]]
                previous = [a for a in past_and_current if a[1] < pred_start]
                previous.sort(key=lambda a: (a[2] == surface, a[1], a[0], a[3]), reverse=True)
                if set(valid).intersection(a[3] for a in previous[:8]):
                    c["found"] += 1
                if set(valid).intersection(a[3] for a in previous):
                    c["found_uncapped"] += 1
            prior_gold += current_gold
        prior += local
    return counts


RATES = {
    "recall": ("matched", "mentions"),
    "precision": ("matched", "predictions"),
    "top8_end_to_end": ("found", "applicable"),
    "top8_given_current": ("found", "current_detected"),
    "uncapped_end_to_end": ("found_uncapped", "applicable"),
    "predictions_per_indexed_sentence": ("predictions", "sentences"),
}


def rates(count):
    return {k: count[n] / count[d] if count[d] else None for k, (n, d) in RATES.items()}


def summarize(counts):
    total = sum(counts.values(), Counter())
    sample_counts = list(counts.values())
    rng = random.Random(1127)
    sims = defaultdict(list)
    for _ in range(10000):
        draw = sum(rng.choices(sample_counts, k=len(sample_counts)), Counter())
        for k, v in rates(draw).items():
            if v is not None:
                sims[k].append(v)
    return {
        "counts": dict(total),
        "rates": rates(total),
        "topic_bootstrap95": {
            k: [sorted(v)[int(0.025 * (len(v) - 1))], sorted(v)[int(0.975 * (len(v) - 1))]]
            for k, v in sims.items()
        },
        "by_topic": {k: {"counts": dict(v), "rates": rates(v)} for k, v in sorted(counts.items())},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prepared", type=Path)
    p.add_argument("predictions", type=Path)
    args = p.parse_args()
    runtime = rows(args.prepared / "ecb_runtime.jsonl")
    reference = {(r["doc"], r["sentence"]): r for r in rows(args.prepared / "ecb_gold.jsonl")}
    answers = rows(args.predictions)
    print(
        json.dumps(
            {
                k: summarize(contributions(runtime, reference, answers, overlap))
                for k, overlap in [("exact", False), ("overlap", True)]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
