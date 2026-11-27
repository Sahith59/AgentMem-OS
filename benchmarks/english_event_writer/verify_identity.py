"""Independent path enumeration, gradient and prediction checks for the short-prefix writer."""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from identity import PARTITIONS, canonical, distribution, objective


def check_math():
    random = np.random.default_rng(17)
    for _ in range(64):
        values = random.normal(size=3)
        scores = torch.tensor(values[None, :], dtype=torch.float64, requires_grad=True)
        full, twin = distribution(scores)
        weights = np.exp(values)
        enumerated = np.zeros(5)
        first_probs = np.array([1, weights[0]]) / (1 + weights[0])
        second_probs = np.array([1, weights[1], weights[2]]) / (1 + weights[1:].sum())
        for choice1, choice2 in itertools.product(range(2), range(3)):
            group1 = 1 - choice1
            group2 = (2, 0, group1)[choice2]
            group = canonical((0, group1, group2))
            enumerated[PARTITIONS.index(group)] += first_probs[choice1] * second_probs[choice2]
        np.testing.assert_allclose(full.detach()[0].numpy(), enumerated, atol=1e-9)
        np.testing.assert_allclose(
            twin.detach()[0].numpy(), np.array([1, weights[1]]) / (1 + weights[1])
        )
    # In float32 use a 0.002 finite-difference step. Check all 5 possible targets.
    for label in range(5):
        for arm in ("action", "absolute", "paired"):
            scores = torch.tensor([[0.25, -0.38, 0.71]], requires_grad=True)
            gold = torch.tensor([label])
            loss = objective(scores, gold, arm)
            (gradient,) = torch.autograd.grad(loss, scores)
            for i in range(3):
                step = torch.zeros_like(scores)
                step[0, i] = 0.002
                numerical = (
                    objective(scores + step, gold, arm) - objective(scores - step, gold, arm)
                ) / 0.004
                assert abs(numerical.item() - gradient[0, i].item()) < 2e-4


def analyse(folder):
    records = [
        json.loads(s) for s in (folder / "identity_predictions.jsonl").read_text().splitlines()
    ]
    assert len({(r["doc"], r["index"]) for r in records}) == len(records)
    reports = {}
    for method in records[0]["pred"]:
        by_doc = defaultdict(list)
        for r in records:
            by_doc[r["doc"]].append(r["pred"][method] == r["gold"])
        reports[method] = {
            "both_correct": sum(map(sum, by_doc.values())),
            "macro_doc": float(np.mean([np.mean(s) for s in by_doc.values()])),
            "repeated_partition_correct": sum(
                r["gold"] == r["pred"][method] for r in records if r["gold"][0] != 4
            ),
            "repeated_examples": sum(r["gold"][0] != 4 for r in records),
            "all_distinct_correct": sum(
                r["gold"] == r["pred"][method] for r in records if r["gold"][0] == 4
            ),
        }
    pairs = {}
    for seed in (17, 31, 47):
        per_topic = defaultdict(list)
        for r in records:
            delta = int(r["pred"][f"paired{seed}"] == r["gold"]) - int(
                r["pred"][f"absolute{seed}"] == r["gold"]
            )
            per_topic[r["doc"].split("_")[0]].append(delta)
        values = [(sum(v), len(v)) for v in per_topic.values()]
        sampled = np.random.default_rng(20261127).choice(len(values), (10000, len(values)))
        bags = np.asarray(values)[sampled].sum(axis=1)
        intervals = np.quantile(bags[:, 0] / bags[:, 1], [0.025, 0.975]).tolist()
        pairs[str(seed)] = {
            "gain": sum(v.count(1) for v in per_topic.values()),
            "loss": sum(v.count(-1) for v in per_topic.values()),
            "topics": dict(zip(per_topic, values)),
            "dev_topic_bootstrap95": intervals,
        }
    print(
        json.dumps(
            {
                "rows": len(records),
                "docs": len({r["doc"] for r in records}),
                "arms": reports,
                "paired_vs_absolute": pairs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    check_math()
    analyse(p.parse_args().folder)
