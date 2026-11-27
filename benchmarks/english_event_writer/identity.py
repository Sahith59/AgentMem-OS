"""Development-only partition writer, exactly marginalized over all pointers in a triple."""

import argparse
import json
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from corpus import DEVELOPMENT, documents
from sklearn.feature_extraction import DictVectorizer

# Every equivalence relation over three time-ordered mentions, canonical identities.
PARTITIONS = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, 2))
# columns: first/last equality, one, two or three distinct observed events.
PROBES = torch.tensor(
    [[int(p[0] == p[2]), *[int(len(set(p)) == k) for k in (1, 2, 3)]] for p in PARTITIONS],
    dtype=torch.float32,
)
TWIN_PROBES = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
torch.set_num_threads(1)


def canonical(labels):
    unique = {}
    return tuple(unique.setdefault(x, len(unique)) for x in labels)


def features(a, b):
    x, y = " ".join(a.words), " ".join(b.words)
    left, right = set(a.left), set(b.left)
    return {
        "a:" + x: 1.0,
        "b:" + y: 1.0,
        "exact": float(x == y),
        "same_head": float(a.words[-1] == b.words[-1]),
        "pair:" + x + "|" + y: 1.0,
        "word_overlap": len(set(a.words) & set(b.words)) / len(set(a.words) | set(b.words)),
        "char_sim": SequenceMatcher(None, x, y).ratio(),
        "prefix3": float(x[:3] == y[:3]),
        "context_jaccard": len(left & right) / max(1, len(left | right)),
        "same_sentence": float(a.sentence == b.sentence),
        "subject_in_left": float(any(w in right for w in a.words)),
    }


def families(folder):
    splits = [[], []]
    for topic, doc, _, mentions in documents(folder):
        # Bounded consecutive observations. Ambiguous overlapping source spans are excluded.
        for i, triple in enumerate(zip(mentions, mentions[1:], mentions[2:])):
            a, b, c = triple
            if a.end >= b.start or b.end >= c.start:
                continue
            label = PARTITIONS.index(canonical([m.label for m in triple]))
            splits[int(topic in DEVELOPMENT)].append(
                (doc, str(i), [features(a, b), features(a, c), features(b, c)], label)
            )
    return splits


def tensors(rows, vectorizer):
    # Densify minibatches only: lexical feature space is fitted on train alone.
    feature_matrix = vectorizer.transform([f for _, _, fs, _ in rows for f in fs]).tocsr()
    y = torch.tensor([r[3] for r in rows])
    return feature_matrix, y


def distribution(s):
    e = torch.sigmoid(s[:, 0])
    n, a, b = torch.softmax(torch.stack((torch.zeros_like(e), s[:, 1], s[:, 2]), 1), dim=1).unbind(
        1
    )
    full = torch.stack((e * (a + b), e * n, (1 - e) * a, (1 - e) * b, (1 - e) * n), 1)
    q = torch.sigmoid(s[:, 1])
    twin = torch.stack((1 - q, q), 1)
    return full, twin


def objective(scores, y, arm):
    full, twin = distribution(scores)
    truth = PROBES[y]
    yt = truth[:, 0].long()
    ty = TWIN_PROBES[yt]
    ix = torch.arange(len(y))
    action = (
        -(torch.log(full[ix, y].clamp_min(1e-12)) + torch.log(twin[ix, yt].clamp_min(1e-12))).mean()
        / 2
    )
    pf, pt = full @ PROBES, twin @ TWIN_PROBES
    absolute = ((pf - truth).square().mean() + (pt - ty).square().mean()) / 2
    change = ((pf - pt) - (truth - ty)).square().mean()
    return action + int(arm != "action") * absolute + int(arm == "paired") * change


def batch(x, indices):
    j = (3 * np.array(indices)[:, None] + np.arange(3)).ravel()
    return torch.from_numpy(x[j].toarray().astype("float32")).reshape(len(indices), 3, -1)


def run(train, dev, width, seed, arm):
    torch.manual_seed(seed)
    model = torch.nn.Linear(width, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015, weight_decay=0.02)
    rng = np.random.default_rng(seed)
    x, y = train
    for epoch in range(14):
        order = rng.permutation(len(y))
        for i in range(0, len(order), 64):
            ix = order[i : i + 64]
            loss = objective(model(batch(x, ix)).squeeze(-1), y[ix], arm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(dev[1]), 64):
            ix = np.arange(start, min(start + 64, len(dev[1])))
            full, twin = distribution(model(batch(dev[0], ix)).squeeze(-1))
            predictions.extend(zip(full.argmax(1).tolist(), twin.argmax(1).tolist()))
    return predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data")
    p.add_argument("out")
    args = p.parse_args()
    tr, dv = families(args.data)
    vectorizer = DictVectorizer().fit([f for row in tr for f in row[2]])
    train, dev = tensors(tr, vectorizer), tensors(dv, vectorizer)
    rows = [
        {
            "doc": doc,
            "index": int(i),
            "gold": [y, int(PARTITIONS[y][0] == PARTITIONS[y][2])],
            "pred": {},
        }
        for doc, i, _, y in dv
    ]
    for r, (_, _, f, _) in zip(rows, dv):
        a, b, c = (int(col["exact"]) for col in f)
        exact = canonical([0, 0 if a else 1, 0 if b else (0 if a else 1) if c else 2])
        r["pred"].update(new=[4, 0], exact=[PARTITIONS.index(exact), b])
    for seed in (17, 31, 47):
        for arm in ("action", "absolute", "paired"):
            t = time.monotonic()
            pred = run(train, dev, len(vectorizer.vocabulary_), seed, arm)
            for r, answer in zip(rows, pred):
                r["pred"][f"{arm}{seed}"] = list(answer)
            print(arm, seed, round(time.monotonic() - t, 1), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "identity_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {
        "train_triples": len(tr),
        "dev_triples": len(dv),
        "train_docs": len(set(r[0] for r in tr)),
        "dev_docs": len(set(r[0] for r in dv)),
        "features": len(vectorizer.vocabulary_),
        "class_counts": dict(Counter(r[3] for r in dv)),
        "correct": {},
    }
    for key in rows[0]["pred"]:
        summary["correct"][key] = sum(r["gold"] == r["pred"][key] for r in rows)
    (out / "identity_counts.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
