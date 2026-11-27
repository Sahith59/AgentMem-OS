"""Contiguous event spans with training-topic calibration; JSONL in, spans out."""

import argparse
import json
import math
import sys

import joblib
import numpy as np
import spacy
from corpus import DEVELOPMENT, documents
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from spacy.tokens import Doc

CALIBRATION = {3, 9, 16, 24, 32}
SHIFTS = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)


def features(doc):
    result = []
    for i, t in enumerate(doc):
        row = {
            "w": t.lower_,
            "l": t.lemma_.lower(),
            "p": t.pos_,
            "t": t.tag_,
            "d": t.dep_,
            "head_lemma": t.head.lemma_.lower(),
            "head_pos": t.head.pos_,
            "dep_pos": t.dep_ + "/" + t.pos_,
            "morph": str(t.morph),
            "shape": t.shape_,
            "suffix2": t.lower_[-2:],
            "suffix3": t.lower_[-3:],
            "suffix4": t.lower_[-4:],
        }
        for delta in (-2, -1, 1, 2):
            if 0 <= i + delta < len(doc):
                n = doc[i + delta]
                row[f"l{delta}"] = n.lemma_.lower()
                row[f"p{delta}"] = n.pos_
                row[f"d{delta}"] = n.dep_
        for delta in (-1, 1):
            if 0 <= i + delta < len(doc):
                n = doc[i + delta]
                row[f"bigram{delta}"] = t.lemma_.lower() + "/" + n.lemma_.lower()
        result.append(row)
    return result


def targets(length, spans):
    labels = ["O"] * length
    for span in sorted(spans, key=lambda x: (-len(x), x)):
        if list(span) != list(range(span[0], span[-1] + 1)) or any(labels[i] != "O" for i in span):
            continue
        labels[span[0]] = "B"
        for j in span[1:]:
            labels[j] = "I"
    return labels


def decode(scores, order, shift):
    if not len(scores):
        return []
    # B/I/O order with I disallowed after O or at the beginning of a sentence.
    local = scores[:, order].copy()
    local[:, :2] += shift
    state = local[0].copy()
    state[1] = -math.inf
    history = []
    for row in local[1:]:
        best = int(np.argmax(state))
        prev = [best, int(np.argmax(state[:2])), best]
        state = row + state[prev]
        history.append(prev)
    result = [int(np.argmax(state))]
    for prev in reversed(history):
        result.append(prev[result[-1]])
    spans = []
    for i, label in enumerate(reversed(result)):
        if label == 0:
            spans.append([i])
        elif label == 1:
            spans[-1].append(i)
    return spans


def fit(examples):
    x, y = [], []
    for _, feat, gold in examples:
        x.extend(feat)
        y.extend(targets(len(feat), gold))
    # Sparse SAGA avoids a dense Hessian on the lexical features.
    model = make_pipeline(
        DictVectorizer(),
        LogisticRegression(solver="saga", max_iter=400, tol=0.002, random_state=17),
    )
    model.fit(x, y)
    return model


def train(data, nlp):
    raw = []
    for topic, _, sents, mentions in documents(data):
        if topic in DEVELOPMENT:
            continue
        by_sent = {}
        for m in mentions:
            by_sent.setdefault(m.sentence, set()).add(m.offsets)
        for sid, words in sents.items():
            raw.append((topic, words, by_sent.get(sid, set())))
    docs = nlp.pipe((Doc(nlp.vocab, words=words) for _, words, _ in raw), batch_size=64)
    examples = [
        (topic, features(doc), gold) for (topic, _, gold), doc in zip(raw, docs, strict=True)
    ]
    model = fit([r for r in examples if r[0] not in CALIBRATION])
    order = [list(model.classes_).index(c) for c in ("B", "I", "O")]
    calibration = [
        (model.predict_log_proba(feat), gold)
        for t, feat, gold in examples
        if t in CALIBRATION and feat
    ]
    choices = []
    for shift in SHIFTS:
        tp = n_pred = n_gold = 0
        for scores, gold in calibration:
            spans = {tuple(s) for s in decode(scores, order, shift)}
            tp += len(spans & gold)
            n_pred += len(spans)
            n_gold += len(gold)
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gold
        choices.append(
            {
                "shift": shift,
                "precision": precision,
                "recall": recall,
                "f2": 5 * tp / (4 * n_gold + n_pred),
                "true_positives": tp,
                "predicted": n_pred,
                "gold": n_gold,
            }
        )
    guarded = [c for c in choices if c["precision"] >= 0.65]
    qualifying = [c for c in guarded if c["recall"] >= 0.85]
    selected = (
        max(qualifying, key=lambda c: (c["precision"], -c["shift"]))
        if qualifying
        else max(guarded or choices, key=lambda c: (c["f2"], -c["shift"]))
    )
    model = fit(examples)
    assert np.isfinite(model[-1].coef_).all() and np.isfinite(model[-1].intercept_).all()
    print(
        json.dumps(
            {
                "calibration_topics": sorted(CALIBRATION),
                "grid": choices,
                "chosen_shift": selected["shift"],
                "fit_topics": sorted({t for t, _, _ in examples}),
                "fit_sentences": len(examples),
                "fit_tokens": sum(len(f) for _, f, _ in examples),
                "finite_coefficients": True,
            }
        )
    )
    return model, selected["shift"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "detect"])
    parser.add_argument("model")
    parser.add_argument("--data")
    args = parser.parse_args()
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    if args.mode == "train":
        joblib.dump(train(args.data, nlp), args.model)
        return
    model, shift = joblib.load(args.model)
    order = [list(model.classes_).index(c) for c in ("B", "I", "O")]

    def sentences():
        for line in sys.stdin:
            r = json.loads(line)
            yield Doc(nlp.vocab, words=r["tokens"]), r

    for doc, r in nlp.pipe(sentences(), as_tuples=True, batch_size=64):
        feat = features(doc)
        spans = decode(model.predict_log_proba(feat), order, shift) if feat else []
        print(json.dumps({"doc": r["doc"], "sentence": r["sentence"], "spans": spans}))


if __name__ == "__main__":
    main()
