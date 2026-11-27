"""Train a shared lexical/POS trigger detector; JSONL input/output, list of token indices."""

import argparse
import json
import sys

import joblib
import spacy
from corpus import DEVELOPMENT, documents
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from spacy.tokens import Doc


def sentence_features(nlp, words):
    doc = nlp(Doc(nlp.vocab, words=words))
    feats = []
    for i, tok in enumerate(doc):
        values = {
            "word": tok.lower_,
            "pos": tok.pos_,
            "tag": tok.tag_,
            "lemma": tok.lemma_.casefold(),
            "suf3": tok.lower_[-3:],
            "suf2": tok.lower_[-2:],
        }
        for offset in (-2, -1, 1, 2):
            if 0 <= i + offset < len(doc):
                near = doc[i + offset]
                values[f"word{offset}"] = near.lower_
                values[f"pos{offset}"] = near.pos_
        feats.append(values)
    return feats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "detect"])
    p.add_argument("model")
    p.add_argument("--data")
    args = p.parse_args()
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    if args.mode == "train":
        features, targets = [], []
        for topic, _, sentences, mentions in documents(args.data):
            if topic in DEVELOPMENT:
                continue
            labels = {(m.sentence, i) for m in mentions for i in m.offsets}
            for sid, words in sentences.items():
                features.extend(sentence_features(nlp, words))
                targets.extend(int((sid, i) in labels) for i in range(len(words)))
        model = make_pipeline(
            DictVectorizer(), LogisticRegression(max_iter=300, C=1.0, random_state=17)
        )
        model.fit(features, targets)
        joblib.dump(model, args.model)
        print(json.dumps({"train_tokens": len(targets), "positive_tokens": sum(targets)}))
        return
    model = joblib.load(args.model)
    for line in sys.stdin:
        words = json.loads(line)["tokens"]
        features = sentence_features(nlp, words)
        pred = model.predict(features) if features else []
        print(json.dumps([i for i, x in enumerate(pred) if x]))


if __name__ == "__main__":
    main()
