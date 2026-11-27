"""Run the frozen SMCalFlow operation comparison with a local output directory."""

import argparse
import collections
import gzip
import hashlib
import io
import json
import platform
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from threadpoolctl import threadpool_limits

PROTOCOL = Path(__file__).resolve().with_name("PROTOCOL.md")
SEED = 27112026
NAMES = ["create", "update", "delete", "other"]
BLOBS = {
    "train": (
        "468c9b4612ea35cf2f79b278ae5683669da97f23",
        "2beb5fd51056999051081300aea33fdf228378fa214f915f6167cb4924e23d28",
    ),
    "valid": (
        "006c95c6b1b210cc2f986dc85a2e7b685133f82c",
        "934630242caadae80a0702e940f09ccec9f9f94422a842df06e2c5b46cfbd0d8",
    ),
}
STRING = re.compile(r'"(?:\\.|[^"\\])*"')
MARKER = re.compile(r"\(\s*(Create|Update|Delete)PreflightEventWrapper(?=[\s)])")
TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
ARMS = {"current": (0,), "prefix": (0, 1, 2), "human_prefix": (0, 1), "prior_only": (1, 2)}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tokens(text):
    return TOKEN.findall(text.casefold())


def normalize(text):
    return " ".join(tokens(text))


def read_split(folder, name, excluded_hashes, fetch=False):
    blob, checksum = BLOBS[name]
    path = folder / "local-data" / (name + ".zip")
    if not path.exists() and fetch:
        url = (
            "https://api.github.com/repos/microsoft/task_oriented_dialogue_as_dataflow_synthesis/git/blobs/"
            + blob
        )
        path.parent.mkdir(exist_ok=True)
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.raw+json"})
        data = urllib.request.urlopen(req, timeout=90).read()
        assert digest(data) == checksum
        path.write_bytes(data)
    data = path.read_bytes()
    assert digest(data) == checksum
    ids, fingerprints, rows = set(), set(), []
    counts = collections.Counter()
    with (
        zipfile.ZipFile(io.BytesIO(data)) as z,
        z.open(name + ".dataflow_dialogues.jsonl") as stream,
    ):
        for line in stream:
            dialogue = json.loads(line)
            did = dialogue["dialogue_id"]
            assert did not in ids
            ids.add(did)
            turns = dialogue["turns"]
            counts["raw_turns"] += len(turns)
            fp = digest(
                json.dumps(
                    [normalize(t["user_utterance"]["original_text"]) for t in turns],
                    ensure_ascii=False,
                ).encode()
            )
            fingerprints.add(fp)
            if fp in excluded_hashes:
                counts["excluded_overlap_dialogues"] += 1
                counts["excluded_overlap_turns"] += len(turns)
                continue
            user, assistant = "", ""
            for i, t in enumerate(turns):
                assert t["turn_index"] == i
                current = t["user_utterance"]["original_text"]
                op = set(MARKER.findall(STRING.sub('""', t["lispress"])))
                if t["skip"]:
                    counts["skip"] += 1
                elif len(op) > 1:
                    counts["multiple"] += 1
                else:
                    target = NAMES.index(next(iter(op)).lower()) if op else 3
                    rows.append((did, i, current, user, assistant, target))
                user = current
                assistant = t["agent_utterance"]["original_text"]
    counts.update(
        {
            "raw_dialogues": len(ids),
            "unique_text_sequences": len(fingerprints),
            "retained_turns": len(rows),
            "retained_dialogues": len(set(r[0] for r in rows)),
        }
    )
    return rows, ids, fingerprints, dict(counts)


def encode_field(field, text):
    words = tokens(text)[: (32 if field == 0 else 16)]
    prefix = str(field) + ":"
    return [prefix + w for w in words] + [prefix + a + "~" + b for a, b in zip(words, words[1:])]


def features(rows):
    fields = []
    for field in range(3):
        vectorizer = HashingVectorizer(
            n_features=65536,
            norm="l2",
            alternate_sign=False,
            dtype=np.float64,
            analyzer=lambda s: encode_field(field, s),
        )
        fields.append(vectorizer.transform(r[2 + field] for r in rows))
    return fields


def combine(fields, selected, perm=None):
    result = None
    for f in selected:
        matrix = fields[f] if perm is None or f == 0 else fields[f][perm]
        result = matrix if result is None else result + matrix
    return result


def confusion(y, pred):
    return np.bincount(4 * y + pred, minlength=16).reshape(4, 4)


def scores(cm):
    diag = np.diagonal(cm, axis1=-2, axis2=-1)
    den = cm.sum(axis=-1) + cm.sum(axis=-2)
    return np.divide(2 * diag, den, out=np.zeros_like(diag, dtype=float), where=den != 0)


def summarize(y, pred, mask):
    cm = confusion(y[mask], pred[mask])
    return {
        "n": int(cm.sum()),
        "accuracy": float(cm.trace() / cm.sum()),
        "macro_f1": float(scores(cm).mean()),
        "f1": dict(zip(NAMES, scores(cm).tolist())),
        "confusion_actual_by_predicted": cm.tolist(),
    }


def permutations(rows):
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r[1]:
            groups[r[1]].append(i)
    perm = np.arange(len(rows))
    single = 0
    for values in groups.values():
        idx = sorted(values, key=lambda i: digest(rows[i][0].encode()))
        if len(idx) == 1:
            single += 1
        perm[idx] = idx[1:] + idx[:1]
    return perm, single


def clustered_delta(rows, y, previous, current):
    docs, group = np.unique([r[0] for r in rows], return_inverse=True)
    matrices = []
    for pred in [previous, current]:
        mats = np.bincount(group * 16 + y * 4 + pred, minlength=len(docs) * 16).reshape(
            len(docs), 4, 4
        )
        matrices.append(mats)
    rng = np.random.default_rng(SEED)
    results = []
    for _ in range(1000):
        sample = rng.integers(0, len(docs), size=len(docs))
        results.append(
            scores(matrices[1][sample].sum(axis=0)).mean()
            - scores(matrices[0][sample].sum(axis=0)).mean()
        )
    return np.percentile(results, [2.5, 97.5]).tolist()


def run(folder, fetch):
    folder.mkdir(exist_ok=True, parents=True)
    train, train_ids, train_fps, train_counts = read_split(folder, "train", set(), fetch)
    valid, valid_ids, _, valid_counts = read_split(folder, "valid", train_fps, fetch)
    assert train_ids.isdisjoint(valid_ids)
    train_fields, valid_fields = features(train), features(valid)
    y, v = (np.array([r[-1] for r in rows]) for rows in [train, valid])
    prior = np.array([r[1] > 0 for r in valid])
    vocab = {normalize(r[2]) for r in train}
    unseen = np.array([normalize(r[2]) not in vocab for r in valid])
    masks = {
        "overall": np.ones(len(v), bool),
        "initial": ~prior,
        "followup": prior,
        "unseen_current_text": unseen,
    }
    predictions, fit_stats = {}, {}
    perm, single = permutations(valid)
    for name, selected in ARMS.items():
        train_matrix = combine(train_fields, selected)
        model = SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            alpha=0.0001,
            random_state=SEED,
            max_iter=12,
            tol=None,
            n_jobs=1,
        )
        then = time.monotonic()
        model.fit(train_matrix, y)
        assert model.n_iter_ == 12
        predictions[name] = model.predict(combine(valid_fields, selected))
        fit_stats[name] = {
            "seconds": round(time.monotonic() - then, 3),
            "nnz_train": train_matrix.nnz,
            "training_passes": model.n_iter_,
        }
        if name == "prefix":
            predictions["shuffled_prefix"] = model.predict(combine(valid_fields, selected, perm))
    gains = (predictions["prefix"] == v) & (predictions["current"] != v)
    losses = (predictions["prefix"] != v) & (predictions["current"] == v)
    allscores = {
        name: {subset: summarize(v, pred, mask) for subset, mask in masks.items()}
        for name, pred in predictions.items()
    }
    delta = allscores["prefix"]["overall"]["macro_f1"] - allscores["current"]["overall"]["macro_f1"]
    lines = [
        json.dumps(
            {
                "dialogue_id": row[0],
                "turn": row[1],
                "actual": NAMES[int(v[i])],
                "unseen_normalized_current_text": bool(unseen[i]),
                **{n: NAMES[int(p[i])] for n, p in predictions.items()},
            }
        )
        for i, row in enumerate(valid)
    ]
    payload = ("\n".join(lines) + "\n").encode()
    (folder / "validation_predictions.jsonl.gz").write_bytes(gzip.compress(payload, mtime=0))
    out = {
        "protocol_sha256": digest(PROTOCOL.read_bytes()),
        "run_code_sha256": digest(Path(__file__).read_bytes()),
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "archives": BLOBS,
        "train": train_counts,
        "validation": valid_counts,
        "id_overlap": len(train_ids & valid_ids),
        "labels_train": dict(zip(NAMES, np.bincount(y).tolist())),
        "labels_valid": dict(zip(NAMES, np.bincount(v).tolist())),
        "full_model_validation_unpermuted_singleton_followups": single,
        "metrics": allscores,
        "fit_stats": fit_stats,
        "primary_paired_turn_gains": int(gains.sum()),
        "primary_paired_turn_losses": int(losses.sum()),
        "primary_delta_macro_f1": delta,
        "dialogue_bootstrap_95pct": clustered_delta(
            valid, v, predictions["current"], predictions["prefix"]
        ),
        "validation_prediction_uncompressed_sha256": digest(payload),
    }
    (folder / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: out[k]
                for k in [
                    "train",
                    "validation",
                    "labels_train",
                    "labels_valid",
                    "primary_delta_macro_f1",
                    "dialogue_bootstrap_95pct",
                ]
            },
            indent=2,
        )
    )
    for n in allscores:
        print(n, allscores[n]["overall"]["macro_f1"], allscores[n]["overall"]["accuracy"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="download the two pinned official archives when missing",
    )
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.folder, args.fetch)
