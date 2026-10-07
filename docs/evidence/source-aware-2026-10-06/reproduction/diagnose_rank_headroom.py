"""Offline, evaluator-only rank and budget diagnosis of the frozen 500-case audit.

Reads cached embeddings; never loads a model, answers questions, or changes packets.
Output is exclusive. Labels are loaded separately and never used in ranking.
"""

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "2026-10-06-hybrid-source-audit"
HASHES = {
    "runtime.json": "d913dd05f2d28791264f1c7d5df431b8c6e05ea2554096db0f889a8a658fcedb",
    "labels.json": "3225604ba0b8aa0061ee868051a950efb76f96397a4b994ab98ec7850177ec01",
    "audit-v2.json": "b3382aa65c826f73fdc23de58afd135384c271d0bf46247c51f56170418dad24",
    "packets-v2.jsonl": "46caf43c3a3fdd22b7860c4b69d9616ca1d59e48dcafcabe798d95d805e71680",
    "embedding-keys.json": "6e398bcbad625ccb2f6320849170e6c4678d9a966f679a9e8397893bcc50d003",
    "embeddings.npy": "a27105ac43379d76bfc4f20ff97e87231afff57bd9b721aa9fbf54d6308e78d4",
    "preparation.json": "e9de7b7b99a013f3ef11909a5c7e2d992bad87d07bfbf78af17bde18863b2f20",
}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def norm(text):
    return " ".join(text.split())


def parse(text):
    return datetime.strptime(text, "%Y/%m/%d (%a) %H:%M")


for filename, expected in HASHES.items():
    assert sha_bytes((SOURCE / filename).read_bytes()) == expected, filename
runtime = json.loads((SOURCE / "runtime.json").read_text())
labels = {x["id"]: x for x in json.loads((SOURCE / "labels.json").read_text())}
audited = {x["id"]: x for x in json.loads((SOURCE / "audit-v2.json").read_text())["rows"]}
packets = {x["id"]: x for x in map(json.loads, (SOURCE / "packets-v2.jsonl").open())}
keys = json.loads((SOURCE / "embedding-keys.json").read_text())
lookup = {key: i for i, key in enumerate(keys)}
vectors = np.load(SOURCE / "embeddings.npy", mmap_mode="r")
assert len(lookup) == len(keys) == len(vectors)
assert np.isfinite(vectors).all()

cases = []
for case in runtime["cases"]:
    question_id = case["id"]
    cutoff = parse(case["date"])
    scope = [
        (mid, turn)
        for mid in case["scope"]
        if parse(runtime["sessions"][mid]["date"]) <= cutoff
        for turn in runtime["sessions"][mid]["turns"]
        if turn["text"].strip()
    ]
    baseline = norm(case["baseline"])
    ids = [turn["id"] for _, turn in scope]
    texts = [turn["text"] for _, turn in scope]
    if scope:
        passages = vectors[[lookup[sha_bytes(("passage: " + text).encode())] for text in texts]]
        query = vectors[lookup[sha_bytes(("query: " + case["question"]).encode())]]
        dense = passages @ query
        assert np.isfinite(dense).all()
        vectorizer = TfidfVectorizer(max_features=512, sublinear_tf=True, min_df=1)
        tfidf = vectorizer.fit_transform(texts)
        lexical = cosine_similarity(vectorizer.transform([case["question"]]), tfidf)[0]
        score = np.zeros(len(texts), dtype=np.float64)
        for similarities in (dense, lexical):
            for rank, index in enumerate(similarities.argsort()[::-1]):
                score[index] += 1.0 / (61 + rank)
        ranked_ids = [ids[i] for i in np.argsort(score)[::-1]]
    else:
        ranked_ids = []
    assert ranked_ids[:8] == [h["source_id"] for h in packets[question_id]["report"]["candidate_hits"]]
    ranks = {source_id: i + 1 for i, source_id in enumerate(ranked_ids)}
    novel_ids = {turn["id"] for _, turn in scope if norm(turn["text"]) not in baseline}
    novel_ranks = {source_id: i + 1 for i, source_id in enumerate(
        source_id for source_id in ranked_ids if source_id in novel_ids
    )}

    row = audited[question_id]
    allowance = max(0, min(4000, 40000 - len(case["baseline"])) - 2)
    block_header = len("[ORIGINAL SOURCE EVIDENCE: relevance and completeness are unverified]\n")
    missing = []
    for annotated in labels[question_id]["flagged"]:
        source_id = annotated["id"]
        if source_id in row["flagged_after"]:
            continue
        mid = annotated["session"]
        turn = runtime["sessions"][mid]["turns"][annotated["position"]]
        assert turn["id"] == source_id
        observed = parse(runtime["sessions"][mid]["date"])
        cost = block_header + len(
            f"[{source_id} | {turn['role']} | observed {observed.isoformat()}]\n"
        ) + len(turn["text"]) + 1
        missing.append({
            "id": source_id,
            "full_rank": ranks.get(source_id),
            "rank_among_exact_text_novel": novel_ranks.get(source_id),
            "future_ineligible": observed > cutoff,
            "standalone_chars": cost,
            "standalone_fit": cost <= allowance,
            "recorded_reason": next(x["reason"] for x in row["missing"] if x["id"] == source_id),
        })
    cases.append({
        "id": question_id,
        "historical_correct": row["historical_correct"],
        "eligible_turns": len(scope),
        "exact_text_novel_eligible_turns": len(novel_ids),
        "exact_text_novel_top8_turns": sum(x in novel_ids for x in ranked_ids[:8]),
        "supplement_allowance_chars": allowance,
        "missing_annotated_turns": missing,
    })


def summarize(rows):
    missing = [turn for row in rows for turn in row["missing_annotated_turns"]]
    eligible = [turn for turn in missing if turn["full_rank"] is not None]
    bins = {
        "rank_1_8": sum(turn["full_rank"] <= 8 for turn in eligible),
        "rank_9_16": sum(9 <= turn["full_rank"] <= 16 for turn in eligible),
        "rank_17_32": sum(17 <= turn["full_rank"] <= 32 for turn in eligible),
        "rank_33_64": sum(33 <= turn["full_rank"] <= 64 for turn in eligible),
        "rank_65_plus": sum(turn["full_rank"] >= 65 for turn in eligible),
    }
    return {
        "cases": len(rows),
        "cases_with_missing_annotated_turn": sum(bool(row["missing_annotated_turns"]) for row in rows),
        "missing_annotated_turns": len(missing),
        "eligible_missing_turns": len(eligible),
        "future_ineligible_missing_turns": sum(turn["future_ineligible"] for turn in missing),
        "full_rank_bins": bins,
        "eligible_missing_standalone_fit": sum(turn["standalone_fit"] for turn in eligible),
        "exact_text_novel_eligible_turns": sum(row["exact_text_novel_eligible_turns"] for row in rows),
        "eligible_turns": sum(row["eligible_turns"] for row in rows),
        "exact_text_novel_top8_turns": sum(row["exact_text_novel_top8_turns"] for row in rows),
        "recorded_missing_reasons": dict(Counter(turn["recorded_reason"] for turn in missing)),
    }


result = {
    "schema": "source-selection-rank-headroom-v1",
    "status": "OFFLINE_EVALUATOR_ONLY",
    "source_sha256": HASHES,
    "all500": summarize(cases),
    "historical77misses": summarize([case for case in cases if not case["historical_correct"]]),
    "cases": cases,
    "limits": [
        "Rank is frozen dense plus lexical RRF over time-eligible question-local turns; no labels enter ranking.",
        "Exact-text novelty is normalized full stamped-turn substring absence from baseline, not certified source identity or semantic novelty.",
        "Standalone fit assumes an otherwise empty supplement and ignores neighbors, qualifiers, and competition among turns; it is an optimistic budget upper bound.",
        "Annotation presence uses the audit's normalized body-text proxy; no answer accuracy or causal score gain is inferred.",
    ],
}
assert result["all500"]["cases"] == 500
assert result["historical77misses"]["cases"] == 77
with (OUT / "rank-headroom.json").open("x") as output:
    json.dump(result, output, indent=2)
print(json.dumps({k: result[k] for k in ("all500", "historical77misses")}, indent=2))
