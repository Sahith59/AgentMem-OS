"""S1 alias eval at scale: 53 entities, 5 classes, hard negatives.

Extends cross_lingual_kg_eval.py (n=10 pilot) to the SARVAM_TRACK
section 4 spec. $0, local CPU, multilingual-e5-small (the model the
product ships). Scoring is precision-first: the operating threshold is
chosen as the highest-recall point with precision >= 0.95, because a
false merge corrupts memory silently while a miss only loses a link.

Usage: python3 benchmarks/xling_alias_eval_v2.py
Output: benchmarks/xling_alias_eval_v2_results.json
"""
import json
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
from data.xling_alias_eval_v1 import ENTITIES, CLASSES  # noqa: E402


def main():
    from agentmem_os.db.entity_aliases import get_shared_encoder
    model = get_shared_encoder()
    assert model is not None, "multilingual extra not installed"

    surfaces = []
    index = {}

    def sid(s):
        if s not in index:
            index[s] = len(surfaces)
            surfaces.append(s)
        return index[s]

    positives, negatives = [], []  # (i, j, class, entity_id)
    for e in ENTITIES:
        fids = [sid(f) for f in e["forms"]]
        if e["class"] != "E":
            for a, b in combinations(fids, 2):
                positives.append((a, b, e["class"], e["id"]))
        for neg, _reason in e["negatives"]:
            n = sid(neg)
            for f in fids:
                negatives.append((f, n, e["class"], e["id"]))
    # cross-cluster easy negatives: first form of each entity vs first
    # form of every other entity (excluding class E vs its own lang pair)
    firsts = [(sid(e["forms"][0]), e["id"], e["class"]) for e in ENTITIES]
    for (i, ida, _ca), (j, idb, _cb) in combinations(firsts, 2):
        if ida != idb:
            negatives.append((i, j, "easy", f"{ida}|{idb}"))

    emb = model.encode([f"passage: {s}" for s in surfaces],
                       normalize_embeddings=True, show_progress_bar=False,
                       batch_size=64)
    import numpy as np
    sims = emb @ emb.T

    def score(th):
        tp = sum(1 for a, b, *_ in positives if sims[a, b] >= th)
        fp_hard = sum(1 for a, b, c, _ in negatives
                      if c != "easy" and sims[a, b] >= th)
        fp_easy = sum(1 for a, b, c, _ in negatives
                      if c == "easy" and sims[a, b] >= th)
        fn = len(positives) - tp
        prec = tp / max(1, tp + fp_hard + fp_easy)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        return {"threshold": round(th, 2), "precision": round(prec, 3),
                "recall": round(rec, 3), "f1": round(f1, 3),
                "fp_hard": fp_hard, "fp_easy": fp_easy}

    sweep = [score(t / 100) for t in range(80, 99)]
    best_f1 = max(sweep, key=lambda r: r["f1"])
    prec95 = [r for r in sweep if r["precision"] >= 0.95]
    operating = max(prec95, key=lambda r: r["recall"]) if prec95 else None

    # per-class recall at the operating point (or best_f1 fallback)
    op = operating or best_f1
    th = op["threshold"]
    per_class = {}
    for cls in sorted(CLASSES):
        if cls == "E":
            # class E scores PRECISION only: nothing should merge
            pairs = [(a, b) for a, b, c, _ in negatives if c == "E"]
            bad = sum(1 for a, b in pairs if sims[a, b] >= th)
            per_class[cls] = {"false_merges": bad, "pairs": len(pairs)}
        else:
            pairs = [(a, b) for a, b, c, _ in positives if c == cls]
            got = sum(1 for a, b in pairs if sims[a, b] >= th)
            per_class[cls] = {"recall_at_op": round(got / max(1, len(pairs)), 3),
                              "pairs": len(pairs)}

    worst = sorted(((float(sims[a, b]), surfaces[a], surfaces[b])
                    for a, b, c, _ in negatives if c != "easy"),
                   reverse=True)[:12]

    out = {
        "model": "intfloat/multilingual-e5-small",
        "entities": len(ENTITIES),
        "positive_pairs": len(positives),
        "hard_negative_pairs": sum(1 for *_ , c, _i in [(0,0,c,i) for a,b,c,i in negatives] if c != "easy"),
        "easy_negative_pairs": sum(1 for a, b, c, _ in negatives if c == "easy"),
        "sweep": sweep,
        "best_f1": best_f1,
        "operating_precision_first": operating,
        "per_class_at_operating": per_class,
        "worst_hard_negatives": [
            {"sim": round(s, 3), "a": a, "b": b} for s, a, b in worst],
        "status": "DRAFT dataset — native-speaker review pending",
    }
    path = HERE / "xling_alias_eval_v2_results.json"
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"{len(ENTITIES)} entities | {len(positives)} pos | "
          f"{out['hard_negative_pairs']} hard-neg | "
          f"{out['easy_negative_pairs']} easy-neg")
    print("best F1:", best_f1)
    print("operating (precision>=0.95):", operating)
    for cls, d in per_class.items():
        print(f"  class {cls} ({CLASSES[cls]}): {d}")
    print("wrote", path)


if __name__ == "__main__":
    main()
