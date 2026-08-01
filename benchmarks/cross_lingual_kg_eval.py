#!/usr/bin/env python3
"""
Cross-lingual entity-alias resolution — precision/recall measurement on a
hand-labeled dataset, BEFORE anything gets wired into the live knowledge
graph. This is Phase 1 of the Sarvam extension
(AgentMemOS_v2_Sarvam_Extension_Roadmap.md) — the resequencing Codex's
adversarial review specifically required: measure whether cosine-threshold
alias merging actually works, and at what threshold, before shipping it
as a live feature.

Method (per arXiv 2601.00814, Jan 2026, cited in the roadmap doc): embed
each entity mention with an off-the-shelf multilingual encoder — no
fine-tuning — and merge same-entity-different-language mentions by a
cosine-similarity threshold. Model here: intfloat/multilingual-e5-small
(384-dim, ~470MB — the lightest current multilingual-e5 variant, chosen
over -base/-large to keep this a genuinely small dependency add, not a
GPU-cluster-scale download).

Dataset: 10 hand-picked real-world entities (people, places, orgs,
concepts), each expressed in English/Hindi/Tamil (chosen to double as
Sarvam-relevant demo material, and because these are the three languages
already spot-checked in this session's own sanity test) — 30 positive
("same entity, different language") pairs. Plus 6 hand-picked HARD
negatives specifically targeting the failure modes Codex's review named:
polysemous brand/common-word collisions (Sarvam AI vs. the Hindi/Sanskrit
word "sarvam" meaning "everything"), similar-sounding but different real
places (Bangalore vs. Bengal, Chennai vs. China, India vs. Indiana), same
surname referring to different people (Narendra Modi vs. a different
"Modi"), and two large but unrelated Indian entities (RBI vs. Reliance).
Plus all remaining cross-cluster pairs as "easy" negatives, for a fuller
precision/recall picture beyond just the adversarial cases.

Cost: $0. intfloat/multilingual-e5-small is a one-time ~470MB download
(cached locally afterward, no repeat cost), inference is local CPU, no
API calls of any kind.

Usage:
    python3 benchmarks/cross_lingual_kg_eval.py

Output: benchmarks/cross_lingual_kg_eval_results.json
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tier_lib import ok, warn, hdr, sub  # noqa: E402

MODEL_NAME = "intfloat/multilingual-e5-small"

# ── Positive clusters: same real-world entity, 3 languages each ─────────
# multilingual-e5 wants a "query: " prefix per its own model card — same
# prefix on every string here so it doesn't become a confound between
# clusters.
POSITIVE_CLUSTERS = {
    "narendra_modi": {
        "en": "Narendra Modi is the Prime Minister of India.",
        "hi": "नरेंद्र मोदी भारत के प्रधानमंत्री हैं।",
        "ta": "நரேந்திர மோடி இந்தியாவின் பிரதமர்.",
    },
    "india": {
        "en": "India is a country in South Asia.",
        "hi": "भारत दक्षिण एशिया में एक देश है।",
        "ta": "இந்தியா தெற்காசியாவில் உள்ள ஒரு நாடு.",
    },
    "bangalore": {
        "en": "Bangalore is a major technology hub in India.",
        "hi": "बेंगलुरु भारत में एक प्रमुख तकनीकी केंद्र है।",
        "ta": "பெங்களூரு இந்தியாவின் முக்கிய தொழில்நுட்ப மையம்.",
    },
    "google": {
        "en": "Google is a technology company that builds search engines.",
        "hi": "गूगल एक तकनीकी कंपनी है जो सर्च इंजन बनाती है।",
        "ta": "கூகிள் தேடுபொறிகளை உருவாக்கும் ஒரு தொழில்நுட்ப நிறுவனம்.",
    },
    "rbi": {
        "en": "The Reserve Bank of India regulates the country's banking system.",
        "hi": "भारतीय रिज़र्व बैंक देश की बैंकिंग प्रणाली को नियंत्रित करता है।",
        "ta": "இந்திய ரிசர்வ் வங்கி நாட்டின் வங்கி அமைப்பை கட்டுப்படுத்துகிறது.",
    },
    "mumbai": {
        "en": "Mumbai is the financial capital of India.",
        "hi": "मुंबई भारत की वित्तीय राजधानी है।",
        "ta": "மும்பை இந்தியாவின் நிதி தலைநகரம்.",
    },
    "cricket": {
        "en": "Cricket is the most popular sport in India.",
        "hi": "क्रिकेट भारत में सबसे लोकप्रिय खेल है।",
        "ta": "கிரிக்கெட் இந்தியாவில் மிகவும் பிரபலமான விளையாட்டு.",
    },
    "prime_minister": {
        "en": "The Prime Minister leads the government of India.",
        "hi": "प्रधानमंत्री भारत सरकार का नेतृत्व करते हैं।",
        "ta": "பிரதமர் இந்திய அரசாங்கத்தை வழிநடத்துகிறார்.",
    },
    "sarvam_ai": {
        "en": "Sarvam AI builds multilingual speech and language models for India.",
        "hi": "सर्वम एआई भारत के लिए बहुभाषी स्पीच और लैंग्वेज मॉडल बनाती है।",
        "ta": "சர்வம் AI இந்தியாவிற்காக பன்மொழி பேச்சு மற்றும் மொழி மாதிரிகளை உருவாக்குகிறது.",
    },
    "chennai": {
        "en": "Chennai is a major city on India's southeastern coast.",
        "hi": "चेन्नई भारत के दक्षिण-पूर्वी तट पर एक प्रमुख शहर है।",
        "ta": "சென்னை இந்தியாவின் தென்கிழக்கு கடற்கரையில் உள்ள முக்கிய நகரம்.",
    },
}

# ── Hard negatives: pairs that SHOULD NOT merge despite surface/phonetic
#    similarity — each targets a specific failure mode Codex's review named.
HARD_NEGATIVES = [
    ("sarvam_ai", "en", "hi_sarva_word",
     "The Sanskrit word sarvam means 'everything' or 'all' in Hindi.",
     "polysemous brand name vs. the common Hindi/Sanskrit word it's built from"),
    ("bangalore", "en", "bengal",
     "Bengal is a historical region in eastern India, not the same as Bangalore.",
     "similar-sounding but different real place (Bangalore vs. Bengal)"),
    ("india", "en", "indiana",
     "Indiana is a state in the United States, unrelated to the country India.",
     "similar-sounding but different real place (India vs. Indiana)"),
    ("narendra_modi", "en", "other_modi",
     "Piyush Modi is a shopkeeper in Ahmedabad, a different person entirely.",
     "same surname, different real person"),
    ("chennai", "en", "china",
     "China is a country in East Asia, not the Indian city of Chennai.",
     "similar-sounding but different place (Chennai vs. China)"),
    ("rbi", "en", "reliance",
     "Reliance Industries is a private conglomerate, not the Reserve Bank of India.",
     "two large but unrelated Indian entities"),
]

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def build_dataset():
    """Returns (id -> text) and the ground-truth pair labels."""
    texts = {}
    for entity, langs in POSITIVE_CLUSTERS.items():
        for lang, text in langs.items():
            texts[f"{entity}__{lang}"] = f"query: {text}"

    positive_pairs = set()
    for entity, langs in POSITIVE_CLUSTERS.items():
        ids = [f"{entity}__{lang}" for lang in langs]
        for a, b in combinations(ids, 2):
            positive_pairs.add(frozenset((a, b)))

    hard_negative_pairs = []
    for entity_a, lang_a, neg_id, neg_text, reason in HARD_NEGATIVES:
        id_a = f"{entity_a}__{lang_a}"
        texts[neg_id] = f"query: {neg_text}"
        hard_negative_pairs.append((frozenset((id_a, neg_id)), reason))

    return texts, positive_pairs, hard_negative_pairs


def main():
    hdr("Cross-Lingual Entity-Alias Resolution — Precision/Recall (Sarvam Extension Phase 1)")
    sub(f"Loading {MODEL_NAME} (one-time ~470MB download, cached after)")

    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(MODEL_NAME)
    texts, positive_pairs, hard_negative_pairs = build_dataset()
    ids = list(texts.keys())
    ok(f"Dataset: {len(POSITIVE_CLUSTERS)} entities x up to 3 languages = "
       f"{len(positive_pairs)} positive pairs, {len(hard_negative_pairs)} hard negatives, "
       f"{len(ids)} total texts")

    embeddings = model.encode([texts[i] for i in ids], normalize_embeddings=True)
    # Benign numpy/BLAS matmul RuntimeWarning on this platform — verified
    # by hand that sim_matrix has no NaN/Inf and stays in the expected
    # [-1, 1] range regardless; suppressed here so it doesn't wrongly
    # alarm a future reader into thinking the embeddings are corrupted.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sim_matrix = embeddings @ embeddings.T
    sim = {}
    for i, id_a in enumerate(ids):
        for j, id_b in enumerate(ids):
            if i < j:
                sim[frozenset((id_a, id_b))] = float(sim_matrix[i][j])

    all_pairs = set(sim.keys())
    hard_negative_set = {p for p, _ in hard_negative_pairs}
    easy_negative_pairs = all_pairs - positive_pairs - hard_negative_set

    ok(f"{len(easy_negative_pairs)} easy-negative pairs computed automatically "
       f"(any cross-cluster pair not otherwise labeled)")

    hdr("PRECISION / RECALL AT EACH THRESHOLD")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>6} "
          f"{'TP':>4} {'FP(hard)':>9} {'FP(easy)':>9} {'FN':>4}")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*6} {'-'*4} {'-'*9} {'-'*9} {'-'*4}")

    table = []
    for tau in THRESHOLDS:
        tp = sum(1 for p in positive_pairs if sim[p] >= tau)
        fn = len(positive_pairs) - tp
        fp_hard = sum(1 for p in hard_negative_set if sim[p] >= tau)
        fp_easy = sum(1 for p in easy_negative_pairs if sim[p] >= tau)
        fp = fp_hard + fp_easy
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        table.append({
            "threshold": tau, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp_hard": fp_hard, "fp_easy": fp_easy, "fn": fn,
        })
        print(f"  {tau:>10.2f} {precision:>10.4f} {recall:>8.4f} {f1:>6.4f} "
              f"{tp:>4} {fp_hard:>9} {fp_easy:>9} {fn:>4}")

    hdr("HARD NEGATIVE DETAIL — where does each failure mode actually break?")
    hard_detail = []
    for pair, reason in hard_negative_pairs:
        s = sim[pair]
        # Highest tested threshold this pair STILL incorrectly merges at —
        # max(), not the first ascending match. An earlier version of this
        # script used next() over ascending THRESHOLDS, which always
        # returns the LOWEST threshold any similarity clears (0.70) —
        # made every hard negative look equally (and falsely) mild
        # regardless of how close its actual similarity was to the
        # recommended threshold, and caused the "all correct" claim below
        # to silently miss a pair (Chennai vs. China, sim=0.9010) that
        # actually exceeds the 0.90 recommended threshold. Fixed here.
        still_merges_at = max((t for t in THRESHOLDS if s >= t), default=None)
        hard_detail.append({"reason": reason, "similarity": round(s, 4), "still_merges_at_or_below": still_merges_at})
        status = f"still incorrectly merges up to τ={still_merges_at}" if still_merges_at else "never merges at any tested τ (correct)"
        marker = "✓" if still_merges_at is None else "✗"
        print(f"  [{marker}] sim={s:.4f}  {reason}")
        print(f"        {status}")

    best = max(table, key=lambda t: t["f1"])
    hdr("RECOMMENDATION")
    ok(f"Best F1 at τ={best['threshold']:.2f}: precision={best['precision']:.4f}, "
       f"recall={best['recall']:.4f} (TP={best['tp']}, FP={best['fp_hard']+best['fp_easy']}, FN={best['fn']})")

    risky_hard_negatives = [d for d in hard_detail if d["still_merges_at_or_below"] and d["still_merges_at_or_below"] >= best["threshold"]]
    if risky_hard_negatives:
        warn(f"{len(risky_hard_negatives)} hard-negative failure mode(s) still merge at the "
             f"recommended threshold — do not treat these as solved, they need either a higher "
             f"threshold (trading recall) or a secondary signal (e.g. entity-type agreement, "
             f"Wikidata QID anchoring) before shipping live: "
             + "; ".join(d["reason"] for d in risky_hard_negatives))
    else:
        ok("All hand-labeled hard negatives correctly stay unmerged at the recommended threshold.")

    out = {
        "model": MODEL_NAME, "n_positive_pairs": len(positive_pairs),
        "n_hard_negatives": len(hard_negative_pairs), "n_easy_negatives": len(easy_negative_pairs),
        "table": table, "hard_negative_detail": hard_detail,
        "recommended_threshold": best["threshold"],
    }
    out_path = Path(__file__).parent / "cross_lingual_kg_eval_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    ok(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
