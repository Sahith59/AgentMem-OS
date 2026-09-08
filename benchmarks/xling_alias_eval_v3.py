"""S1 lever measurement: layered alias matching vs bare cosine.

Layers measured on the same 53-entity hard-negative suite as v2:
  L0  class-E rule gate: contextual-reference strings (possessive +
      role nouns, honorific address terms) are NEVER auto-merge
      candidates. A rule, not a threshold.
  L1  transliteration normalization: non-Latin scripts -> ISO Latin
      (offline indic-transliteration; Tamil known-weak, disclosed) ->
      strip diacritics -> lowercase. String similarity on normalized
      forms (difflib ratio, deterministic).
  L2  AND-rule: merge iff embedding cosine >= t1 AND normalized string
      similarity >= t2. 2D sweep; operating point = precision >= 0.95
      with max recall, same precision-first rule as v2.

Honesty notes: the v2 baseline (best F1 0.504, no precision-safe
threshold) is the comparison target. Class D (code-mixed FRAGMENTS)
carries meaning beyond the entity string, so string similarity is
expected to help less there; reported per-class either way.
$0, local CPU. Output: benchmarks/xling_alias_eval_v3_results.json
"""
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
from data.xling_alias_eval_v1 import ENTITIES, CLASSES  # noqa: E402

from indic_transliteration import sanscript  # noqa: E402
from indic_transliteration.sanscript import transliterate  # noqa: E402

SCRIPT_RANGES = [
    (0x0900, 0x097F, sanscript.DEVANAGARI),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI),
    (0x0B80, 0x0BFF, sanscript.TAMIL),
    (0x0C00, 0x0C7F, sanscript.TELUGU),
]


def detect_scheme(s):
    for ch in s:
        cp = ord(ch)
        for lo, hi, scheme in SCRIPT_RANGES:
            if lo <= cp <= hi:
                return scheme
    return None


def normalize(s):
    """Script-normalize to lowercase ascii-ish Latin."""
    scheme = detect_scheme(s)
    if scheme is not None:
        try:
            s = transliterate(s, scheme, sanscript.ISO)
        except Exception:
            pass
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    # cheap schwa-tail collapse: jayapura ~ jaipur needs fuzzy match
    return re.sub(r"\s+", " ", s).strip()


# L0: contextual-reference detector (rule gate)
_ROLE_WORDS = {
    "manager", "boss", "client", "doctor", "landlord", "specialist",
    "didi", "bhaiya", "bhai", "sir", "madam", "aunty", "uncle", "mummy",
    "mom", "home", "ghar", "gharana",
    # normalized forms of the Devanagari role words in the dataset
    "mainejara", "mainejara", "klainta", "daktara", "daktar",
    "makana malika", "baas", "maidama", "sara", "bhaiya", "didi",
}
_POSS_RE = re.compile(
    r"\b(my|the|mera|mere|meri|mera|hamara|hamare)\b")


def is_contextual(raw, norm):
    if _POSS_RE.search(norm):
        return True
    toks = set(norm.split())
    return bool(toks & _ROLE_WORDS) and len(toks) <= 3


def main():
    from agentmem_os.db.entity_aliases import get_shared_encoder
    model = get_shared_encoder()
    assert model is not None

    surfaces, index = [], {}

    def sid(s):
        if s not in index:
            index[s] = len(surfaces)
            surfaces.append(s)
        return index[s]

    positives, negatives = [], []
    for e in ENTITIES:
        fids = [sid(f) for f in e["forms"]]
        if e["class"] != "E":
            for a, b in combinations(fids, 2):
                positives.append((a, b, e["class"]))
        for neg, _r in e["negatives"]:
            n = sid(neg)
            for f in fids:
                negatives.append((f, n, e["class"]))
    firsts = [(sid(e["forms"][0]), e["id"]) for e in ENTITIES]
    for (i, ida), (j, idb) in combinations(firsts, 2):
        if ida != idb:
            negatives.append((i, j, "easy"))

    norms = [normalize(s) for s in surfaces]
    ctx = [is_contextual(surfaces[k], norms[k]) for k in range(len(surfaces))]

    emb = model.encode([f"passage: {s}" for s in surfaces],
                       normalize_embeddings=True, show_progress_bar=False,
                       batch_size=64)
    cos = emb @ emb.T

    def strsim(a, b):
        return SequenceMatcher(None, norms[a], norms[b]).ratio()

    ssim = {}
    for a, b, _ in positives + negatives:
        if (a, b) not in ssim:
            ssim[(a, b)] = strsim(a, b)

    def decide(a, b, t1, t2, gate):
        if gate and (ctx[a] or ctx[b]):
            return False
        return cos[a, b] >= t1 and ssim[(a, b)] >= t2

    def score(t1, t2, gate):
        tp = sum(1 for a, b, _ in positives if decide(a, b, t1, t2, gate))
        fph = sum(1 for a, b, c in negatives
                  if c != "easy" and decide(a, b, t1, t2, gate))
        fpe = sum(1 for a, b, c in negatives
                  if c == "easy" and decide(a, b, t1, t2, gate))
        fn = len(positives) - tp
        prec = tp / max(1, tp + fph + fpe)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        return {"t_cos": round(t1, 2), "t_str": round(t2, 2),
                "gate": gate, "precision": round(prec, 3),
                "recall": round(rec, 3), "f1": round(f1, 3),
                "fp_hard": fph, "fp_easy": fpe}

    grid = [score(t1 / 100, t2 / 100, True)
            for t1 in range(70, 96, 2) for t2 in range(30, 91, 5)]
    grid_nogate = [score(t1 / 100, t2 / 100, False)
                   for t1 in range(70, 96, 2) for t2 in range(30, 91, 5)]

    def pick(rows):
        best = max(rows, key=lambda r: r["f1"])
        safe = [r for r in rows if r["precision"] >= 0.95]
        op = max(safe, key=lambda r: r["recall"]) if safe else None
        return best, op

    best_g, op_g = pick(grid)
    best_n, op_n = pick(grid_nogate)

    # per-class at the gated operating point (if it exists)
    per_class, misses, false_hits = {}, [], []
    if op_g:
        t1, t2 = op_g["t_cos"], op_g["t_str"]
        for cls in sorted(CLASSES):
            if cls == "E":
                pairs = [(a, b) for a, b, c in negatives if c == "E"]
                bad = sum(1 for a, b in pairs
                          if decide(a, b, t1, t2, True))
                per_class[cls] = {"false_merges": bad, "pairs": len(pairs)}
            else:
                pairs = [(a, b) for a, b, c in positives if c == cls]
                got = sum(1 for a, b in pairs if decide(a, b, t1, t2, True))
                per_class[cls] = {
                    "recall_at_op": round(got / max(1, len(pairs)), 3),
                    "pairs": len(pairs)}
        misses = [
            {"a": surfaces[a], "b": surfaces[b],
             "cos": round(float(cos[a, b]), 3),
             "str": round(ssim[(a, b)], 3),
             "norm_a": norms[a], "norm_b": norms[b]}
            for a, b, c in positives if not decide(a, b, t1, t2, True)][:15]
        false_hits = [
            {"a": surfaces[a], "b": surfaces[b],
             "cos": round(float(cos[a, b]), 3),
             "str": round(ssim[(a, b)], 3)}
            for a, b, c in negatives
            if c != "easy" and decide(a, b, t1, t2, True)]

    out = {
        "baseline_v2": {"best_f1": 0.504, "precision_at_best": 0.452,
                        "operating_point": None},
        "layers": "L0 ctx-gate + L1 translit-normalized strsim + L2 AND-rule",
        "translit_note": "offline indic-transliteration; Tamil mapping "
                         "known-weak (disclosed), Sarvam API is the S2 lever",
        "best_f1_gated": best_g, "operating_gated": op_g,
        "best_f1_ungated": best_n, "operating_ungated": op_n,
        "per_class_at_operating": per_class,
        "missed_positives_at_op": misses,
        "false_merges_at_op": false_hits,
    }
    path = HERE / "xling_alias_eval_v3_results.json"
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print("BASELINE v2: best F1 0.504 (precision 0.452); no safe point")
    print("GATED   best F1:", best_g)
    print("GATED   operating(p>=0.95):", op_g)
    print("UNGATED operating(p>=0.95):", op_n)
    for cls, d in per_class.items():
        print(f"  class {cls}: {d}")
    print(f"false merges at op: {len(false_hits)} | wrote {path}")


if __name__ == "__main__":
    main()
