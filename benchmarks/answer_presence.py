"""
ANSWER-PRESENCE DETECTOR ($0, local) — a CALIBRATED instrument.

Why this exists: the lexical check in `fidelity_ladder.py` scores a
correct PARAPHRASE as a loss. Facts are rewritten by construction; raw
turns keep the gold answer's original wording. So lexical matching is
biased AGAINST the fact tier, and every fidelity gap it reports is an
UPPER bound (DECISION_AND_FAILURE_LOG §3.1h).

The fix is NOT "use embeddings and pick a threshold that looks right" —
an unvalidated threshold is the same class of error as an unvalidated
config label (F-14). This module CALIBRATES against ground truth and
publishes its own error rates, so the ladder inherits a known instrument
rather than a plausible one.

GROUND TRUTH USED FOR CALIBRATION
  POSITIVES — single-session questions (single-session-user,
    single-session-assistant, knowledge-update) paired with their OWN gold
    session's raw turns. For these the answer IS present by dataset
    construction: one session holds the evidence.
    Deliberately EXCLUDED: multi-session and temporal-reasoning, where the
    answer must be ASSEMBLED across sessions and is genuinely absent from
    any single one — labelling those "present" would poison the positives.
  NEGATIVES — the same gold answers paired with a RANDOM NON-GOLD session
    from the same question's own haystack. Same corpus, same style, same
    speaker; only the evidence is missing. (A negative drawn from an
    unrelated corpus would make specificity look better than it is.)

HYBRID BY DESIGN
  Numeric gold answers keep the STRICT rule — the number itself must
  appear. Embeddings cannot separate "20 playlists" from "25 playlists",
  and that distinction IS the failure mode under investigation (§3.1a).
  Semantic matching applies ONLY to non-numeric answers.

MODEL: intfloat/multilingual-e5-small — the model the PRODUCT already
uses (db/entity_aliases.py:43), with the model card's "query:"/"passage:"
prefixes. Using a stronger reranker would measure an instrument the
product does not own.
"""
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

_MODEL_NAME = "intfloat/multilingual-e5-small"
_model = None
_CHUNK_MIN = 12


def _get_model():
    global _model
    if _model is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import huggingface_hub.constants as _hc
        _hc.HF_HUB_OFFLINE = True
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def numbers(t):
    return set(re.findall(r"\d+(?:\.\d+)?", t or ""))


def chunks(text, cap=400):
    """Split into comparable units: one per line (facts are line-shaped),
    then sentence-split anything long (turns are paragraph-shaped)."""
    out = []
    for line in (text or "").split("\n"):
        line = line.strip()
        if len(line) < _CHUNK_MIN:
            continue
        if len(line) <= 300:
            out.append(line)
        else:
            for s in re.split(r"(?<=[.!?])\s+", line):
                if len(s.strip()) >= _CHUNK_MIN:
                    out.append(s.strip())
    return out[:cap]


def max_similarity(gold, text, model=None):
    """Best cosine between the gold answer and any chunk."""
    cs = chunks(text)
    if not cs:
        return 0.0
    m = model or _get_model()
    import numpy as np
    # float32 EXPLICITLY: the encoder can hand back float16 on this
    # backend, and float16 matmul over a 400-chunk batch overflowed to
    # inf/NaN — silently corrupting the similarities the calibration is
    # derived from. Caught by RuntimeWarning on the first calibration run.
    q = m.encode([f"query: {gold}"], normalize_embeddings=True,
                 show_progress_bar=False).astype(np.float32)
    p = m.encode([f"passage: {c}" for c in cs], normalize_embeddings=True,
                 batch_size=128, show_progress_bar=False).astype(np.float32)
    # Degenerate chunks (punctuation-only) normalize to a zero vector ->
    # NaN row. Drop them rather than let NaN propagate into max().
    ok = np.isfinite(p).all(axis=1)
    if not ok.any():
        return 0.0
    sims = q @ p[ok].T
    sims = sims[np.isfinite(sims)]
    return float(sims.max()) if sims.size else 0.0


def recoverable(gold, text, tau, model=None):
    """HYBRID: numbers must survive literally; otherwise semantic."""
    if not text:
        return False
    gn = numbers(gold)
    if gn:
        return all(n in numbers(text) for n in gn)
    return max_similarity(gold, text, model) >= tau


# ── Calibration ────────────────────────────────────────────────────────
def calibrate(verbose=True):
    import random
    import numpy as np
    from corpus_loaders import load_longmemeval

    ds = load_longmemeval(n_queries=150, seed=42, split="s")
    mem = {m.mid: m for m in ds.memories}
    rng = random.Random(42)
    SINGLE = {"single-session-user", "single-session-assistant",
              "knowledge-update"}

    pos, neg = [], []
    for q in ds.queries:
        if q.question_type not in SINGLE or numbers(q.gold_answer):
            continue                      # numeric answers use the strict rule
        gold_txt = " ".join(t.get("content", "") for k in q.gold_keys
                            if k in mem for t in mem[k].turns)
        others = [k for k in q.scope_keys if k not in q.gold_keys and k in mem]
        if not gold_txt or not others:
            continue
        other = mem[rng.choice(others)]
        pos.append((q.gold_answer, gold_txt))
        neg.append((q.gold_answer, " ".join(t.get("content", "")
                                            for t in other.turns)))

    m = _get_model()
    if verbose:
        print(f"calibration set: {len(pos)} positives / {len(neg)} negatives "
              f"(non-numeric single-session questions)")
    ps = np.array([max_similarity(g, t, m) for g, t in pos])
    ns = np.array([max_similarity(g, t, m) for g, t in neg])

    best = None
    for tau in np.arange(0.60, 0.96, 0.005):
        sens = float((ps >= tau).mean())
        spec = float((ns < tau).mean())
        bal = (sens + spec) / 2
        if best is None or bal > best[3]:
            best = (tau, sens, spec, bal)
    tau, sens, spec, bal = best
    if verbose:
        print(f"  positives  mean={ps.mean():.3f}  median={np.median(ps):.3f}")
        print(f"  negatives  mean={ns.mean():.3f}  median={np.median(ns):.3f}")
        print(f"  CHOSEN tau={tau:.3f}  sensitivity={sens:.1%}  "
              f"specificity={spec:.1%}  balanced={bal:.1%}")
        lex_sens = float(np.mean([_lex(g, t) for g, t in pos]))
        lex_spec = float(np.mean([not _lex(g, t) for g, t in neg]))
        print(f"  (lexical baseline on the SAME set: sensitivity="
              f"{lex_sens:.1%}  specificity={lex_spec:.1%})")
    return float(tau), sens, spec


_STOP = {"the", "and", "for", "with", "that", "this", "you", "your", "was",
         "were", "have", "has", "had", "are", "not", "but", "from", "they",
         "their", "them", "when", "what", "which", "would", "could",
         "should", "about", "into", "user", "users", "there", "then",
         "than", "some", "just", "also", "been", "will", "can", "may",
         "his", "her", "its", "our", "out", "one", "all"}


def _lex(gold, hay):
    gw = {w for w in re.findall(r"[a-z]{3,}", gold.lower()) if w not in _STOP}
    if not gw:
        return gold.strip().lower() in (hay or "").lower()
    hw = {w for w in re.findall(r"[a-z]{3,}", (hay or "").lower())
          if w not in _STOP}
    return len(gw & hw) / len(gw) >= 0.60


if __name__ == "__main__":
    calibrate()
