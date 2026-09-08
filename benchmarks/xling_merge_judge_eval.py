"""LLM merge-judge prototype, scored on the 53-entity trap suite.

The probabilistic decision layer from SARVAM_TRACK's corrected
architecture: a model reads two entity mentions and judges whether
they refer to the same specific thing, answering MERGE / NO_MERGE /
NEED_CONTEXT with a confidence. NEED_CONTEXT counts as NO_MERGE for
auto-merge scoring (the safe refusal is a correct behavior on
context-dependent pairs, and is reported separately).

Standalone by design: needs only the dataset file + an ollama server
(OLLAMA_URL env, default localhost:11434). Runs on the cluster so the
Mac's headline run is untouched. ~344 judged pairs, $0.

Usage: python3 xling_merge_judge_eval.py [model]
Output: xling_merge_judge_results.json
"""
import json
import os
import re
import sys
import time
import urllib.request
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from data.xling_alias_eval_v1 import ENTITIES, CLASSES  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama3.1:latest"
URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

# .env from repo root (keys for the API judges)
_env = HERE.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.strip().startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

PROMPT = """You decide whether two mentions refer to the SAME specific \
real-world entity. Mentions may be in different languages or scripts \
(English, Hindi, Tamil, Telugu, Punjabi, romanized Indic, code-mixed).

Rules:
- MERGE only if they must be the same specific entity (e.g. the same \
city written in two scripts, or a well-known alias of the same place).
- NO_MERGE if they are different entities, or one is a brand/product \
while the other is an ordinary word, or the names are similar but not \
the same (one changed letter can be a different person).
- NEED_CONTEXT if it depends on who is speaking or on surrounding \
conversation (e.g. relative references like "my manager", honorifics \
like "didi", a nickname that could be several people).

Mention A: {a}
Mention B: {b}

Answer with exactly one line in this format:
VERDICT=<MERGE|NO_MERGE|NEED_CONTEXT> CONFIDENCE=<0.0-1.0>"""


def judge(a, b, retries=2):
    body = json.dumps({
        "model": MODEL,
        "prompt": PROMPT.format(a=a, b=b),
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 40},
    }).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                f"{URL}/api/generate", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                text = json.loads(r.read())["response"]
            m = re.search(r"VERDICT=(MERGE|NO_MERGE|NEED_CONTEXT)", text)
            c = re.search(r"CONFIDENCE=([0-9.]+)", text)
            if m:
                return m.group(1), float(c.group(1)) if c else 0.5, text
        except Exception:
            if attempt == retries:
                return "ERROR", 0.0, ""
            time.sleep(2)
    return "ERROR", 0.0, ""




def judge_api(a, b, retries=2):
    """OpenAI-compatible chat path (OpenAI, Sarvam). Chosen when MODEL
    is not an ollama tag. Sarvam uses api-subscription-key header."""
    if MODEL.startswith("sarvam"):
        url = "https://api.sarvam.ai/v1/chat/completions"
        headers = {"api-subscription-key": os.environ["SARVAM_API_KEY"]}
    else:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    headers["Content-Type"] = "application/json"
    # sarvam-105b is a reasoning model: it spends tokens on
    # reasoning_content BEFORE the answer, and a tight cap starves the
    # verdict entirely (339/344 ERROR on the first attempt — measured).
    body = json.dumps({
        "model": MODEL, "temperature": 0,
        "messages": [{"role": "user", "content": PROMPT.format(a=a, b=b)}],
        "max_tokens": 1500 if MODEL.startswith("sarvam") else 40,
    }).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as r:
                msg = json.loads(r.read())["choices"][0]["message"]
                text = (msg.get("content") or "") + "\n" + \
                    (msg.get("reasoning_content") or "")
            m = re.search(r"VERDICT=(MERGE|NO_MERGE|NEED_CONTEXT)", text)
            c = re.search(r"CONFIDENCE=([0-9.]+)", text)
            if m:
                return m.group(1), float(c.group(1)) if c else 0.5, text
        except Exception:
            if attempt == retries:
                return "ERROR", 0.0, ""
            time.sleep(2)
    return "ERROR", 0.0, ""

def main():
    pairs = []  # (a, b, label, species)
    for e in ENTITIES:
        forms = e["forms"]
        if e["class"] != "E":
            for a, b in combinations(forms, 2):
                pairs.append((a, b, "MERGE", e["class"]))
        for neg, _r in e["negatives"]:
            for f in forms:
                pairs.append((f, neg, "NO_MERGE", e["class"]))

    print(f"{len(pairs)} pairs -> {MODEL} at {URL}", flush=True)
    results, t0 = [], time.time()
    for i, (a, b, label, cls) in enumerate(pairs):
        fn = judge_api if ("gpt" in MODEL or MODEL.startswith("sarvam")) else judge
        v, conf, raw = fn(a, b)
        results.append({"a": a, "b": b, "label": label, "class": cls,
                        "verdict": v, "confidence": conf})
        if (i + 1) % 25 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(pairs)} | {rate*60:.0f}/min", flush=True)

    def eff(v):  # NEED_CONTEXT and ERROR act as NO_MERGE (safe refusal)
        return "MERGE" if v == "MERGE" else "NO_MERGE"

    tp = sum(1 for r in results
             if r["label"] == "MERGE" and eff(r["verdict"]) == "MERGE")
    fp = sum(1 for r in results
             if r["label"] == "NO_MERGE" and eff(r["verdict"]) == "MERGE")
    fn = sum(1 for r in results
             if r["label"] == "MERGE" and eff(r["verdict"]) == "NO_MERGE")
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    per_class = {}
    for cls in sorted(CLASSES):
        rs = [r for r in results if r["class"] == cls]
        if cls == "E":
            bad = sum(1 for r in rs if eff(r["verdict"]) == "MERGE")
            nc = sum(1 for r in rs if r["verdict"] == "NEED_CONTEXT")
            per_class[cls] = {"false_merges": bad, "need_context": nc,
                              "pairs": len(rs)}
        else:
            p = [r for r in rs if r["label"] == "MERGE"]
            n = [r for r in rs if r["label"] == "NO_MERGE"]
            per_class[cls] = {
                "recall": round(sum(1 for r in p
                                    if eff(r["verdict"]) == "MERGE")
                                / max(1, len(p)), 3),
                "false_merges": sum(1 for r in n
                                    if eff(r["verdict"]) == "MERGE"),
                "neg_pairs": len(n)}

    errors = sum(1 for r in results if r["verdict"] == "ERROR")
    out = {"model": MODEL, "pairs": len(pairs),
           "precision": round(prec, 3), "recall": round(rec, 3),
           "f1": round(f1, 3), "errors": errors,
           "need_context_total": sum(1 for r in results
                                     if r["verdict"] == "NEED_CONTEXT"),
           "per_class": per_class,
           "false_merge_examples": [
               {"a": r["a"], "b": r["b"], "conf": r["confidence"]}
               for r in results if r["label"] == "NO_MERGE"
               and eff(r["verdict"]) == "MERGE"][:15],
           "missed_examples": [
               {"a": r["a"], "b": r["b"], "verdict": r["verdict"]}
               for r in results if r["label"] == "MERGE"
               and eff(r["verdict"]) == "NO_MERGE"][:15],
           "results": results}
    safe = MODEL.replace(":", "_").replace("/", "_")
    path = HERE / f"xling_merge_judge_results_{safe}.json"
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"\nJUDGE {MODEL}: precision={prec:.3f} recall={rec:.3f} "
          f"f1={f1:.3f} errors={errors}")
    for cls, d in per_class.items():
        print(f"  class {cls}: {d}")
    print("wrote", path)


if __name__ == "__main__":
    main()
