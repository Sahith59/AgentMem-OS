"""Extraction-model preflight: gpt-4o-mini vs gpt-5.6-luna, head to
head on 25 real full-turn sessions with the EXACT production
consolidation prompt. Measures, per model: real token consumption
(from API usage fields — the basis for corpus cost arithmetic), facts
per session, wall time, and needle survival (known details that a
faithful note-taker must capture). ~$0.50 total.
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

for line in (HERE.parent / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

# Needles: (search string in session text, list of substrings at least
# one of which must appear in the extracted facts for survival)
NEEDLES = [
    ("70-200mm zoom lens", ["70-200"]),
    ("portable power bank", ["power bank"]),
    ("Premier Silver", ["premier silver"]),
    ("shoe rack", ["shoe rack"]),
    ("marigold", ["marigold"]),
]

PROMPT_TMPL = """You are a memory consolidation engine. Extract atomic facts about the USER from this conversation session.

Rules:
- Each fact is ONE self-contained proposition. Name "the user", never bare pronouns. Include the concrete details (names, numbers, dates) IN the fact text — a fact must make sense alone, months later.
- PRESERVE exactly: counts, quantities, prices, dates, times, schedules, proper names. Never round, merge, or drop a number. If the user did something N times, the fact states N.
- fact_type: "event" = something that happened at a specific time, OR a plan/appointment for a SPECIFIC FUTURE DATE ("The user plans to attend X on DATE" is an event with t_occurred = that date; the system marks future-dated events as planned automatically). A plan with NO stated date is a "state". "state" = an ongoing situation; "preference" = a like/dislike/choice; "identity" = who the user is.
- t_occurred: the date the event happened — or, for a planned event, the date it is planned FOR — YYYY/MM/DD (or YYYY/MM if only the month is known) — resolve relative references ("last Tuesday", "two weeks ago") against the session date {session_date}, NEVER against today. null if no date is stated or implied.
- NEVER extract a fact that merely restates what the user ASKED or wants to learn — questions and curiosity are not facts about the user's life.
- ONLY facts stated by the USER about their own life. Never extract the assistant's knowledge, recommendations, availability information, or tool/system output.

Return a JSON object: {{"facts": [{{"text": ..., "fact_type": ..., "t_occurred": ..., "source_turns": [...]}}]}}

Session date: {session_date}
Transcript:
{transcript}"""

PRICES = {  # $ per 1M tokens (input, output)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.6-luna": (0.20, 1.20),
}


def call(model, prompt):
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}}
    if model.startswith("gpt-5"):
        body["max_completion_tokens"] = 6000
    else:
        body["max_tokens"] = 3000
        body["temperature"] = 0
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                 "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    usage = out["usage"]
    content = out["choices"][0]["message"]["content"] or ""
    try:
        facts = json.loads(content).get("facts", [])
    except Exception:
        m = re.search(r"\{.*\}", content, re.S)
        facts = json.loads(m.group(0)).get("facts", []) if m else []
    return facts, usage["prompt_tokens"], usage["completion_tokens"], dt


def main():
    ds = json.load(open(HERE / "benchmark_cache/longmemeval_s.json"))
    mems = ds["memories"]
    import random
    rng = random.Random(42)

    picked, picked_ids = [], set()
    for needle, keys in NEEDLES:
        for m in mems:
            if needle in m["content"] and m["mid"] not in picked_ids:
                picked.append((m, keys))
                picked_ids.add(m["mid"])
                break
    pool = [m for m in mems if m["mid"] not in picked_ids]
    for m in rng.sample(pool, 25 - len(picked)):
        picked.append((m, []))
    print(f"{len(picked)} sessions ({sum(1 for _,k in picked if k)} needle)")

    report = {}
    for model in PRICES:
        rows, t0 = [], time.time()
        for i, (m, keys) in enumerate(picked):
            date = (m.get("dates") or ["2023/01/01"])[0] if isinstance(
                m.get("dates"), list) else "2023/01/01"
            transcript = m["content"][:60000]
            try:
                facts, pt, ct, dt = call(
                    model, PROMPT_TMPL.format(session_date=date,
                                              transcript=transcript))
            except Exception as e:
                print(f"  {model} ERROR on {m['mid']}: {e}")
                facts, pt, ct, dt = [], 0, 0, 0
            blob = " ".join(str(f.get("text", "")) for f in facts).lower()
            survived = [k for k in keys if k.lower() in blob]
            rows.append({"mid": m["mid"], "n_facts": len(facts),
                         "pt": pt, "ct": ct, "sec": round(dt, 1),
                         "needles": keys, "survived": survived,
                         "facts_sample": [str(f.get("text"))[:90]
                                          for f in facts[:4]]})
            print(f"  {model} {i+1}/{len(picked)} facts={len(facts)} "
                  f"pt={pt} ct={ct} {dt:.0f}s"
                  + (f" needles {len(survived)}/{len(keys)}" if keys else ""),
                  flush=True)
        pin, pout = PRICES[model]
        tot_pt = sum(r["pt"] for r in rows)
        tot_ct = sum(r["ct"] for r in rows)
        n = len(rows)
        corpus_cost = (tot_pt / n * 19195 * pin + tot_ct / n * 19195 * pout) / 1e6
        needle_hit = sum(len(r["survived"]) for r in rows)
        needle_tot = sum(len(r["needles"]) for r in rows)
        report[model] = {
            "sessions": n, "mean_facts": round(sum(r["n_facts"] for r in rows) / n, 1),
            "mean_prompt_tokens": round(tot_pt / n),
            "mean_completion_tokens": round(tot_ct / n),
            "mean_seconds": round(sum(r["sec"] for r in rows) / n, 1),
            "needle_survival": f"{needle_hit}/{needle_tot}",
            "est_corpus_cost_usd": round(corpus_cost, 2),
            "preflight_cost_usd": round((tot_pt * pin + tot_ct * pout) / 1e6, 3),
            "rows": rows,
        }
        print(f"== {model}: facts/sess {report[model]['mean_facts']} | "
              f"tok {report[model]['mean_prompt_tokens']}in/"
              f"{report[model]['mean_completion_tokens']}out | "
              f"needles {report[model]['needle_survival']} | "
              f"corpus est ${report[model]['est_corpus_cost_usd']}", flush=True)

    json.dump(report, open(HERE / "extraction_preflight_results.json", "w"),
              indent=1, ensure_ascii=False)
    print("wrote extraction_preflight_results.json")


if __name__ == "__main__":
    main()
