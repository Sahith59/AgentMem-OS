"""
LoCoMo / LongMemEval dataset loaders — ported and adapted from X-MemoryArch
(RetrievalEngine/benchmark_4approaches.py), a separate project by the same
author with an already-tested, already-bug-fixed loader for these exact
datasets. See LAUNCH_ROADMAP.md Phase 1 Group D and
agentmem_os_xmemoryarch_reuse.md for the reuse rationale.

Deliberately NOT ported from X-MemoryArch's qa_accuracy_eval.py, whose
loaders read precomputed LLM-extraction JSON — that shortcut is right for
X-MemoryArch's own retrieval engine but wrong here, since Phase 2 of this
project's roadmap needs every competitor library to run its OWN extraction
over the same raw dialogue turns. These loaders build directly from the
raw session/turn data, not from any precomputed "memory" representation.

Bug-fix history preserved from the source (do not re-truncate more
aggressively than this): LoCoMo sessions were originally capped at 1,200
chars during extraction, silently discarding 58% of every session (mean
raw session is ~2,842 chars, max ~5,867) — the dominant cause of missed
facts in the back half of every conversation. Fixed upstream to a 6,000
char cap; kept here.

Usage:
    from agentmem_os.benchmarks.corpus_loaders import load_locomo, load_longmemeval
    ds = load_locomo(n_queries=150)
    ds = load_longmemeval(n_queries=100)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).resolve().parent / "benchmark_cache"


@dataclass
class MemEntry:
    """One session-level memory unit — a conversation session (LoCoMo) or
    haystack session (LongMemEval)."""
    mid: str
    title: str
    content: str            # formatted "speaker: text" transcript, capped
    search_text: str        # text to index/embed for retrieval (== content here)
    gold_key: str           # this session's own ID, used as a relevance label
    turns: list = field(default_factory=list)  # [{"role": str, "content": str}, ...]
    # raw per-turn structure for adapters that need to run their own
    # extraction over individual turns rather than one joined blob
    # (see LAUNCH_ROADMAP.md Phase 2 — real baseline adapters)
    facts: list = field(default_factory=list)
    # LLM-extracted, date-grounded atomic memories for this session, when a
    # cache is present — see attach_extracted_memories(). Empty otherwise.


EXTRACTED_DIR = Path(__file__).resolve().parent / "extracted_memories"

# LLM-extracted memory caches, keyed by dataset. Produced by X-MemoryArch
# (github.com/Sahith59/X-MemoryArch, same author) with Claude Sonnet
# (LoCoMo) / Haiku (LongMemEval) and released here so this benchmark is
# reproducible at $0 instead of requiring every runner to pay for their own
# extraction pass. Format: {session_id: [{"memory": str, "session_date":
# str|None, "memory_type": str, ...}, ...]} — session_id matches the ids
# these loaders build, so the join is exact.
_EXTRACTED_FILES = {
    "LoCoMo": "rich_memories_sonnet_LoCoMo.json",
    "LongMemEval": "rich_memories_haiku_LongMemEval.json",
}


def attach_extracted_memories(ds: "BenchDataset") -> int:
    """
    Populate MemEntry.facts from the extracted-memory cache for this
    dataset. Returns how many sessions got facts (0 if no cache is
    present — callers then fall back to raw turns).

    Why this exists: storage granularity is the single biggest measured
    lever in this problem — X-MemoryArch's own conclusion after a full
    phase of experiments was "the biggest lever is what you store, not how
    you retrieve it", and AgentMem OS was storing raw dialogue turns while
    every competitor in the harness (mem0, langmem, letta) runs its own LLM
    extraction at ingest. Feeding raw turns to one system and extracted
    facts to the others is not a fair comparison; this closes that gap.
    """
    fname = _EXTRACTED_FILES.get(ds.name)
    if not fname:
        return 0
    path = EXTRACTED_DIR / fname
    if not path.exists():
        return 0

    cache = json.loads(path.read_text())
    hits = 0
    for mem in ds.memories:
        records = cache.get(mem.mid)
        if not records:
            continue
        mem.facts = [
            {
                "content": r["memory"],
                "date": r.get("session_date"),
                "type": r.get("memory_type", "state"),
            }
            for r in records
            if r.get("memory")
        ]
        if mem.facts:
            hits += 1
    return hits


def facts_as_turns(mem: "MemEntry") -> list:
    """
    A session's extracted facts in the {"role", "content"} shape every
    adapter's ingest_session already speaks, so switching granularity needs
    no adapter changes. Dates are prefixed into the text (not dropped into
    metadata no retriever reads) because temporal grounding in the memory
    string itself is what makes "when/how long ago" questions answerable.
    """
    out = []
    for f in mem.facts:
        text = f["content"]
        if f.get("date") and f["date"] not in text:
            text = f"[{f['date']}] {text}"
        out.append({"role": "user", "content": text})
    return out


@dataclass
class QueryEntry:
    question: str
    gold_keys: list          # list[gold_key] — all session IDs that answer this
    scope_keys: list = field(default_factory=list)
    # every session ID that should be searched to answer this question (the
    # "haystack") — a superset of gold_keys. For LoCoMo this is every
    # session in the same conversation; for LongMemEval it's the dataset's
    # own haystack_session_ids. Needed for QA-accuracy eval (Group E): the
    # model must find the answer among a real haystack, not be handed only
    # the gold sessions directly.
    gold_answer: str = ""
    # the actual answer text, for QA-accuracy scoring (retrieve->generate->
    # judge). Empty for questions without one — callers doing QA-accuracy
    # eval should filter those out (see qa_accuracy_eval.py).
    question_date: str = ""
    # the "today" the question is asked relative to. LongMemEval ships this
    # per question and 133/500 of its questions are temporal-reasoning
    # ("how many days ago did I meet Emma?"). Without it those questions are
    # UNANSWERABLE no matter how good retrieval is — measured: the oracle
    # answerer replied "not mentioned" to every one of them. Never drop it.
    question_type: str = ""
    # LongMemEval's own category label, kept so results can be broken down
    # per category instead of hiding a 30-question preference-following
    # subtask inside a factual-QA average.


@dataclass
class BenchDataset:
    name: str
    memories: list           # list[MemEntry]
    queries: list            # list[QueryEntry]


def _cached(path: Path, fn):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return json.loads(path.read_text())
    result = fn()
    path.write_text(json.dumps(result, ensure_ascii=False))
    return result


# LoCoMo category id -> what the questions in it ACTUALLY are, measured from
# the released data rather than copied from the ecosystem's labels.
#
# The Mem0-derived harness that Memobase, Backboard, Hindsight and others all
# forked maps 1->single_hop, 2->temporal, 3->multi_hop, 4->open_domain. That
# mapping is wrong for 1, 3 and 4, and it is checkable in one pass over the
# evidence field: the LoCoMo paper defines multi-hop as needing several
# sessions and single-hop as answerable from one.
#
#   cat 1: n=282, mean 2.68 evidence sessions, 95.4% span >1 session -> MULTI-HOP
#   cat 2: n=321, mean 1.10,                    8.8%                 -> temporal (correct)
#   cat 3: n= 96, mean 1.75,                   34.8%                 -> open-domain/commonsense
#   cat 4: n=841, mean 1.00,                    0.1%                 -> SINGLE-HOP
#   cat 5: n=446 adversarial — no usable ground truth, excluded by everyone
#
# Cross-check: the paper puts open-domain at 3.9% of the set, which cannot be
# the 841-question column; single-hop is the largest category, which it must be.
LOCOMO_CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

# Categories 1-4 = 1,540 questions, the pool every published LoCoMo number
# uses (category 5 excluded for missing ground truth). Default to it so our
# numbers are comparable; pass a narrower set to evaluate the hard subset.
LOCOMO_DEFAULT_CATEGORIES = (1, 2, 3, 4)


def load_locomo(n_queries: int, cache_dir: Path = CACHE_DIR, seed: int = 42,
                 categories: tuple = LOCOMO_DEFAULT_CATEGORIES) -> BenchDataset:
    """
    LoCoMo (Maharana et al., ACL 2024).

    Note what the public release actually is: the paper describes 50
    conversations / 7,512 questions, but `data/locomo10.json` — the only
    released file, and the basis of every published LoCoMo number in
    existence — holds **10 conversations / 1,986 questions**. Paper baselines
    and vendor numbers are therefore not comparable to each other.

    categories: which question categories to keep. Defaults to (1,2,3,4) =
    1,540 questions, the industry-standard pool. This loader previously kept
    only (1,2,3) = 699 questions, which silently EXCLUDED category 4 — the
    841 single-hop questions that are the easiest bulk of the set (0.1% span
    more than one session). Numbers produced under that filter are measured
    on a substantially harder subset and must never be compared against a
    1,540-question number.
    """
    print("  Loading LoCoMo...")
    cats_tag = "".join(str(c) for c in sorted(categories))
    cache = cache_dir / (f"locomo_c{cats_tag}.json")

    def _build():
        url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
        mems, qs = [], []
        for conv_idx, conv in enumerate(data):
            conv_data = conv["conversation"]
            session_keys = sorted(
                k for k in conv_data if k.startswith("session_") and "date" not in k
            )

            # Map dia_id -> session_key within this conversation, to resolve
            # QA evidence (turn-level) into session-level gold keys.
            dia_to_session = {}
            conv_session_ids = []  # every session_id ingested for this conv — the haystack scope

            for sk in session_keys:
                turns_raw = conv_data[sk]
                lines = []
                structured_turns = []
                for turn in turns_raw:
                    dia_id = turn.get("dia_id", "")
                    speaker = turn.get("speaker", "?")
                    text = turn.get("text", "").strip()
                    if dia_id:
                        dia_to_session[dia_id] = sk
                    if text:
                        lines.append(f"{speaker}: {text}")
                        structured_turns.append({"role": speaker, "content": text})

                if not lines:
                    continue

                # Full session context, capped at 6,000 chars (max raw
                # session is 5,867) — NOT the original 1,200-char cap that
                # discarded 58% of every session. See module docstring.
                session_id = f"c{conv_idx}_{sk}"
                date_val = conv_data.get(f"{sk}_date", "")
                # Same reason as LongMemEval: the date has to live IN the
                # retrievable text, not only in the title.
                header = f"Session dated {date_val}\n" if date_val else ""
                session_content = (header + "\n".join(lines))[:6000]
                title = f"Conv{conv_idx+1} {sk}" + (f" ({date_val})" if date_val else "")

                mems.append({
                    "mid": session_id, "title": title,
                    "content": session_content, "search_text": session_content,
                    "gold_key": session_id, "turns": structured_turns,
                })
                conv_session_ids.append(session_id)

            for qa in conv["qa"]:
                evidence = qa.get("evidence", [])
                if not evidence:
                    continue
                if qa.get("category", 99) not in categories:
                    continue
                gold_sessions = list({
                    f"c{conv_idx}_{dia_to_session[e]}"
                    for e in evidence if e in dia_to_session
                })
                if not gold_sessions:
                    continue
                qs.append({"question": qa["question"], "gold_keys": gold_sessions,
                           "scope_keys": list(conv_session_ids),
                           "gold_answer": str(qa.get("answer", "")),
                           "question_date": "",  # LoCoMo ships no per-question date
                           "question_type": LOCOMO_CATEGORY_NAMES.get(
                               qa.get("category"), f"category_{qa.get('category', '?')}")})
        return {"memories": mems, "queries": qs}

    raw = _cached(cache, _build)
    mems = [MemEntry(**m) for m in raw["memories"]]
    all_qs = [QueryEntry(**q) for q in raw["queries"]]
    rng = random.Random(seed)
    sampled = rng.sample(all_qs, min(n_queries, len(all_qs)))
    print(f"    {len(mems):,} sessions  ·  {len(sampled):,} queries (of {len(all_qs):,})")
    return BenchDataset("LoCoMo", mems, sampled)


def load_longmemeval(n_queries: int, cache_dir: Path = CACHE_DIR, seed: int = 42,
                      split: str = "oracle") -> BenchDataset:
    """
    LongMemEval (ICLR 2026) — 500 questions over freely-scalable synthetic
    chat histories, 5 core abilities (information extraction, multi-session
    reasoning, temporal reasoning, knowledge updates, abstention). This is
    the "oracle" split (haystack sessions are the ones actually relevant,
    not the full multi-hundred-session haystack) via a third-party
    HuggingFace mirror — cross-check against the canonical ICLR release
    before citing provenance in the paper (see LAUNCH_ROADMAP.md Phase 1
    risk notes).
    """
    if split not in ("oracle", "s"):
        raise ValueError(f"split must be 'oracle' or 's', got {split!r}")
    print(f"  Loading LongMemEval ({split})...")
    cache = cache_dir / (f"longmemeval.json" if split == "oracle" else "longmemeval_s.json")

    def _build():
        fname = "longmemeval_oracle.json" if split == "oracle" else "longmemeval_s_cleaned.json"
        url = ("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
               f"resolve/main/{fname}")
        r = requests.get(url, timeout=60, allow_redirects=True)
        r.raise_for_status()
        data = r.json()

        sess_map = {}  # session_id -> {"content": str, "turns": [...]}
        for item in data:
            ids = item.get("haystack_session_ids", [])
            sessions = item.get("haystack_sessions", [])
            dates = item.get("haystack_dates", [])
            for i, (sid, turns_raw) in enumerate(zip(ids, sessions)):
                if sid in sess_map:
                    continue
                # The session's own date, stamped onto every turn. Temporal
                # questions ("how many days ago...") are answerable only if
                # each memory carries WHEN it happened — a date sitting in
                # loader metadata that never reaches the retrieved text is
                # the same as no date at all.
                sdate = dates[i] if i < len(dates) else ""
                lines = []
                structured_turns = []
                for turn in turns_raw[:40]:  # was 20 — measured: gold evidence sat past it
                    role = turn.get("role", "?")
                    content = (turn.get("content", "") or "")[:800]  # was 300
                    stamped = f"[{sdate}] {content}" if sdate else content
                    lines.append(f"{role.capitalize()}: {stamped}")
                    structured_turns.append({"role": role, "content": stamped})
                header = f"Session dated {sdate}\n" if sdate else ""
                sess_map[sid] = {
                    "content": header + "\n".join(lines),
                    "turns": structured_turns,
                    "date": sdate,
                }

        mems = []
        for sid, payload in sess_map.items():
            title = f"Session {sid[:16]}"
            if payload.get("date"):
                title += f" ({payload['date']})"
            mems.append({
                "mid": sid, "title": title,
                "content": payload["content"], "search_text": payload["content"],
                "gold_key": sid, "turns": payload["turns"],
            })

        qs = []
        for item in data:
            ans_ids = item.get("answer_session_ids", [])
            if not ans_ids or not item.get("answer"):
                continue
            qs.append({"question": item["question"], "gold_keys": ans_ids,
                       "scope_keys": item.get("haystack_session_ids", []),
                       "gold_answer": str(item["answer"]),
                       "question_date": item.get("question_date", ""),
                       "question_type": item.get("question_type", "")})
        return {"memories": mems, "queries": qs}

    raw = _cached(cache, _build)
    mems = [MemEntry(**m) for m in raw["memories"]]
    all_qs = [QueryEntry(**q) for q in raw["queries"]]
    rng = random.Random(seed)
    sampled = rng.sample(all_qs, min(n_queries, len(all_qs)))
    avg_scope = sum(len(q.scope_keys) for q in sampled) / max(1, len(sampled))
    print(f"    {len(mems):,} sessions  ·  {len(sampled):,} queries (of {len(all_qs):,})"
          f"  ·  {avg_scope:.1f} sessions/question haystack")
    return BenchDataset("LongMemEval", mems, sampled)
