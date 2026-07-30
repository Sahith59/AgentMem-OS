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


@dataclass
class QueryEntry:
    question: str
    gold_keys: list          # list[gold_key] — all session IDs that answer this


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


def load_locomo(n_queries: int, cache_dir: Path = CACHE_DIR, seed: int = 42) -> BenchDataset:
    """
    LoCoMo (Maharana et al., ACL 2024) — 10 long-term conversations, up to
    35 sessions each, 1,540 QA pairs across single-hop/multi-hop/temporal/
    open-domain categories. Category filter here keeps 1/2/3 (matches the
    upstream script's scope), matching the categories that have clean
    session-level evidence mapping.
    """
    print("  Loading LoCoMo...")
    cache = cache_dir / "locomo.json"

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
                session_content = "\n".join(lines)[:6000]
                session_id = f"c{conv_idx}_{sk}"
                date_val = conv_data.get(f"{sk}_date", "")
                title = f"Conv{conv_idx+1} {sk}" + (f" ({date_val})" if date_val else "")

                mems.append({
                    "mid": session_id, "title": title,
                    "content": session_content, "search_text": session_content,
                    "gold_key": session_id, "turns": structured_turns,
                })

            for qa in conv["qa"]:
                evidence = qa.get("evidence", [])
                if not evidence:
                    continue
                if qa.get("category", 99) not in (1, 2, 3):
                    continue
                gold_sessions = list({
                    f"c{conv_idx}_{dia_to_session[e]}"
                    for e in evidence if e in dia_to_session
                })
                if not gold_sessions:
                    continue
                qs.append({"question": qa["question"], "gold_keys": gold_sessions})
        return {"memories": mems, "queries": qs}

    raw = _cached(cache, _build)
    mems = [MemEntry(**m) for m in raw["memories"]]
    all_qs = [QueryEntry(**q) for q in raw["queries"]]
    rng = random.Random(seed)
    sampled = rng.sample(all_qs, min(n_queries, len(all_qs)))
    print(f"    {len(mems):,} sessions  ·  {len(sampled):,} queries (of {len(all_qs):,})")
    return BenchDataset("LoCoMo", mems, sampled)


def load_longmemeval(n_queries: int, cache_dir: Path = CACHE_DIR, seed: int = 42) -> BenchDataset:
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
    print("  Loading LongMemEval (oracle)...")
    cache = cache_dir / "longmemeval.json"

    def _build():
        url = ("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
               "resolve/main/longmemeval_oracle.json")
        r = requests.get(url, timeout=60, allow_redirects=True)
        r.raise_for_status()
        data = r.json()

        sess_map = {}  # session_id -> {"content": str, "turns": [...]}
        for item in data:
            ids = item.get("haystack_session_ids", [])
            sessions = item.get("haystack_sessions", [])
            for sid, turns_raw in zip(ids, sessions):
                if sid in sess_map:
                    continue
                lines = []
                structured_turns = []
                for turn in turns_raw[:20]:  # cap at 20 turns/session
                    role = turn.get("role", "?")
                    content = (turn.get("content", "") or "")[:300]
                    lines.append(f"{role.capitalize()}: {content}")
                    structured_turns.append({"role": role, "content": content})
                sess_map[sid] = {"content": "\n".join(lines), "turns": structured_turns}

        mems = []
        for sid, payload in sess_map.items():
            mems.append({
                "mid": sid, "title": f"Session {sid[:16]}",
                "content": payload["content"], "search_text": payload["content"],
                "gold_key": sid, "turns": payload["turns"],
            })

        qs = []
        for item in data:
            ans_ids = item.get("answer_session_ids", [])
            if not ans_ids:
                continue
            qs.append({"question": item["question"], "gold_keys": ans_ids})
        return {"memories": mems, "queries": qs}

    raw = _cached(cache, _build)
    mems = [MemEntry(**m) for m in raw["memories"]]
    all_qs = [QueryEntry(**q) for q in raw["queries"]]
    rng = random.Random(seed)
    sampled = rng.sample(all_qs, min(n_queries, len(all_qs)))
    print(f"    {len(mems):,} sessions  ·  {len(sampled):,} queries (of {len(all_qs):,})")
    return BenchDataset("LongMemEval", mems, sampled)
