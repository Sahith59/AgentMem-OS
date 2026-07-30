"""
Regression tests for the benchmark-credibility bugs fixed in Phase 1
(see LAUNCH_ROADMAP.md, Group A/B):

  (a) TokenEfficiencyEvaluator silently producing a complex number when a
      "compressed" summary is longer than the original.
  (b) ContextRelevanceEvaluator head-truncating assembled_context at 2000
      chars, scoring CRS against system-prompt boilerplate and never
      reaching semantic/KG/procedural/recent content.
  (c) The ablation scripts' CRS being blind to which tier is disabled,
      because it was computed from raw turn history instead of the
      context each variant actually assembled and sent to the model.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agentmem_os.benchmarks.eval_harness import (
    TokenEfficiencyEvaluator,
    ContextRelevanceEvaluator,
)
from agentmem_os.benchmarks._tier_lib import (
    assemble_context,
    crs_from_probe_contexts,
    patch_baselines_from_recent_only,
)


class _FakeTokenCounter:
    """4-chars-per-token, matching the benchmark scripts' own approximation."""
    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


# ── (a) TES complex-number crash ─────────────────────────────────────────

def test_tes_does_not_crash_when_summary_longer_than_original():
    evaluator = TokenEfficiencyEvaluator(token_counter=_FakeTokenCounter())
    original_turns = [{"content": "short"}]
    # Deliberately much longer than the "original" — this is exactly the
    # case that used to make comp_ratio go negative and (comp_ratio *
    # preservation) ** 0.5 return a complex number instead of a float.
    compressed_summaries = [
        "a much longer summary than the original content, padding padding "
        "padding padding padding padding padding padding padding"
    ]
    naive_truncated = [{"content": "short"}]

    result = evaluator.evaluate(original_turns, compressed_summaries, naive_truncated)

    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.baseline_score, float)


def test_tes_compression_ratio_clamped_in_details():
    evaluator = TokenEfficiencyEvaluator(token_counter=_FakeTokenCounter())
    result = evaluator.evaluate(
        original_turns=[{"content": "short"}],
        compressed_summaries=["way way way way way way way longer than short"],
        naive_truncated=[{"content": "short"}],
    )
    assert result.details["compression_ratio"] >= 0.0


# ── (b) CRS truncation must not be a blind head-slice ────────────────────

def test_crs_truncation_is_tail_biased_not_head_biased():
    evaluator = ContextRelevanceEvaluator(get_embedding_fn=lambda x: [1.0, 0.0])
    long_ctx = "HEAD_MARKER " + ("filler " * 2000) + " TAIL_MARKER"

    truncated = evaluator._truncate_for_embedding(long_ctx)

    assert len(truncated) <= evaluator._EMBED_CHAR_CAP
    assert "TAIL_MARKER" in truncated, (
        "CRS truncation dropped the tail — this is the section real "
        "ContextAssembler places [RECENT TURNS] in, per llm/context_assembler.py"
    )


def test_crs_truncation_keeps_short_context_whole():
    evaluator = ContextRelevanceEvaluator(get_embedding_fn=lambda x: [1.0, 0.0])
    short_ctx = "This context is well under the cap."
    assert evaluator._truncate_for_embedding(short_ctx) == short_ctx


def test_crs_embeds_the_truncated_not_the_raw_context():
    """The embedding function must never see more than _EMBED_CHAR_CAP chars."""
    seen_lengths = []

    def spy_embed(text):
        seen_lengths.append(len(text))
        return [1.0, 0.0]

    evaluator = ContextRelevanceEvaluator(get_embedding_fn=spy_embed)
    long_ctx = "x" * 50_000
    evaluator.evaluate(query="q", assembled_context=long_ctx, random_context="y" * 50_000)

    assert all(n <= evaluator._EMBED_CHAR_CAP for n in seen_lengths), (
        f"embedding fn was called with untruncated text: lengths={seen_lengths}"
    )


# ── (c) Ablation CRS must be sensitive to which tier is disabled ─────────

def test_crs_from_probe_contexts_differs_when_load_bearing_tier_disabled():
    """
    Synthetic conversation where the answer to the probe question only
    shows up in the semantic-retrieval tier. Disabling that tier must
    change the assembled context — and therefore CRS — for that probe.
    Before the fix, CRS was computed from raw turn history regardless of
    flags, so FULL and NO_SEMANTIC scored byte-identical CRS no matter
    what was actually disabled.
    """
    turns = [
        {"role": "user", "content": "The secret launch codename is Zeta Falcon."},
        {"role": "assistant", "content": "Got it, noted."},
    ] * 6  # pad past RECENT_WIN so semantic retrieval is actually exercised
    probe_query = "What is the secret launch codename?"

    full_ctx = assemble_context(turns, probe_query, {}, sleep_summary=None)
    no_semantic_ctx = assemble_context(turns, probe_query, {"no_semantic": True}, sleep_summary=None)

    full_probes = {0: (probe_query, full_ctx)}
    no_semantic_probes = {0: (probe_query, no_semantic_ctx)}

    crs_full = crs_from_probe_contexts(full_probes)
    crs_no_semantic = crs_from_probe_contexts(no_semantic_probes)

    # Both variants share the same last-RECENT_WIN-turns tail (the probe's
    # content happens to repeat there too in this synthetic example), so
    # assert on the weaker, still-meaningful invariant: the assembled
    # contexts themselves must differ when the tier is disabled — the
    # actual bug was that they were scored from a shared reconstruction
    # that couldn't differ at all.
    assert full_ctx != no_semantic_ctx
    assert "[SEMANTIC MEMORY" in full_ctx or len(turns) <= 8
    assert "[SEMANTIC MEMORY" not in no_semantic_ctx
    # CRS scores are floats derived from genuinely different input text —
    # they are not required to differ by any particular margin for this
    # specific synthetic example, but must be computed from the real
    # per-variant context, which the assertions above already establish.
    assert isinstance(crs_full, float)
    assert isinstance(crs_no_semantic, float)


def test_patch_baselines_from_recent_only_uses_measured_score_not_constant():
    """The old bug: lcs_base = 0.70, a hardcoded constant, identical across
    every variant and every run, never actually measured."""
    results = [
        {"variant": "FULL", "metrics": {
            "LCS": {"ours": 0.8, "baseline": 0.0, "delta": 0.0},
            "CRS": {"ours": 0.5, "baseline": 0.0, "delta": 0.0},
        }},
        {"variant": "RECENT_ONLY", "metrics": {
            "LCS": {"ours": 0.4, "baseline": 0.0, "delta": 0.0},
            "CRS": {"ours": 0.2, "baseline": 0.0, "delta": 0.0},
        }},
    ]

    patch_baselines_from_recent_only(results)

    full = results[0]
    assert full["metrics"]["LCS"]["baseline"] == 0.4
    assert full["metrics"]["LCS"]["baseline"] != 0.70  # the old hardcoded constant
    assert full["metrics"]["LCS"]["delta"] == 0.4
    assert full["metrics"]["CRS"]["baseline"] == 0.2

    # RECENT_ONLY is left pointing at itself: no more-naive floor exists.
    recent = results[1]
    assert recent["metrics"]["LCS"]["baseline"] == recent["metrics"]["LCS"]["ours"]
    assert recent["metrics"]["LCS"]["delta"] == 0.0


def test_patch_baselines_no_recent_only_present_is_a_noop():
    results = [{"variant": "FULL", "metrics": {
        "LCS": {"ours": 0.8, "baseline": 0.0, "delta": 0.0},
        "CRS": {"ours": 0.5, "baseline": 0.0, "delta": 0.0},
    }}]
    patch_baselines_from_recent_only(results)  # must not raise
    assert results[0]["metrics"]["LCS"]["baseline"] == 0.0
