"""
Tests for AgentMem OS MCP Server

Validates all 6 tool handlers work correctly without requiring
a live MCP transport (tests the async handler functions directly).
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

# Bootstrap import path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture(scope="module")
def session_id():
    return "mcp-test-session-001"


@pytest.fixture(scope="module", autouse=True)
def setup_session(session_id):
    """Pre-populate the test session with enough turns for pattern mining."""
    from mcp_server.server import handle_call_tool

    async def _seed():
        for i in range(6):
            await handle_call_tool("save_memory", {
                "session_id": session_id,
                "role": "user",
                "content": f"I found a bug in module {i}: the cache is returning stale data.",
            })
            await handle_call_tool("save_memory", {
                "session_id": session_id,
                "role": "assistant",
                "content": f"To fix bug {i}: clear the Redis cache key and force a re-fetch from SQLite.",
            })

    asyncio.run(_seed())


# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_sessions_returns_data():
    from mcp_server.server import handle_call_tool

    result = await handle_call_tool("list_sessions", {})
    assert len(result) == 1
    data = json.loads(result[0].text)
    assert "sessions" in data
    assert isinstance(data["sessions"], list)


@pytest.mark.asyncio
async def test_list_sessions_contains_test_session(session_id):
    from mcp_server.server import handle_call_tool

    result = await handle_call_tool("list_sessions", {})
    sessions = json.loads(result[0].text)["sessions"]
    ids = [s["session_id"] for s in sessions]
    assert session_id in ids


@pytest.mark.asyncio
async def test_save_memory_returns_saved(session_id):
    from mcp_server.server import handle_call_tool

    result = await handle_call_tool("save_memory", {
        "session_id": session_id,
        "role": "user",
        "content": "What is the token budget allocation for semantic memory?",
    })
    data = json.loads(result[0].text)
    assert data["status"] == "saved"
    assert data["session_id"] == session_id
    assert data["turn_count"] >= 1


@pytest.mark.asyncio
async def test_recall_memory_returns_context(session_id):
    from mcp_server.server import handle_call_tool

    result = await handle_call_tool("recall_memory", {
        "session_id": session_id,
        "query": "cache bug stale data",
        "model_window": 128000,
    })
    data = json.loads(result[0].text)
    assert "context" in data or "recent_turns" in data
    # token_estimate > 0 when assembler produces output
    if not data.get("fallback"):
        assert data.get("token_estimate", 0) > 0


@pytest.mark.asyncio
async def test_get_knowledge_graph_returns_dict(session_id):
    from mcp_server.server import handle_call_tool

    result = await handle_call_tool("get_knowledge_graph", {"session_id": session_id})
    data = json.loads(result[0].text)
    assert "session_id" in data
    assert "total_entities" in data or "subgraph" in data


@pytest.mark.asyncio
async def test_get_patterns_returns_valid_schema(session_id):
    from mcp_server.server import handle_call_tool

    result = await handle_call_tool("get_patterns", {
        "session_id": session_id,
        "min_confidence": 0.0,
    })
    data = json.loads(result[0].text)
    assert "pattern_count" in data
    assert "patterns" in data
    assert isinstance(data["patterns"], list)
    # With 6 bug-report turns, we expect at least 1 mined pattern
    assert data["pattern_count"] >= 1
    if data["patterns"]:
        p = data["patterns"][0]
        assert "trigger" in p
        assert "action" in p
        assert "confidence" in p
        assert 0.0 <= p["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_tool_error_returns_text_not_exception():
    from mcp_server.server import handle_call_tool

    # Bad session should return a JSON error, not raise
    result = await handle_call_tool("recall_memory", {
        "session_id": "session-that-does-not-exist-xyz",
        "query": "anything",
    })
    assert len(result) == 1
    text = result[0].text
    # Must be parseable (error dict or fallback JSON)
    data = json.loads(text)
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_text():
    from mcp_server.server import handle_call_tool

    result = await handle_call_tool("nonexistent_tool", {})
    data = json.loads(result[0].text)
    assert "error" in data
    assert "Unknown tool" in data["error"]
