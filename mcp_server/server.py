"""
AgentMem OS — MCP Server
========================
Exposes AgentMem OS memory capabilities as Model Context Protocol (MCP) tools
so any MCP-compatible client (Claude Desktop, Cursor, etc.) can use persistent
4-tier memory with a single config entry.

Tools:
  save_memory      — persist a user/assistant turn to the memory hierarchy
  recall_memory    — semantic + entity-aware retrieval across all tiers
  get_knowledge_graph — return entity relationship subgraph for a session
  list_sessions    — enumerate stored sessions with token/turn counts
  get_patterns     — return mined procedural patterns for a session
  summarize_session — force consolidation and return latest summaries

Run:
  agentmem-os-mcp              (stdio, for Claude Desktop)
  agentmem-os-mcp --transport sse --port 8765   (SSE, for web clients)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Optional

# Bootstrap path so imports resolve when run as a script outside the package
_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Load .env if present
_env = _HERE / ".env"
if _env.exists():
    from dotenv import load_dotenv
    load_dotenv(_env)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from agentmem_os.db.engine import get_session, init_db
from agentmem_os.db.models import Session as DBSession, Turn, Summary, ProceduralPattern
from agentmem_os.storage.store import ConversationStore
from agentmem_os.llm.context_assembler import ContextAssembler


# ─────────────────────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────────────────────

init_db()
_store: Optional[ConversationStore] = None


def _get_store() -> ConversationStore:
    global _store
    if _store is None:
        _store = ConversationStore()
    return _store


# ─────────────────────────────────────────────────────────────────────────────
# MCP Server Definition
# ─────────────────────────────────────────────────────────────────────────────

server = Server("agentmem-os")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="save_memory",
            description=(
                "Persist a conversation turn into AgentMem OS's 4-tier memory hierarchy. "
                "Triggers importance scoring, entity KG ingestion, and background consolidation. "
                "Use this after every user/assistant exchange to build persistent memory."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Unique identifier for the conversation session"
                    },
                    "role": {
                        "type": "string",
                        "enum": ["user", "assistant"],
                        "description": "Speaker role"
                    },
                    "content": {
                        "type": "string",
                        "description": "The message text to store"
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Optional agent namespace (for multi-agent federation)"
                    }
                },
                "required": ["session_id", "role", "content"]
            }
        ),
        types.Tool(
            name="recall_memory",
            description=(
                "Retrieve relevant memory context for a query from all 4 memory tiers: "
                "episodic (exact turns), semantic (DBSCAN-compressed summaries), "
                "entity KG (relationship graph), and procedural (behavioral patterns). "
                "Returns a structured XML context block ready to prepend to any LLM prompt."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session to retrieve memory from"
                    },
                    "query": {
                        "type": "string",
                        "description": "The current user query to retrieve relevant memory for"
                    },
                    "model_window": {
                        "type": "integer",
                        "description": "LLM context window size in tokens (default: 128000)",
                        "default": 128000
                    }
                },
                "required": ["session_id", "query"]
            }
        ),
        types.Tool(
            name="get_knowledge_graph",
            description=(
                "Return the entity relationship subgraph for a session. "
                "Shows entities extracted via spaCy NER, co-occurrence edges with weights, "
                "and BFS-reachable neighbors up to 2 hops from a seed entity. "
                "Useful for understanding what concepts the agent knows about."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session to query the knowledge graph for"
                    },
                    "entity": {
                        "type": "string",
                        "description": "Optional seed entity for BFS subgraph retrieval"
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "BFS depth from seed entity (default: 2)",
                        "default": 2
                    }
                },
                "required": ["session_id"]
            }
        ),
        types.Tool(
            name="list_sessions",
            description=(
                "List all stored sessions with their metadata: turn count, token usage, "
                "creation time, and parent branch info. Useful for session management "
                "and understanding what conversations are stored in memory."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Optional agent namespace to filter sessions"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum sessions to return (default: 20)",
                        "default": 20
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="get_patterns",
            description=(
                "Return mined procedural memory patterns for a session. "
                "Patterns are (trigger→action) pairs extracted from conversation history "
                "with confidence scores. Examples: bug_report→explain_solution (0.82), "
                "feature_request→provide_implementation (0.75). "
                "Use these to predict what kind of response is most useful."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session to retrieve patterns for"
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence threshold (0.0–1.0, default: 0.5)",
                        "default": 0.5
                    }
                },
                "required": ["session_id"]
            }
        ),
        types.Tool(
            name="summarize_session",
            description=(
                "Force DBSCAN sleep consolidation on a session and return the generated "
                "summaries. Clusters semantically similar turns, generates one LLM summary "
                "per cluster at abstraction levels L1 (episode), L2 (pattern), L3 (principle). "
                "Call this to compact a long session before context runs out."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session to consolidate"
                    }
                },
                "required": ["session_id"]
            }
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        if name == "save_memory":
            return await _save_memory(**arguments)
        elif name == "recall_memory":
            return await _recall_memory(**arguments)
        elif name == "get_knowledge_graph":
            return await _get_knowledge_graph(**arguments)
        elif name == "list_sessions":
            return await _list_sessions(**arguments)
        elif name == "get_patterns":
            return await _get_patterns(**arguments)
        elif name == "summarize_session":
            return await _summarize_session(**arguments)
        else:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"}),
            )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": f"Error in {name}: {e!r}"}),
        )]


# ─────────────────────────────────────────────────────────────────────────────
# Tool Implementations
# ─────────────────────────────────────────────────────────────────────────────

async def _save_memory(
    session_id: str,
    role: str,
    content: str,
    agent_id: Optional[str] = None,
) -> list[types.TextContent]:
    store = _get_store()
    store.get_or_create_session(session_id, name=session_id)
    store.save_turn(session_id, role, content, agent_id=agent_id)

    db = get_session()
    try:
        turn_count = db.query(Turn).filter(Turn.session_id == session_id).count()
    finally:
        db.close()

    result = {
        "status": "saved",
        "session_id": session_id,
        "role": role,
        "turn_count": turn_count,
        "tiers_updated": ["episodic (SQLite)", "working (Redis if available)", "KG (background)"]
    }
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _recall_memory(
    session_id: str,
    query: str,
    model_window: int = 128000,
) -> list[types.TextContent]:
    store = _get_store()

    # Ensure session exists
    db = get_session()
    try:
        sess = db.query(DBSession).filter(DBSession.session_id == session_id).first()
        if not sess:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Session '{session_id}' not found. Save some turns first."})
            )]
    finally:
        db.close()

    try:
        assembler = ContextAssembler(model_window=model_window)
        context_xml = assembler.assemble(session_id, query)
        token_estimate = len(context_xml) // 4

        result = {
            "session_id": session_id,
            "query": query,
            "token_estimate": token_estimate,
            "context": context_xml,
        }
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        # Fallback: return raw recent turns
        turns = store.get_history(session_id, last_n=10)
        result = {
            "session_id": session_id,
            "query": query,
            "fallback": True,
            "error": str(e),
            "recent_turns": turns,
        }
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _get_knowledge_graph(
    session_id: str,
    entity: Optional[str] = None,
    max_hops: int = 2,
) -> list[types.TextContent]:
    store = _get_store()
    kg = store._get_kg()

    if entity:
        subgraph_text = kg.get_relevant_subgraph(
            query=entity,
            agent_id=None,
            max_hops=max_hops,
        )
        result = {
            "session_id": session_id,
            "seed_entity": entity,
            "max_hops": max_hops,
            "subgraph": subgraph_text,
        }
    else:
        entity_count = kg.get_entity_count()
        # Return top-level graph summary via the KG's internal serializer
        result = {
            "session_id": session_id,
            "total_entities": entity_count,
            "note": "Provide an 'entity' parameter to retrieve a focused BFS subgraph",
            "graph_summary": kg._top_entities_summary(agent_id=None, top_k=15),
        }

    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _list_sessions(
    agent_id: Optional[str] = None,
    limit: int = 20,
) -> list[types.TextContent]:
    db = get_session()
    try:
        q = db.query(DBSession)
        if agent_id:
            q = q.filter(DBSession.agent_id == agent_id)
        sessions = q.order_by(DBSession.created_at.desc()).limit(limit).all()

        rows = []
        for s in sessions:
            turn_count = db.query(Turn).filter(Turn.session_id == s.session_id).count()
            rows.append({
                "session_id": s.session_id,
                "name": s.name,
                "model": s.model,
                "turn_count": turn_count,
                "total_tokens": s.total_tokens,
                "parent_session_id": s.parent_session_id,
                "created_at": str(s.created_at),
            })

        return [types.TextContent(type="text", text=json.dumps({"sessions": rows}, indent=2))]
    finally:
        db.close()


async def _get_patterns(
    session_id: str,
    min_confidence: float = 0.5,
) -> list[types.TextContent]:
    db = get_session()
    try:
        # ProceduralPattern uses source_sessions (comma-sep), not a single session_id FK
        patterns = (
            db.query(ProceduralPattern)
            .filter(
                ProceduralPattern.source_sessions.like(f"%{session_id}%"),
                ProceduralPattern.confidence >= min_confidence,
            )
            .order_by(ProceduralPattern.confidence.desc())
            .all()
        )

        if not patterns:
            # Try mining first, then re-query
            db.close()
            store = _get_store()
            proc = store._get_proc()
            proc.mine_patterns(session_id)
            db = get_session()
            patterns = (
                db.query(ProceduralPattern)
                .filter(
                    ProceduralPattern.source_sessions.like(f"%{session_id}%"),
                    ProceduralPattern.confidence >= min_confidence,
                )
                .order_by(ProceduralPattern.confidence.desc())
                .all()
            )

        rows = [
            {
                "trigger": p.trigger,
                "action": p.action,
                "confidence": round(p.confidence, 3),
                "support_count": p.support_count,
                "full_pattern": p.full_pattern[:200] if p.full_pattern else None,
            }
            for p in patterns
        ]

        return [types.TextContent(type="text", text=json.dumps({
            "session_id": session_id,
            "pattern_count": len(rows),
            "min_confidence": min_confidence,
            "patterns": rows,
        }, indent=2))]
    finally:
        db.close()


async def _summarize_session(session_id: str) -> list[types.TextContent]:
    store = _get_store()

    # Attempt consolidation — swallow LLM errors (e.g. ollama not running)
    consolidation_result = None
    consolidation_error = None
    try:
        engine = store._get_engine()
        consolidation_result = engine.consolidate(session_id, force=True)
    except Exception as exc:
        consolidation_error = str(exc)[:200]

    # Pull summaries from DB regardless of whether consolidation succeeded
    db = get_session()
    try:
        summaries = (
            db.query(Summary)
            .filter(Summary.session_id == session_id)
            .order_by(Summary.created_at.desc())
            .limit(10)
            .all()
        )
        rows = [
            {
                "id": s.id,
                "abstraction_level": s.abstraction_level,
                "content": s.content[:300] + "…" if len(s.content) > 300 else s.content,
                "turn_range": s.turn_range,
                "created_at": str(s.created_at),
            }
            for s in summaries
        ]

        payload = {
            "session_id": session_id,
            "consolidation_result": consolidation_result,
            "summary_count": len(rows),
            "summaries": rows,
        }
        if consolidation_error:
            payload["consolidation_warning"] = consolidation_error

        return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

async def _run_stdio():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    parser = argparse.ArgumentParser(description="AgentMem OS MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio for Claude Desktop)",
    )
    parser.add_argument("--port", type=int, default=8765, help="SSE server port (default: 8765)")
    parser.add_argument("--host", default="127.0.0.1", help="SSE server host (default: 127.0.0.1)")
    args = parser.parse_args()

    if args.transport == "stdio":
        import asyncio
        asyncio.run(_run_stdio())
    else:
        try:
            from mcp.server.sse import SseServerTransport
            from starlette.applications import Starlette
            from starlette.routing import Route, Mount
            import uvicorn

            sse = SseServerTransport("/messages/")

            async def handle_sse(request):
                async with sse.connect_sse(
                    request.scope, request.receive, request._send
                ) as streams:
                    await server.run(streams[0], streams[1], server.create_initialization_options())

            starlette_app = Starlette(
                routes=[
                    Route("/sse", endpoint=handle_sse),
                    Mount("/messages/", app=sse.handle_post_message),
                ]
            )
            print(f"AgentMem OS MCP server running on http://{args.host}:{args.port}/sse")
            uvicorn.run(starlette_app, host=args.host, port=args.port)
        except ImportError as e:
            print(f"SSE transport requires additional packages: {e}")
            print("Run: pip install uvicorn starlette")
            sys.exit(1)


if __name__ == "__main__":
    main()
