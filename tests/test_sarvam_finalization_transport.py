import copy
import json

import pytest

from benchmarks.sarvam_native_controls import finalization_transport as repair
from benchmarks.sarvam_native_study.backend import UPDATE_TOOL


def history():
    return {
        "model": "sarvam-105b",
        "tools": [UPDATE_TOOL],
        "messages": [
            {"role": "system", "content": "Answer from actual receipts."},
            {"role": "user", "content": "Update the approved date."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "update_appointment",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": json.dumps(
                    {
                        "status": "confirmed",
                        "date": "2026-12-21",
                        "version": 4,
                    }
                ),
            },
        ],
    }


def test_native_history_is_preserved_and_only_format_differs():
    source = history()
    before = copy.deepcopy(source)
    a, b = [repair.request_for(source, v) for v in ("control", "schema")]
    assert a["messages"][:-1] == source["messages"]
    assert a["tools"] == [UPDATE_TOOL] and a["tool_choice"] == "none"
    assert a.pop("response_format") == {"type": "json_object"}
    assert b.pop("response_format")["type"] == "json_schema"
    assert a == b and source == before


@pytest.mark.parametrize(
    "defect", ["missing_definition", "unmatched", "missing_result", "duplicate"]
)
def test_malformed_native_history_is_rejected(defect):
    source = history()
    if defect == "missing_definition":
        source["tools"] = []
    elif defect == "unmatched":
        source["messages"][-1]["tool_call_id"] = "other-call"
    elif defect == "missing_result":
        source["messages"].pop()
    else:
        source["messages"].insert(3, copy.deepcopy(source["messages"][2]))
    with pytest.raises(ValueError):
        repair.request_for(source, "schema")


def test_unexpected_final_tool_call_never_becomes_an_answer():
    result = repair.classify(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": "2026-12-21",
                                "needs_clarification": False,
                                "action_status": "confirmed",
                            }
                        ),
                        "tool_calls": history()["messages"][2]["tool_calls"],
                    },
                }
            ]
        }
    )
    assert result["terminal"] == "refused_final_tool_request" and result["parsed"] is None
