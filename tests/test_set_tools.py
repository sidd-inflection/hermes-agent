"""Tests for AIAgent.set_tools() — request-scoped tool rebinding (Task A7).

Grid builds a fresh tool surface per HTTP request (persona exclusions, memory
gates, feature flags) and must rebind an already-constructed AIAgent's tools.
Reassigning ``agent.tools`` alone advertises new schemas to the model but the
model's resulting tool call is then silently discarded, because
``valid_tool_names`` is frozen at init and the dispatch check in
conversation_loop.py only accepts calls whose name is in that set.
``set_tools`` must set both atomically.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from run_agent import AIAgent
from tools.todo_tool import TODO_SCHEMA

DEF = {"type": "function", "function": {"name": "checklist_create",
       "parameters": {"type": "object", "properties": {}}}}

TODO_DEF = {"type": "function", "function": TODO_SCHEMA}


def test_set_tools_rebinds_names_too():
    agent = AIAgent(model="dummy", base_url="http://localhost:1", api_key="x",
                    max_tokens=16, skip_memory=True, skip_context_files=True,
                    quiet_mode=True)
    agent.set_tools([DEF])
    assert agent.tools == [DEF]
    assert agent.valid_tool_names == {"checklist_create"}


def test_set_tools_empty_list_clears_both():
    agent = AIAgent(model="dummy", base_url="http://localhost:1", api_key="x",
                    max_tokens=16, skip_memory=True, skip_context_files=True,
                    quiet_mode=True)
    assert agent.tools  # sanity: agent started with a non-empty default toolset
    agent.set_tools([])
    assert agent.tools == []
    assert agent.valid_tool_names == set()


def test_set_tools_skips_malformed_schema_without_raising():
    agent = AIAgent(model="dummy", base_url="http://localhost:1", api_key="x",
                    max_tokens=16, skip_memory=True, skip_context_files=True,
                    quiet_mode=True)
    malformed = [
        {"type": "function", "function": {"parameters": {}}},  # missing name
        {"type": "function"},  # missing function entirely
        "not-a-dict",  # not even a dict
        DEF,
    ]
    agent.set_tools(malformed)
    # tools is a verbatim copy of what was passed (including the junk entries) —
    # valid_tool_names is the derived, filtered index used for dispatch.
    assert agent.tools == malformed
    assert agent.valid_tool_names == {"checklist_create"}


def test_set_tools_publishes_under_shared_lock_and_bumps_generation():
    """set_tools must join the same atomic-publish protocol
    refresh_agent_mcp_tools uses (tools/mcp_tool.py), not invent its own —
    same lock instance, and _tool_snapshot_generation advanced to at least
    the current registry generation so a concurrent refresh computed from
    an older generation can't win a race against this publish."""
    import tools.mcp_tool as mcp_tool
    from tools.registry import registry as _registry

    agent = AIAgent(model="dummy", base_url="http://localhost:1", api_key="x",
                    max_tokens=16, skip_memory=True, skip_context_files=True,
                    quiet_mode=True)

    calls = []
    real_lock = mcp_tool._agent_tools_lock

    class _TrackingLock:
        def __enter__(self):
            calls.append("enter")
            return real_lock.__enter__()

        def __exit__(self, *a):
            calls.append("exit")
            return real_lock.__exit__(*a)

    tracking = _TrackingLock()
    orig = mcp_tool._agent_tools_lock
    mcp_tool._agent_tools_lock = tracking
    try:
        agent.set_tools([DEF])
    finally:
        mcp_tool._agent_tools_lock = orig

    assert calls == ["enter", "exit"], "set_tools did not publish under tools.mcp_tool._agent_tools_lock"
    assert agent._tool_snapshot_generation >= _registry._generation


# ── End-to-end proof: a newly-bound tool's call actually dispatches ──
#
# The trap this method exists to fix is silent: the model calls the tool and
# nothing happens, because the dispatch check at conversation_loop.py rejects
# any name not in valid_tool_names, which was frozen at init. Asserting the
# two attributes (above) is necessary but not sufficient — it doesn't prove
# the dispatch path itself accepts the call. This drives a real
# ``run_conversation`` turn against an in-process mock provider and confirms
# the "todo" tool — bound via set_tools() onto an agent built with NO
# tools — actually executes and returns real output, not a rejection.


class _MockHandler(BaseHTTPRequestHandler):
    captured_requests: list = []
    response_queue: list = []

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode())
        type(self).captured_requests.append(req)
        is_stream = req.get("stream") is True
        resp = type(self).response_queue.pop(0) if type(self).response_queue else _text_resp("DONE")
        msg = resp["choices"][0]["message"]
        if is_stream:
            content = msg.get("content") or ""
            tcs = msg.get("tool_calls")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [{"id": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}]
            if content:
                chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
            if tcs:
                for ti, tc in enumerate(tcs):
                    chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"tool_calls": [{
                        "index": ti, "id": tc["id"], "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}]}, "finish_reason": None}]})
            chunks.append({"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tcs else "stop"}]})
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a, **kw):  # silence default stderr logging
        pass


def _tc_resp(name: str, args: str) -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": name, "arguments": args}}]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


def _text_resp(text: str) -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


@pytest.fixture()
def agent_env():
    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    test_home = tempfile.mkdtemp(prefix="hermes_set_tools_")
    os.makedirs(os.path.join(test_home, ".hermes"))
    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = os.path.join(test_home, ".hermes")

    for mod in list(sys.modules):
        if mod == "run_agent" or mod.startswith("agent.") or mod.startswith("tools.") or mod.startswith("hermes_"):
            del sys.modules[mod]
    from run_agent import AIAgent as _AIAgent

    # enabled_toolsets=[] mirrors Grid's per-request agent: constructed with
    # NO tools, so any acceptance of the "todo" call below can only come from
    # set_tools(), never from the init-time toolset.
    agent = _AIAgent(
        api_key="test-key", base_url=f"http://127.0.0.1:{port}/v1",
        provider="openai-compat", model="test-model",
        max_iterations=10, enabled_toolsets=[],
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, platform="cli",
    )
    assert agent.tools == []
    assert agent.valid_tool_names == set()

    try:
        yield agent, _MockHandler
    finally:
        srv.shutdown()
        shutil.rmtree(test_home, ignore_errors=True)
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home


def test_set_tools_newly_bound_tool_actually_dispatches(agent_env):
    agent, handler = agent_env

    agent.set_tools([TODO_DEF])
    assert agent.valid_tool_names == {"todo"}

    args = json.dumps({"todos": [
        {"id": "1", "content": "grid-set-tools-e2e-proof", "status": "pending"}
    ]})
    handler.response_queue.append(_tc_resp("todo", args))
    handler.response_queue.append(_text_resp("done"))

    result = agent.run_conversation("track work", conversation_history=[], task_id="t")

    tool_results = [
        m.get("content", "") for m in result["messages"]
        if isinstance(m, dict) and m.get("role") == "tool"
    ]
    assert tool_results, "expected a tool result message for the dispatched call"
    joined = " ".join(tool_results)
    # Rejection path (valid_tool_names miss) never reaches the handler and
    # never echoes back the arguments we sent — it emits an "unknown tool"
    # style error instead. Seeing our content prove the call was actually
    # executed, not merely schema-accepted.
    assert "grid-set-tools-e2e-proof" in joined
    assert "unknown tool" not in joined.lower()
    assert "invalid tool" not in joined.lower()
