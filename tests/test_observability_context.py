"""Tests for observability hooks: client_metadata kwarg + emitted_at on stream deltas.

Grid embeds Hermes in-process and already has its own tracing stack (Langfuse +
OTel) with dashboards that join on trace ids Grid derives itself. These tests
verify (1) a caller-supplied ``client_metadata`` dict reaches every hook/
middleware context a plugin can observe, so the bundled Langfuse plugin (and
Grid's own observability bridge) can join Grid's trace ids instead of minting
new ones, and (2) the stream-delta *observer* payload carries a real
``emitted_at`` timestamp so a time-to-first-token span can be closed without
wrapping the plain ``stream_delta_callback(text)`` callback.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def test_client_metadata_stored_and_default_empty():
    a = AIAgent(model="dummy", base_url="http://localhost:1", api_key="x",
                max_tokens=16, skip_memory=True, skip_context_files=True,
                quiet_mode=True, client_metadata={"trace_id": "t-1"})
    assert a.client_metadata == {"trace_id": "t-1"}
    b = AIAgent(model="dummy", base_url="http://localhost:1", api_key="x",
                max_tokens=16, skip_memory=True, skip_context_files=True,
                quiet_mode=True)
    assert b.client_metadata == {}


# ---------------------------------------------------------------------------
# Fixtures / helpers (mirrors tests/run_agent/test_run_agent.py conventions)
# ---------------------------------------------------------------------------


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_tool_call(name="web_search", arguments="{}", call_id="c1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(
        role="assistant", content=content, tool_calls=tool_calls,
        reasoning=None, reasoning_content=None, reasoning_details=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = None
    return resp


@pytest.fixture()
def agent():
    """Minimal AIAgent, mocked client, client_metadata set for hook assertions."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            client_metadata={"trace_id": "grid-t-1", "user_id": "u-1", "conversation_sid": "conv-1"},
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = False
        a.save_trajectories = False
        return a


def test_stream_delta_observer_payload_has_emitted_at(agent):
    """emitted_at is on the plugin observer payload, captured at emission time.

    The plain stream_delta_callback(text) signature is untouched — this only
    asserts on the on_stream_delta plugin hook payload.
    """
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    calls = []

    def on_stream_delta(**kwargs):
        calls.append(kwargs)

    plain_calls = []
    agent.stream_delta_callback = lambda text: plain_calls.append(text)

    with patch(
        "hermes_cli.plugins.iter_hook_callbacks",
        lambda name: (on_stream_delta,) if name == "on_stream_delta" else (),
    ):
        before = time.time()
        agent._fire_stream_delta("hello")
        after = time.time()

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not calls:
            time.sleep(0.01)
        shutdown_plugin_stream_hook_dispatcher()

    # Plain callback signature is unchanged: exactly one positional str arg.
    assert plain_calls == ["hello"]

    assert len(calls) == 1
    assert calls[0]["delta"] == "hello"
    assert "emitted_at" in calls[0]
    emitted_at = calls[0]["emitted_at"]
    assert isinstance(emitted_at, float)
    # Captured at actual emission time, not construction time.
    assert before <= emitted_at <= after
    assert abs(emitted_at - time.time()) < 5


def test_client_metadata_reaches_pre_api_request_hook_and_llm_middleware(agent, monkeypatch):
    agent.client.chat.completions.create.side_effect = [_mock_response(content="Done")]

    hook_calls = []

    def _record_hook(name, **kwargs):
        hook_calls.append((name, kwargs))
        return []

    middleware_calls = []

    def _record_middleware(kind, **kwargs):
        middleware_calls.append((kind, kwargs))
        return []

    monkeypatch.setattr("hermes_cli.middleware._has_middleware", lambda kind: True)
    monkeypatch.setattr("hermes_cli.middleware._invoke_middleware", _record_middleware)

    with (
        patch(
            "hermes_cli.lifecycle.has_hook",
            side_effect=lambda name: name in {"pre_api_request", "post_api_request"},
        ),
        patch("hermes_cli.lifecycle.invoke_hook", side_effect=_record_hook),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hi there")

    assert result["final_response"] == "Done"

    pre_request_calls = [kw for name, kw in hook_calls if name == "pre_api_request"]
    assert len(pre_request_calls) == 1
    assert pre_request_calls[0]["client_metadata"] == {
        "trace_id": "grid-t-1", "user_id": "u-1", "conversation_sid": "conv-1",
    }

    llm_request_calls = [kw for kind, kw in middleware_calls if kind == "llm_request"]
    assert len(llm_request_calls) == 1
    assert llm_request_calls[0]["client_metadata"] == {
        "trace_id": "grid-t-1", "user_id": "u-1", "conversation_sid": "conv-1",
    }


def test_client_metadata_reaches_pre_tool_call_hook(agent):
    tc = _mock_tool_call()
    resp1 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
    resp2 = _mock_response(content="Done searching", finish_reason="stop")
    agent.client.chat.completions.create.side_effect = [resp1, resp2]

    hook_calls = []

    def _record_hook(name, **kwargs):
        hook_calls.append((name, kwargs))
        return []

    with (
        patch("run_agent.handle_function_call", return_value="search result"),
        patch("hermes_cli.lifecycle.has_hook", return_value=False),
        patch("hermes_cli.lifecycle.invoke_hook", side_effect=_record_hook),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search something")

    assert result["final_response"] == "Done searching"
    pre_tool_calls = [kw for name, kw in hook_calls if name == "pre_tool_call"]
    assert len(pre_tool_calls) == 1
    assert pre_tool_calls[0]["client_metadata"] == {
        "trace_id": "grid-t-1", "user_id": "u-1", "conversation_sid": "conv-1",
    }


def test_client_metadata_is_copied_per_dispatch_not_shared(agent):
    """A hook that mutates the client_metadata dict it receives must not
    corrupt agent.client_metadata for later hook/middleware calls on the
    same agent -- each dispatch site must hand out its own copy.
    """
    tc1 = _mock_tool_call(call_id="c1")
    tc2 = _mock_tool_call(call_id="c2")
    resp1 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc1])
    resp2 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc2])
    resp3 = _mock_response(content="Done", finish_reason="stop")
    agent.client.chat.completions.create.side_effect = [resp1, resp2, resp3]

    original = dict(agent.client_metadata)
    hook_calls = []

    def _mutating_hook(name, **kwargs):
        hook_calls.append((name, dict(kwargs.get("client_metadata") or {})))
        cm = kwargs.get("client_metadata")
        if isinstance(cm, dict):
            cm["trace_id"] = "CORRUPTED-by-" + name  # simulate a careless plugin
        return []

    with (
        patch("run_agent.handle_function_call", return_value="search result"),
        patch(
            "hermes_cli.lifecycle.has_hook",
            side_effect=lambda name: name in {"pre_api_request", "pre_tool_call"},
        ),
        patch("hermes_cli.lifecycle.invoke_hook", side_effect=_mutating_hook),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search something")

    assert result["final_response"] == "Done"

    for hook_name in ("pre_api_request", "pre_tool_call"):
        seen = [cm for name, cm in hook_calls if name == hook_name]
        assert len(seen) >= 2, f"expected {hook_name} to fire at least twice"
        # Every invocation must have seen the pristine values -- a mutation
        # from an earlier invocation must never leak into a later one.
        for cm in seen:
            assert cm == original, f"{hook_name} saw a corrupted client_metadata: {cm}"

    # agent.client_metadata itself must be untouched by any hook mutation.
    assert agent.client_metadata == original
