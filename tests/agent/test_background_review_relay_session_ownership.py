"""The background-review fork shares the parent's live session_id (for
prompt-cache warmth — see agent/background_review.py's
``review_agent.session_id = agent.session_id``). ``AIAgent.close()``
finalizes the Relay session for ``self.session_id`` by default, so closing
the fork with the default would finalize the PARENT's still-active Relay
session, not just the fork's. Both teardown paths (normal completion and
the exception-path safety net) must pass ``end_session=False``.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import agent.background_review as bg  # noqa: E402


def _fake_parent(*, run_conversation_error: Exception | None = None) -> SimpleNamespace:
    """The minimal parent-agent surface _run_review_in_thread touches pre-fork."""
    return SimpleNamespace(
        provider="openai",
        model="gpt-5",
        client=MagicMock(),
        session_id="s1",
        platform="cli",
        request_overrides={},
        max_tokens=None,
        acp_command=None,
        acp_args=[],
        enabled_toolsets=None,
        disabled_toolsets=None,
        reasoning_config=None,
        _credential_pool=None,
        _current_main_runtime=lambda: {
            "api_key": "k", "base_url": "https://example.invalid", "api_mode": "chat_completions",
        },
        _emit_auxiliary_failure=lambda *_a, **_k: None,
        _safe_print=lambda *_a, **_k: None,
        background_review_callback=None,
    )


def _run(agent, *, review_agent_raises=False):
    """Run the worker with AIAgent patched; return the review_agent mock."""
    with (
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.AIAgent") as mock_aiagent,
        patch("tools.terminal_tool.set_approval_callback"),
    ):
        review_agent = mock_aiagent.return_value
        if review_agent_raises:
            review_agent.run_conversation.side_effect = RuntimeError("boom")
        bg._run_review_in_thread(
            agent, [{"role": "user", "content": "hi"}], "review please", None
        )
    return review_agent


def test_normal_completion_does_not_finalize_the_shared_parent_session():
    review_agent = _run(_fake_parent())
    review_agent.close.assert_called_once_with(end_session=False)


def test_exception_path_safety_net_does_not_finalize_the_shared_parent_session():
    review_agent = _run(_fake_parent(), review_agent_raises=True)
    review_agent.close.assert_called_once_with(end_session=False)
