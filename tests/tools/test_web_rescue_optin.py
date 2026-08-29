"""``web.keyless_rescue: false`` gives an operator a real query-egress
guarantee: with it off, NO failure of the configured backend — auth or
not — ever routes the query to a keyless third-party provider. That
blanket behavior already existed upstream (see
tests/tools/test_web_keyless_rescue.py::TestSearchRescue::test_no_rescue_when_disabled,
unmodified by this task).

What Task A9 actually adds: when the disabled path is taken AND the
backend's failure looks like a rejected credential (401/403/
'Unauthorized'), the returned failure is a specific, actionable message
naming the backend and the credential problem, instead of a bare
pass-through of the original error.
"""

import json
from unittest.mock import patch

import pytest

import tools.web_tools as web_tools
from plugins.web import keyless_mcp


class _AuthFailProvider:
    """Keyed provider double whose configured backend rejects credentials."""

    name = "tavily"
    display_name = "Tavily"

    def supports_search(self):
        return True

    def is_available(self):
        return True

    def search(self, query, limit=5):
        return {"success": False, "error": "HTTP 401: Unauthorized: missing or invalid API key."}


class _AuthRaisingProvider(_AuthFailProvider):
    def search(self, query, limit=5):
        raise RuntimeError("HTTP 403: Forbidden — API key revoked")


class _NonAuthFailProvider(_AuthFailProvider):
    def search(self, query, limit=5):
        return {"success": False, "error": "HTTP 500 upstream exploded"}


@pytest.fixture(autouse=True)
def _keyed_tavily_env(monkeypatch):
    """Simulate a keyed Tavily setup with the keyless tier available."""
    monkeypatch.setattr(
        "agent.web_search_provider.get_provider_env",
        lambda name: "tvly-real" if name == "TAVILY_API_KEY" else "",
    )
    monkeypatch.setattr(
        "agent.web_search_registry._keyless_tier_enabled", lambda: True
    )
    yield


def _dispatch(monkeypatch, provider):
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        "agent.web_search_registry.get_provider", lambda name: provider
    )
    return json.loads(web_tools.web_search_tool("q", limit=2))


def test_rescue_on_by_default():
    assert web_tools._keyless_rescue_enabled() is True


def test_rescue_off_auth_error_gets_credential_message(monkeypatch):
    """Flag off + auth error (returned-failure path): no keyless contact
    (the pre-existing blanket guarantee) AND the message is now the
    specific, actionable credential-rejection one."""
    monkeypatch.setattr(
        web_tools, "_load_web_config",
        lambda: {"backend": "tavily", "keyless_rescue": False},
    )
    with patch.object(keyless_mcp, "search_with_failover") as ring:
        out = _dispatch(monkeypatch, _AuthFailProvider())
    assert out["success"] is False
    assert "credentials" in out["error"]
    assert out["provider"] == "tavily"
    ring.assert_not_called()


def test_rescue_off_raised_auth_error_gets_credential_message(monkeypatch):
    """Same, but the backend raises instead of returning a failure dict."""
    monkeypatch.setattr(
        web_tools, "_load_web_config",
        lambda: {"backend": "tavily", "keyless_rescue": False},
    )
    with patch.object(keyless_mcp, "search_with_failover") as ring:
        out = _dispatch(monkeypatch, _AuthRaisingProvider())
    assert out["success"] is False
    assert "credentials" in out["error"]
    ring.assert_not_called()


def test_rescue_off_non_auth_error_still_no_keyless_call(monkeypatch):
    """Flag off + a non-auth error: still no keyless contact — the flag is
    a blanket egress guarantee, not an auth-only one. The message is just
    the original error, not the credential-specific wording (nothing here
    looked like a rejected credential)."""
    monkeypatch.setattr(
        web_tools, "_load_web_config",
        lambda: {"backend": "tavily", "keyless_rescue": False},
    )
    with patch.object(keyless_mcp, "search_with_failover") as ring:
        out = _dispatch(monkeypatch, _NonAuthFailProvider())
    assert out["success"] is False
    assert "credentials" not in out["error"]
    ring.assert_not_called()


def test_rescue_on_auth_error_still_rescues(monkeypatch):
    """Default config (flag on): auth failures behave exactly as before —
    rescued, same as any other backend failure. The credential message
    only appears on the disabled path."""
    with patch.object(
        keyless_mcp, "search_with_failover",
        return_value={"success": True, "data": {"web": [{"url": "https://exa.example"}]}},
    ) as ring:
        out = _dispatch(monkeypatch, _AuthFailProvider())
    assert out["success"] is True
    ring.assert_called_once()
