"""Tests for Task A8 — env-overridable budget defaults.

Grid embeds Hermes in-process and needs two upstream defaults tamed:
``max_iterations`` defaults to ``sys.maxsize`` (a runaway agent loops until
something else kills the HTTP request), and ``MINIMUM_CONTEXT_LENGTH`` is a
frozen 64_000 constant that happens to equal Grid's model's nominal window —
but that model's *usable* context is ~44k once the server-side chat-template
preamble is accounted for, so the real floor needs to be lower.

Both overrides are opt-in and default-off: with neither env var set, behavior
is byte-identical to upstream (verified by the "no env" assertions below).

Both env reads happen at call time (construction for ``max_iterations``,
each call for ``minimum_context_length()``), not at module-import time, so
these tests patch ``os.environ`` directly rather than reloading ``run_agent``
or spawning a subprocess — the reload-based approach in the task brief guards
against a module-scope env read, which this implementation deliberately
avoids (see PHASE-A-NOTES HAZARD re: reload fighting module state).
"""
import os
import sys
from unittest.mock import patch

from run_agent import AIAgent


def _make_agent(**kwargs):
    return AIAgent(model="dummy", base_url="http://localhost:1", api_key="x",
                    max_tokens=16, skip_memory=True, skip_context_files=True,
                    quiet_mode=True, **kwargs)


def test_max_iterations_env_default():
    with patch.dict(os.environ, {"HERMES_MAX_ITERATIONS_DEFAULT": "12"}):
        agent = _make_agent()
    assert agent.max_iterations == 12


def test_max_iterations_default_unchanged_without_env():
    """Non-negotiable: with the env var unset, upstream behavior (unlimited
    iterations) is untouched."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("HERMES_MAX_ITERATIONS_DEFAULT", None)
        agent = _make_agent()
    assert agent.max_iterations == sys.maxsize


def test_max_iterations_explicit_kwarg_wins_over_env():
    with patch.dict(os.environ, {"HERMES_MAX_ITERATIONS_DEFAULT": "12"}):
        agent = _make_agent(max_iterations=5)
    assert agent.max_iterations == 5


def test_max_iterations_env_non_numeric_falls_back_to_unlimited():
    """An embedder typo must not crash agent construction or silently
    produce max_iterations=0 (an agent that does nothing)."""
    with patch.dict(os.environ, {"HERMES_MAX_ITERATIONS_DEFAULT": "not-a-number"}):
        agent = _make_agent()
    assert agent.max_iterations == sys.maxsize


def test_max_iterations_env_zero_falls_back_to_unlimited():
    with patch.dict(os.environ, {"HERMES_MAX_ITERATIONS_DEFAULT": "0"}):
        agent = _make_agent()
    assert agent.max_iterations == sys.maxsize


def test_max_iterations_env_negative_falls_back_to_unlimited():
    with patch.dict(os.environ, {"HERMES_MAX_ITERATIONS_DEFAULT": "-3"}):
        agent = _make_agent()
    assert agent.max_iterations == sys.maxsize


def test_minimum_context_length_env_override():
    from agent.model_metadata import minimum_context_length
    with patch.dict(os.environ, {"HERMES_MINIMUM_CONTEXT_LENGTH": "44000"}):
        assert minimum_context_length() == 44000
    assert minimum_context_length() == 64000


def test_minimum_context_length_env_non_numeric_falls_back():
    from agent.model_metadata import minimum_context_length
    with patch.dict(os.environ, {"HERMES_MINIMUM_CONTEXT_LENGTH": "not-a-number"}):
        assert minimum_context_length() == 64000


def test_minimum_context_length_env_non_positive_falls_back():
    from agent.model_metadata import minimum_context_length
    with patch.dict(os.environ, {"HERMES_MINIMUM_CONTEXT_LENGTH": "0"}):
        assert minimum_context_length() == 64000
    with patch.dict(os.environ, {"HERMES_MINIMUM_CONTEXT_LENGTH": "-1"}):
        assert minimum_context_length() == 64000


def test_context_switch_guard_uses_override():
    """Migrated caller: hermes_cli/context_switch_guard.py must read the
    live override, not the frozen module constant, or Grid's lowered floor
    is silently ignored for context-switch-threshold decisions."""
    from hermes_cli.context_switch_guard import _threshold_tokens
    with patch.dict(os.environ, {"HERMES_MINIMUM_CONTEXT_LENGTH": "1000"}):
        # threshold_percent small enough that the floor, not the percentage,
        # determines the result.
        assert _threshold_tokens(2000, threshold_percent=0.1) == 1000
