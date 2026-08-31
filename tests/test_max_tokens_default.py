"""Tests for resolve_max_tokens() — the derived-default max_tokens cap.

Background
----------
With max_tokens=None, Hermes used to leave max_tokens unresolved for the
request built for an unregistered/custom provider (the legacy chat_completions
path in build_api_kwargs), relying on the server's own default. A spike
against an OpenAI-compatible gateway found that gateway defaults an omitted
max_tokens to the model's full *context* length, then 400s on any nonempty
prompt because input + output > context ("requested N output tokens").

resolve_max_tokens() gives Hermes an explicit, safe output budget instead of
ever depending on the server's own (here, buggy) default: an explicit
request passes through unchanged; None resolves to the model's real output
cap when known, else a conservative fraction of the context window capped at
DEFAULT_MAX_OUTPUT_TOKENS — never the context length itself.
"""

from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    resolve_max_tokens,
)
from agent.conversation_loop import _raise_if_max_tokens_config_error


class FakeMeta:
    context_length = 65536
    max_output_tokens = None


def test_derived_default_never_equals_context_length():
    assert resolve_max_tokens(None, FakeMeta()) < 65536


def test_explicit_value_passes_through():
    assert resolve_max_tokens(2048, FakeMeta()) == 2048


def test_metadata_output_cap_wins():
    class M(FakeMeta):
        max_output_tokens = 4096
    assert resolve_max_tokens(None, M()) == 4096


def test_no_metadata_fallback_is_the_output_ceiling_not_ceiling_over_four():
    """I17: with no requested value and no context_length, the no-metadata
    path must land on DEFAULT_MAX_OUTPUT_TOKENS itself (a safe output
    budget), not on DEFAULT_MAX_OUTPUT_TOKENS substituted as a fake context
    length and then divided by CONTEXT_LENGTH_OUTPUT_DIVISOR — that
    misreads an OUTPUT ceiling as a context length and silently halves it
    again, yielding 2048 instead of the documented 8192."""
    class NoMeta:
        context_length = None
        max_output_tokens = None

    assert resolve_max_tokens(None, NoMeta()) == DEFAULT_MAX_OUTPUT_TOKENS


# ---------------------------------------------------------------------------
# _raise_if_max_tokens_config_error — the 400-classification half of the fix.
#
# Background: a 400 phrased "requested N output tokens" means max_tokens
# itself was too large for the call — the input fits. Before this fix,
# error_classifier.py's context_overflow heuristics (bare "max_tokens" /
# "context length" / "maximum context" substrings) classified it as context
# overflow, and conversation_loop.py's output-cap recovery path both retried
# with a smaller ephemeral max_tokens AND compressed conversation history —
# silently discarding history to "fix" a problem compression cannot fix. A
# spike observed exactly this: max_tokens=None derived to the model's full
# context length (the resolve_max_tokens bug above) produced a "Context too
# large — compressing" turn on an otherwise ordinary prompt.
# ---------------------------------------------------------------------------


class TestMaxTokensConfigErrorClassifier:
    def test_oversized_max_tokens_400_raises_config_error_not_compression(self):
        """agent.max_tokens explicitly set larger than the model's real
        output budget (here: no known cap, so the ctx//2 fallback governs)
        must raise immediately, carrying the server's message verbatim —
        never reach the compression path."""
        agent = SimpleNamespace(model="dory1105-64k", max_tokens=65536)
        msg = (
            "This model's maximum context length is 65536 tokens. However, "
            "you requested 65536 output tokens and your prompt contains "
            "1200 characters."
        )
        with pytest.raises(ValueError, match="requested 65536 output tokens"):
            _raise_if_max_tokens_config_error(msg, agent, context_length=65536)

    def test_genuine_context_overflow_is_not_raised(self):
        """A real input-too-long 400 (no 'requested N output tokens'
        phrasing) must NOT be intercepted — it keeps routing to the existing
        context_overflow / compression path."""
        agent = SimpleNamespace(model="dory1105-64k", max_tokens=8000)
        msg = "prompt is too long: 205000 tokens > 200000 maximum"
        _raise_if_max_tokens_config_error(msg, agent, context_length=200000)

    def test_borderline_output_cap_still_uses_shrink_and_retry(self):
        """max_tokens only mildly exceeds the request's budget (not clearly
        misconfigured) — the existing ephemeral shrink-and-retry recovery,
        which does work for this case, must still run."""
        agent = SimpleNamespace(model="dory1105-64k", max_tokens=50000)
        msg = (
            "This model's maximum context length is 200000 tokens. However, "
            "you requested 50000 output tokens and your prompt contains "
            "500000 characters."
        )
        _raise_if_max_tokens_config_error(msg, agent, context_length=200000)

    def test_qwen3_substring_model_name_does_not_suppress_the_raise(self):
        """Regression: agent.anthropic_adapter._ANTHROPIC_OUTPUT_LIMITS has a
        "qwen3" entry (DashScope's Anthropic-compatible-endpoint protocol
        ceiling, 65536) matched by bare substring against the model name.
        Grid's real model id contains that substring but is a plain
        chat_completions session (api_mode != "anthropic_messages"), not an
        Anthropic-compatible endpoint. An explicit max_tokens=65536 (a
        realistic misconfiguration — someone read the same protocol number)
        against a 65536-token context must still raise: half that context
        (32768) is the correct threshold, not the unrelated 65536 protocol
        ceiling coincidentally equal to what was requested.
        """
        agent = SimpleNamespace(
            model="qwen3_235b_fp8_system_prompt_dory1105_64k_noweb",
            max_tokens=65536,
            api_mode="chat_completions",
        )
        msg = (
            "This model's maximum context length is 65536 tokens. However, "
            "you requested 65536 output tokens and your prompt contains "
            "1200 characters."
        )
        with pytest.raises(ValueError, match="requested 65536 output tokens"):
            _raise_if_max_tokens_config_error(msg, agent, context_length=65536)
