"""Grid (PI-9115-adjacent) wants read-only skills (skill_view/skills_list)
without granting write access (skill_manage). Splits the old bundled
``skills`` toolset into a read-only ``skills`` and a new ``skills_write``,
with a ``skills_full`` alias that restores the old bundle so no existing
default-profile caller loses ``skill_manage``.
"""

from types import SimpleNamespace

from toolsets import resolve_toolset


def test_skills_readonly():
    names = resolve_toolset("skills")
    assert "skill_view" in names
    assert "skills_list" in names
    assert "skill_manage" not in names


def test_skills_write_exists():
    assert "skill_manage" in resolve_toolset("skills_write")


def test_skills_full_alias_matches_old_behavior():
    assert set(resolve_toolset("skills_full")) == set(resolve_toolset("skills")) | set(
        resolve_toolset("skills_write")
    )


def _make_prompt_agent(**overrides):
    """Minimal stub for agent/system_prompt.py::build_system_prompt_parts.

    Mirrors tests/agent/test_system_prompt.py::_make_agent; skip_context_files
    is True (rather than patching load_soul_md/build_context_files_prompt)
    since this test only cares about the stable-tier tool guidance block.
    """
    base = dict(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
        _emit_status=lambda *_args, **_kwargs: None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_skills_guidance_absent_without_skill_manage():
    """SKILLS_GUIDANCE ("patch it immediately with skill_manage(...)") must
    not be emitted when only the read-only skills tools are enabled — it
    references a tool the model does not have (phantom-tool reference)."""
    from agent.system_prompt import build_system_prompt_parts

    agent = _make_prompt_agent(valid_tool_names=["skill_view", "skills_list"])
    parts = build_system_prompt_parts(agent)
    assert "patch it immediately with skill_manage(action='patch')" not in parts["stable"]


def test_skills_guidance_present_with_skill_manage():
    """Sanity check: the guidance IS emitted when skill_manage is present, so
    the assertion above is discriminating rather than vacuous."""
    from agent.system_prompt import build_system_prompt_parts

    agent = _make_prompt_agent(
        valid_tool_names=["skill_view", "skills_list", "skill_manage"]
    )
    parts = build_system_prompt_parts(agent)
    assert "patch it immediately with skill_manage(action='patch')" in parts["stable"]
