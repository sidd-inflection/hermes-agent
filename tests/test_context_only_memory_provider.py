"""Context-only memory providers + instance kwargs (grid/v1 task A6).

Covers three deliverables:
  1. ``AIAgent(memory_provider=<instance>)`` wires a live provider instance
     without any config.yaml entry (used by Grid's ``GridMemoryProvider``).
  2. A *host-injected* provider whose ``get_tool_schemas()`` is empty
     (context-only, no tools to advertise) still gets its
     ``system_prompt_block()`` injected even when the ``memory`` toolset
     isn't enabled (#81014 gate narrowing).
  3. That narrowing is scoped to the instance path: a zero-schema provider
     wired the *config.yaml* way must still be suppressed by an operator's
     ``disabled_toolsets=["memory"]`` (#5544/#81014 must keep working for
     providers Grid didn't inject).
"""

from run_agent import AIAgent
from agent.memory_provider import MemoryProvider
from agent.system_prompt import build_system_prompt


class ContextOnly(MemoryProvider):
    @property
    def name(self) -> str:
        return "context_only"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []

    def system_prompt_block(self) -> str:
        return "## Known facts\nUser likes tests."


def test_instance_kwarg_wires_provider():
    agent = AIAgent(model="dummy", base_url="http://localhost:1", api_key="x",
                    max_tokens=16, skip_memory=True, skip_context_files=True,
                    quiet_mode=True, memory_provider=ContextOnly())
    assert agent._memory_manager is not None


def test_context_only_block_injected_without_memory_toolset():
    agent = AIAgent(model="dummy", base_url="http://localhost:1", api_key="x",
                    max_tokens=16, skip_memory=True, skip_context_files=True,
                    quiet_mode=True, enabled_toolsets=["web"],
                    memory_provider=ContextOnly())
    prompt = build_system_prompt(agent)
    assert "User likes tests." in prompt


def test_config_driven_zero_schema_provider_still_respects_disabled_toolset():
    """The #81014 gate narrowing is scoped to memory_provider= instances.

    A zero-schema provider wired the config.yaml way (MemoryManager()
    built and add_provider()'d directly, NOT via memory_provider=, so
    manager.host_injected stays False) must still be suppressed when an
    operator disables the memory toolset -- the exact case #5544/#81014
    exist to honor.
    """
    from agent.memory_manager import MemoryManager

    agent = AIAgent(model="dummy", base_url="http://localhost:1", api_key="x",
                    max_tokens=16, skip_memory=True, skip_context_files=True,
                    quiet_mode=True, disabled_toolsets=["memory"])
    agent._memory_manager = MemoryManager()
    agent._memory_manager.add_provider(ContextOnly())

    prompt = build_system_prompt(agent)
    assert "User likes tests." not in prompt
