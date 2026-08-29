"""Context-only memory providers + instance kwargs (grid/v1 task A6).

Covers two deliverables:
  1. ``AIAgent(memory_provider=<instance>)`` wires a live provider instance
     without any config.yaml entry (used by Grid's ``GridMemoryProvider``).
  2. A provider whose ``get_tool_schemas()`` is empty (context-only, no
     tools to advertise) still gets its ``system_prompt_block()`` injected
     even when the ``memory`` toolset isn't enabled (#81014 gate narrowing).
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
