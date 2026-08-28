from run_agent import AIAgent
from agent.system_prompt import build_system_prompt


def _agent(**kw):
    return AIAgent(model="dummy", base_url="http://localhost:1", api_key="x",
                   max_tokens=16, skip_memory=True, skip_context_files=True,
                   quiet_mode=True, **kw)


def test_suppress_host_context_removes_host_lines():
    prompt = build_system_prompt(_agent(suppress_host_context=True))
    for needle in ("Host:", "User home directory:",
                   "Current working directory:", "Active Hermes profile:"):
        assert needle not in prompt


def test_default_keeps_host_lines():
    prompt = build_system_prompt(_agent())
    assert "Active Hermes profile:" in prompt


def test_product_help_guidance_config_off(monkeypatch, tmp_path):
    # config gate: agent.product_help_guidance=false drops the Nous banner
    agent = _agent()
    agent._product_help_guidance = False   # set by agent_init from config
    prompt = build_system_prompt(agent)
    assert "Nous Research" not in prompt
