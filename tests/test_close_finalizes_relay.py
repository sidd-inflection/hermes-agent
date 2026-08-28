from unittest.mock import patch
from run_agent import AIAgent

def _agent():
    return AIAgent(model="dummy", base_url="http://localhost:1",
                   api_key="x", max_tokens=16, skip_memory=True,
                   skip_context_files=True, quiet_mode=True)

def test_close_finalizes_relay_session():
    agent = _agent()
    with patch("agent.relay_runtime.SESSION_COORDINATOR.finalize_conversation") as fin:
        agent.close()
    fin.assert_called_once()
    assert fin.call_args.kwargs["session_id"] == agent.session_id

def test_close_end_session_false_keeps_conversation_resumable():
    agent = _agent()
    with patch("agent.relay_runtime.SESSION_COORDINATOR.finalize_conversation") as fin:
        agent.close(end_session=False)
    fin.assert_not_called()
