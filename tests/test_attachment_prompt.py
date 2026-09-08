"""attachments 参数：prompt / Agent.prompt / steer / follow_up / next_turn 接受附件。"""

import base64

import pytest
import senza

pytestmark = pytest.mark.skipif(
    not hasattr(senza, "Agent"),
    reason="requires the test-utils feature (senza.Agent / MockLlmClient)",
)

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _agent():
    return senza.Agent(responses=[{"type": "text", "text": "ok"}])


def test_agent_prompt_accepts_attachments():
    agent = _agent()
    out = agent.prompt("what is this", attachments=[senza.image_base64(PNG_1X1)])
    assert out == "ok"
    assert agent.phase() == "idle"
    assert agent.error_message is None


def test_agent_prompt_attachments_in_session():
    """附件块应进入会话历史（run_with_initial 路径写入 state.messages）。"""
    agent = _agent()
    agent.prompt("describe", attachments=[senza.image_base64(PNG_1X1)])
    assert agent.message_count() >= 2  # user + assistant


def test_agent_prompt_empty_attachments_ok():
    agent = _agent()
    out = agent.prompt("plain text", attachments=None)
    assert out == "ok"


def test_agent_prompt_attachments_empty_list_ok():
    agent = _agent()
    out = agent.prompt("plain text", attachments=[])
    assert out == "ok"


def test_agent_prompt_rejects_non_attachment():
    with pytest.raises(TypeError):
        _agent().prompt("x", attachments=["not-an-attachment"])


def test_harness_methods_accept_attachments_kwarg():
    """AgentHarness 入口（真实 provider，不实际调用）接受 attachments kwarg。"""
    provider = senza.providers.openai(api_key="test-key")
    h = senza.HarnessBuilder("gpt-4o").provider("gpt-*", provider).build()
    # Idle 阶段 steer/follow_up 静默丢失（runtime 契约），但不抛 TypeError。
    h.steer("x", attachments=[])
    h.follow_up("x", attachments=None)
    h.next_turn("x", attachments=[senza.image_base64(PNG_1X1)])
    assert h.phase() == "idle"
