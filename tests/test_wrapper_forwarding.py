"""Python wrapper 转发 attachments：chat / chat_async / prompt_async。"""

import pytest
import senza

pytestmark = pytest.mark.skipif(
    not hasattr(senza, "Agent"),
    reason="requires the test-utils feature (senza.Agent / MockLlmClient)",
)


def test_chat_wrapper_signature():
    """AgentHarness.chat 是 Python wrapper，签名含 attachments。"""
    import inspect

    sig = inspect.signature(senza.AgentHarness.chat)
    assert "attachments" in sig.parameters


def test_chat_async_wrapper_signature():
    import inspect

    sig = inspect.signature(senza.AgentHarness.chat_async)
    assert "attachments" in sig.parameters


def test_prompt_async_wrapper_signature():
    import inspect

    sig = inspect.signature(senza.AgentHarness.prompt_async)
    assert "attachments" in sig.parameters
