"""Regression test for issue #29: invalid tool-call JSON no longer crashes agent.

Non-OpenAI models (GLM, DeepSeek, Qwen) occasionally emit tool-call arguments
with invalid JSON. Previously this caused a fatal AgentError::Internal that
crashed prompt(). Now the parse error is embedded in the tool args as a
degradation JSON, the tool executes, and the loop continues — giving the model
a chance to self-correct.

This test requires the `test-utils` feature (senza.Agent + MockLlmClient).
On production wheels it is auto-skipped by conftest.py.
"""

import senza


def _echo_tool():
    """A simple tool that echoes its args back as text."""
    return senza.create_tool(
        name="echo",
        description="Echo the input message",
        parameters_schema='{"type":"object","properties":{"msg":{"type":"string"}},"required":["msg"]}',
        callback=lambda args, ctx: {
            "content": [{"type": "text", "text": f"echo: {args.get('msg', args)}"}],
            "terminate": False,
        },
    )


def test_invalid_tool_args_json_does_not_crash():
    """Agent recovers when the model returns invalid JSON in tool args (#29).

    First mock response: tool_use with broken JSON (missing closing brace).
    Second mock response: clean text — simulating the model self-correcting
    after receiving the tool result with the parse error.
    """
    responses = [
        {"type": "tool_use", "id": "bad-1", "name": "echo", "args": r'{"msg":"hello"'},
        {"type": "text", "text": "recovered"},
    ]
    agent = senza.Agent(
        model="mock-model",
        responses=responses,
        tools=[_echo_tool()],
    )

    # Before the fix this raised RuntimeError("failed to parse tool args...").
    # After the fix it returns normally — the tool executed with degraded args
    # and the model self-corrected on the second call.
    result = agent.prompt("echo hello")

    assert isinstance(result, str)
    assert agent.phase() == "idle"
    assert agent.error_message is None, f"unexpected error_message: {agent.error_message}"


def test_invalid_tool_args_end_turn_does_not_crash():
    """Agent recovers from invalid JSON + EndTurn stop_reason (#29 + 41e2a59).

    Combines two non-standard provider behaviors:
    - tool-call arguments with invalid JSON
    - stop_reason=EndTurn instead of ToolUse

    The message must be classified as Progress (not FinalAnswer) so the loop
    continues to execute the tool and feed the result back.
    """
    responses = [
        {"type": "tool_use_end_turn", "id": "bad-1", "name": "echo", "args": r'{"msg":"hello"'},
        {"type": "text", "text": "recovered"},
    ]
    agent = senza.Agent(
        model="mock-model",
        responses=responses,
        tools=[_echo_tool()],
    )

    result = agent.prompt("echo hello")

    assert isinstance(result, str)
    assert agent.phase() == "idle"
    assert agent.error_message is None, f"unexpected error_message: {agent.error_message}"


def test_valid_tool_args_still_work():
    """Sanity check: valid tool-call JSON works as before."""
    responses = [
        {"type": "tool_use", "id": "ok-1", "name": "echo", "args": r'{"msg":"hello"}'},
        {"type": "text", "text": "done"},
    ]
    agent = senza.Agent(
        model="mock-model",
        responses=responses,
        tools=[_echo_tool()],
    )

    result = agent.prompt("echo hello")

    assert isinstance(result, str)
    assert agent.phase() == "idle"
    assert agent.error_message is None
