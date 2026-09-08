"""Runtime layer live tests: workflow engine, recovery, executors, composite judge, audit/trace, sandbox."""

import os
import tempfile

import pytest

import senza
from base import live_model, make_harness, provider_or_skip, run_prompt


def _flow():
    return {
        "entry_step": "writer",
        "steps": [
            {
                "id": "writer",
                "name": "writer",
                "prompt": "Write one short sentence about the ocean.",
                "allowed_tools": [],
            },
            {
                "id": "reviewer",
                "name": "reviewer",
                "prompt": "Repeat the first word of the previous output.",
                "allowed_tools": [],
            },
        ],
        "edges": [{"from": "writer", "to": "reviewer"}],
    }


def _judge():
    def judge(ctx):
        return "done" if ctx["step_id"] == "reviewer" else "to:reviewer"

    return senza.create_judge(judge)


def test_builder_workflow():
    model = live_model()
    engine = senza.WorkflowEngine(_flow(), provider_or_skip(), model, _judge())
    engine.run()
    assert engine.state() == "succeeded", f"state={engine.state()}"
    assert len(engine.step_history()) >= 2


def test_workflow_recovery():
    model = live_model()
    tmp = tempfile.mkdtemp(prefix="senza_recover_")
    engine = senza.WorkflowEngine(_flow(), provider_or_skip(), model, _judge()).with_task_store(tmp)
    engine.set_context_variable("note", "persist me")
    engine.run()
    tid = engine.task_id()
    restored = senza.WorkflowEngine.restore(tmp, tid, provider_or_skip(), model, _judge())
    assert restored.state() == "succeeded"


def test_shell_executor():
    model = live_model()
    wf = {
        "entry_step": "s",
        "steps": [
            {
                "id": "s",
                "name": "s",
                "executor": "shell",
                "executor_config": {"command": "echo", "args": ["hi"]},
            }
        ],
        "edges": [],
    }
    engine = senza.WorkflowEngine(
        wf,
        provider_or_skip(),
        model,
        senza.create_judge(lambda ctx: "done"),
        env=senza.create_os_env("."),
    ).with_executor("shell", senza.create_shell_executor(["echo"]))
    engine.run()
    assert engine.state() == "succeeded"


def test_composite_judge():
    model = live_model()
    wf = {
        "entry_step": "a",
        "steps": [
            {"id": "a", "name": "a", "prompt": "Say the word ready.", "allowed_tools": []},
            {"id": "b", "name": "b", "prompt": "Repeat the previous answer.", "allowed_tools": []},
        ],
        "edges": [{"from": "a", "to": "b"}],
    }
    judge = senza.create_composite_judge()
    judge.on("a", lambda ctx: "to:b")
    judge.on("b", lambda ctx: "done")
    engine = senza.WorkflowEngine(wf, provider_or_skip(), model, judge)
    engine.run()
    assert engine.state() == "succeeded"


def test_tracing_audit():
    audit_path = os.path.join(tempfile.mkdtemp(), "audit.jsonl")
    with open(audit_path, "w"):
        pass  # JsonlAuditSink opens lazily; touch so validate() can read
    plugin = senza.strategy.audit(sink_path=audit_path, trace_id="lt", task_id="t1")
    h = make_harness(provider_or_skip(), lambda b: b.plugin(plugin))
    ev = run_prompt(h, "Say hello.")
    types = [e.get("type") for e in ev]
    assert "settled" in types, f"expected settled, got {types}"
    assert senza.JsonlAuditSink.validate(audit_path) >= 0


def test_sandbox():
    if not hasattr(senza.infra, "seatbelt_sandbox"):
        return  # platform-specific (macOS only); nothing to assert elsewhere
    sb = senza.infra.seatbelt_sandbox()
    assert sb is not None
    assert sb.is_running() is False


def test_runtime_constructs_offline():
    """No key needed; validates workflow engine construction + judge wiring."""
    stub = senza.providers.openai(api_key="sk-test")
    e = senza.WorkflowEngine(_flow(), stub, live_model(), _judge())
    assert e is not None and e.state() == "idle"


# ── Spawn tests (mirrors runtime runtime_layer/spawn.rs) ─────────────────


def _make_spawn_harness(provider, session_dir):
    """Build a harness with spawn infrastructure enabled."""
    builder = senza.HarnessBuilder(live_model()).provider("*", provider)
    builder = builder.enable_spawn(
        model=live_model(),
        provider=provider,
        session_dir=session_dir,
    ).system_prompt(
        "You are a helpful assistant that can dispatch sub-agents for sub-tasks. "
        "Use the spawn_agent tool to delegate work, await_subagent_reply to wait "
        "for results, and query_subagent to check status."
    )
    return builder.build()
def test_spawn_async_completes():
    """spawn_agent tool is called and spawn completes."""
    provider = provider_or_skip()
    session_dir = tempfile.mkdtemp()
    h = _make_spawn_harness(provider, session_dir)
    ev = run_prompt(
        h,
        "Use the spawn_agent tool to spawn a sub-agent with the prompt 'Say hello in one word.'",
        timeout_ms=120_000,
    )
    from base import assert_settled, assert_tool_called

    assert_settled(ev)
    assert_tool_called(ev, "spawn_agent")


def test_await_subagent_reply():
    """spawn_agent + await_subagent_reply delivers a reply."""
    provider = provider_or_skip()
    session_dir = tempfile.mkdtemp()
    h = _make_spawn_harness(provider, session_dir)
    ev = run_prompt(
        h,
        "Spawn a sub-agent with prompt 'What is 2+2? Answer with just the number.' "
        "Then use await_subagent_reply to get the response.",
        timeout_ms=120_000,
    )
    from base import assert_settled, assert_tool_called

    assert_settled(ev)
    assert_tool_called(ev, "spawn_agent")
    assert_tool_called(ev, "await_subagent_reply")


def test_query_subagent_status():
    """spawn_agent + query_subagent returns status."""
    provider = provider_or_skip()
    session_dir = tempfile.mkdtemp()
    h = _make_spawn_harness(provider, session_dir)
    ev = run_prompt(
        h,
        "Spawn a sub-agent with prompt 'Count from 1 to 5.' "
        "Then use query_subagent to check its status.",
        timeout_ms=120_000,
    )
    from base import assert_settled, assert_tool_called

    assert_settled(ev)
    assert_tool_called(ev, "spawn_agent")
    assert_tool_called(ev, "query_subagent")
