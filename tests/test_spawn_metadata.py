"""enable_spawn(max_concurrent=...) + AgentHarness.session_metadata()。"""

import senza


def test_enable_spawn_accepts_max_concurrent():
    stub = senza.providers.openai(api_key="sk-test")
    b = senza.HarnessBuilder("test-model").provider("*", stub)
    b.enable_spawn("test-model", stub, "/tmp/senza-test-sessions", max_concurrent=2)
    h = b.build()
    assert h is not None


def test_enable_spawn_backward_compat():
    """不带 max_concurrent 时签名兼容。"""
    stub = senza.providers.openai(api_key="sk-test")
    b = senza.HarnessBuilder("test-model").provider("*", stub)
    b.enable_spawn("test-model", stub, "/tmp/senza-test-sessions")
    h = b.build()
    assert h is not None


def test_session_metadata_returns_dict():
    stub = senza.providers.openai(api_key="sk-test")
    h = senza.HarnessBuilder("test-model").provider("*", stub).build()
    md = h.session_metadata()
    assert isinstance(md, dict)
    assert md["id"], "session id should be non-empty"
    assert "created_at" in md
    assert "updated_at" in md
