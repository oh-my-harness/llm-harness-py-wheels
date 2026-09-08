"""Tests for provider_error hook (15th hook kind)."""

import pytest
import senza

# ── surface tests ───────────────────────────────────────────────────────────


def test_hooks_namespace_has_provider_error():
    """The hooks submodule exposes provider_error."""
    assert hasattr(senza.hooks, "provider_error")


def test_provider_error_returns_hook_instance():
    hook = senza.hooks.provider_error(lambda ctx: "retry")
    assert isinstance(hook, senza.Hook)


def test_provider_error_accepts_async_callback():
    async def cb(ctx):
        return None

    hook = senza.hooks.provider_error(cb)
    assert isinstance(hook, senza.Hook)


def test_multiple_provider_error_hooks_independent():
    h1 = senza.hooks.provider_error(lambda ctx: "retry")
    h2 = senza.hooks.provider_error(lambda ctx: "surface")
    assert h1 is not h2


def test_submodules_exports_provider_error():
    from senza import hooks

    assert callable(hooks.provider_error)


# ── builder integration ─────────────────────────────────────────────────────


def _make_builder():
    return senza.HarnessBuilder("claude-3-5-sonnet")


def test_builder_provider_error_hook_chainable():
    """PyHarnessBuilder.provider_error_hook returns self for chaining."""
    builder = _make_builder()
    hook = senza.hooks.provider_error(lambda ctx: "surface")
    result = builder.provider_error_hook(hook)
    assert result is builder


def test_builder_provider_error_hook_rejects_wrong_kind():
    """Passing a different hook kind raises TypeError (fail-closed)."""
    builder = _make_builder()
    wrong = senza.hooks.before_turn(lambda ctx: None)
    with pytest.raises(TypeError):
        builder.provider_error_hook(wrong)
