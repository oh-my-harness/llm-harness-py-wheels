"""Tests for AfterRunHook and OnAbortHook Python bindings."""

import threading
import time

import senza

# ── Surface tests (factory existence) ──────────────────────────────────────


def test_after_run_factory_exists():
    """senza.hooks.after_run should be callable."""
    assert hasattr(senza.hooks, "after_run")


def test_after_run_creates_hook():
    """after_run should return a Hook object."""

    def my_callback():
        pass

    hook = senza.hooks.after_run(my_callback)
    assert hook is not None


def test_after_run_creates_hook_instance():
    """The returned object is an instance of the Hook class."""
    hook = senza.hooks.after_run(lambda: None)
    assert isinstance(hook, senza.Hook)


def test_after_run_accepts_async_callback():
    """after_run should accept an async def callback."""

    async def my_async_callback():
        pass

    hook = senza.hooks.after_run(my_async_callback)
    assert hook is not None


def test_on_abort_factory_exists():
    """senza.hooks.on_abort should be callable."""
    assert hasattr(senza.hooks, "on_abort")


def test_on_abort_creates_hook():
    """on_abort should return a Hook object."""

    def my_callback():
        pass

    hook = senza.hooks.on_abort(my_callback)
    assert hook is not None


def test_on_abort_creates_hook_instance():
    """The returned object is an instance of the Hook class."""
    hook = senza.hooks.on_abort(lambda: None)
    assert isinstance(hook, senza.Hook)


# ── Behavioral tests (callback actually invoked) ───────────────────────────
#
# These tests verify the end-to-end Python→Rust→Python dispatch path:
# senza.hooks.on_abort(cb) → PyOnAbortHook::on_abort() → cb.call0(py) → cb().
#
# Key insight: harness.abort() always invokes on_abort hooks synchronously
# (runtime core.rs:832-834), even when no prompt is running. This lets us
# test the full binding path without a live LLM provider.


def test_on_abort_callback_invoked_on_abort():
    """harness.abort() synchronously fires the on_abort Python callback.

    Builds a harness with a fake provider, registers an on_abort hook,
    calls abort() (no prompt running), and asserts the callback ran.
    """
    called = []

    def on_abort_cb():
        called.append(True)

    provider = senza.providers.openai(api_key="test-key")
    harness = (
        senza.HarnessBuilder("gpt-4o")
        .provider("gpt-*", provider)
        .hooks([senza.hooks.on_abort(on_abort_cb)])
        .build()
    )

    assert called == []  # not yet
    harness.abort()
    assert called == [True], "on_abort callback should fire on abort()"


def test_on_abort_callback_invoked_during_run():
    """abort() during a prompt fires on_abort, then the run terminates.

    Uses a fake provider pointing at an unreachable endpoint. A background
    thread sends a prompt (which will hang on connect); the main thread
    aborts after a short delay. The on_abort callback must fire.
    """
    called = []

    def on_abort_cb():
        called.append(True)

    provider = senza.providers.openai(api_key="test-key", base_url="http://127.0.0.1:1/v1")
    harness = (
        senza.HarnessBuilder("gpt-4o")
        .provider("gpt-*", provider)
        .hooks([senza.hooks.on_abort(on_abort_cb)])
        .build()
    )

    def do_prompt():
        try:
            harness.prompt("hello")
        except Exception:
            pass

    t = threading.Thread(target=do_prompt, daemon=True)
    t.start()

    # Give the prompt a moment to start, then abort.
    time.sleep(0.3)
    harness.abort()
    t.join(timeout=5)

    assert called, "on_abort callback should fire when abort() is called during a run"


def test_multiple_on_abort_hooks_all_invoked():
    """Multiple on_abort hooks are all invoked (registration-order dispatch)."""
    order = []

    def cb_a():
        order.append("a")

    def cb_b():
        order.append("b")

    provider = senza.providers.openai(api_key="test-key")
    harness = (
        senza.HarnessBuilder("gpt-4o")
        .provider("gpt-*", provider)
        .hooks([senza.hooks.on_abort(cb_a), senza.hooks.on_abort(cb_b)])
        .build()
    )

    harness.abort()
    assert order == ["a", "b"], "hooks should fire in registration order"
