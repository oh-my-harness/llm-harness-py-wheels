"""Shared helpers for Senza live LLM integration tests.

Mirrors `llm-harness-runtime/crates/llm-harness-live-tests`.

Run any test that drives a real LLM with a key configured; without one the
test skips gracefully (see `providers_from_env` / `provider_or_skip`). Each
layer file also ships an offline *construction* smoke (no key needed) that
validates every API call signature.

Default OpenAI-compatible provider is the current OMP DeepSeek endpoint:

    base_url = http://api.hyper-op.com/v1   (openai-completions)
    model    = DeepSeek-V4-Flash

Overridable via env:
    OPENAI_API_KEY / ANTHROPIC_API_KEY      provider keys
    OPENAI_API_BASE                         default: http://api.hyper-op.com/v1
    SENZA_LIVE_MODEL                        explicit shared model override
    ANTHROPIC_MODEL                         Anthropic-only default override

If `OPENAI_API_KEY` is unset, `providers_from_env` sources `~/.omp_llm_env`
(the current OMP session's LLM env) so the DeepSeek setup "just works".
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import senza

# ── Timeouts (mirror runtime live-tests tiering) ───────────────────────────
SMOKE_TIMEOUT_MS = 30_000
SINGLE_TURN_TIMEOUT_MS = 60_000
MULTI_TURN_TIMEOUT_MS = 120_000

# ── OMP DeepSeek default ────────────────────────────────────────────────────
DEFAULT_MODEL = "DeepSeek-V4-Flash"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_BASE_URL = "http://api.hyper-op.com/v1"
_OMP_LLM_ENV = Path.home() / ".omp_llm_env"


def _load_omp_env() -> None:
    """Source ~/.omp_llm_env into os.environ if a key isn't already set."""
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        return
    if _OMP_LLM_ENV.exists():
        for line in _OMP_LLM_ENV.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export ") :]
            if "=" not in line or line.startswith("#"):
                continue
            k, _, v = line.partition("=")
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)


_load_omp_env()


def live_model() -> str:
    explicit = os.environ.get("SENZA_LIVE_MODEL")
    if explicit:
        return explicit
    if os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        return os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    return DEFAULT_MODEL


def live_base_url() -> str:
    return os.environ.get("OPENAI_API_BASE", DEFAULT_BASE_URL)


def providers_from_env() -> list[tuple[str, senza.Provider]]:
    """Build (name, provider) pairs from env; empty list when none configured.

    A test should call `provider_or_skip(providers)` and `return` if None.
    """
    entries: list[tuple[str, senza.Provider]] = []
    ok = os.environ.get("OPENAI_API_KEY")
    if ok:
        base = live_base_url()
        # parse_reasoning_content=True + tolerant_keepalive=True are Senza's
        # defaults — required for the DeepSeek/GLM reasoning endpoint.
        entries.append(("openai", senza.providers.openai(api_key=ok, base_url=base)))
    ak = os.environ.get("ANTHROPIC_API_KEY")
    if ak:
        entries.append(("anthropic", senza.providers.anthropic(api_key=ak)))
    return entries


def provider_or_skip() -> senza.Provider:
    """Return the first configured provider, or call pytest.skip()."""
    import pytest

    entries = providers_from_env()
    if not entries:
        pytest.skip("no LLM provider configured (set OPENAI_API_KEY or ANTHROPIC_API_KEY)")
    return entries[0][1]


def document_provider_or_skip() -> tuple[senza.Provider, str]:
    """Return (provider, model) known to accept document (PDF) input, or skip.

    Discovery order (verified by real requests 2026-08-29):
    1. SENZA_DOCUMENT_BASE_URL / SENZA_DOCUMENT_MODEL / SENZA_DOCUMENT_API_KEY
       — explicit override for a document-capable endpoint.
    2. The repo `.env` enabled ANTHROPIC block (claude-sonnet via the local
       gateway), which reads PDF URLs natively.
    3. Otherwise skip — most gateways reject document parts.
    """
    import pytest

    base = os.environ.get("SENZA_DOCUMENT_BASE_URL")
    model = os.environ.get("SENZA_DOCUMENT_MODEL")
    key = os.environ.get("SENZA_DOCUMENT_API_KEY")
    if not base:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        if env_file.exists():
            vals: dict[str, str] = {}
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                vals[k.strip()] = v.strip().strip('"').strip("'")
            if vals.get("ANTHROPIC_API_KEY") and vals.get("ANTHROPIC_BASE_URL"):
                base = vals["ANTHROPIC_BASE_URL"]
                model = model or vals.get("ANTHROPIC_MODEL")
                key = key or vals["ANTHROPIC_API_KEY"]
    if not base or not key:
        pytest.skip("no document-capable provider configured (set SENZA_DOCUMENT_BASE_URL)")
    return (
        senza.providers.openai(api_key=key, base_url=base, documents=True),
        model,
    )


def make_harness(provider, customize=None, *, model=None):
    """Build an AgentHarness bound to a real provider.

    `customize` is `Callable[[HarnessBuilder], HarnessBuilder]` applied after
    the builder is seeded with the selected model + provider.
    """
    builder = senza.HarnessBuilder(model or live_model()).provider("*", provider)
    if customize:
        builder = customize(builder)
    return builder.build()


def run_prompt(harness, text, timeout_ms=SINGLE_TURN_TIMEOUT_MS, attachments=None):
    """prompt_and_collect with an explicit per-call timeout."""
    return harness.prompt_and_collect(text, timeout_ms=timeout_ms, attachments=attachments)


def with_timeout(seconds, fn, *args, **kwargs):
    """Run a sync callable on a worker thread with a hard wall-clock timeout."""

    async def _run():
        return await asyncio.wait_for(asyncio.to_thread(fn, *args, **kwargs), timeout=seconds)

    return asyncio.run(_run())


# ── Event assertions (event dicts from prompt_and_collect / stream) ────────


def event_types(events) -> list[str]:
    return [e.get("type") for e in events]


def text_of(events) -> str:
    return senza.extract_text(events)


def assert_tool_called(events, name: str) -> None:
    """Assert at least one tool_call_start for `name` was emitted."""
    names = [
        e.get("tool_name")
        for e in events
        if e.get("type") in ("tool_call_start", "tool_execution_start")
    ]
    assert name in names, f"expected tool '{name}' to be called, got {names}"


def assert_settled(events) -> None:
    assert "settled" in event_types(events), f"expected settled, got {event_types(events)}"


def assert_no_error(events) -> None:
    errs = [e for e in events if e.get("type") == "error"]
    assert not errs, f"unexpected error events: {errs}"
