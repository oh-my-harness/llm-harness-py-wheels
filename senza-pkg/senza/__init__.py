"""Senza — Python SDK for llm-harness runtime."""

from __future__ import annotations

from .senza import *  # noqa: F401, F403

import asyncio as _asyncio
import threading as _threading
from typing import Any, AsyncGenerator

_TERMINAL_TYPES = frozenset(
    {"agent_end", "error", "settled", "aborted", "workflow_done", "workflow_failed"}
)

_STOP = object()


def _get_event_iterator(obj: Any, timeout_ms: int, max_consecutive_timeouts: int) -> Any:
    """Return the sync event iterator for *obj*, regardless of class."""
    if hasattr(obj, "events"):
        return obj.events(timeout_ms=timeout_ms, max_consecutive_timeouts=max_consecutive_timeouts)
    if hasattr(obj, "subscribe"):
        return obj.subscribe(
            timeout_ms=timeout_ms, max_consecutive_timeouts=max_consecutive_timeouts
        )
    raise TypeError(f"{type(obj).__name__} has no events() or subscribe() method")


async def _next_event(it: Any) -> Any:
    """Call next(it) in a thread, converting StopIteration to a sentinel.

    ``asyncio.to_thread`` cannot propagate ``StopIteration`` because it
    interacts badly with the generator protocol, so we catch it in the
    worker thread and return ``_STOP`` instead.
    """

    def _step() -> Any:
        try:
            return next(it)
        except StopIteration:
            return _STOP

    result = await _asyncio.to_thread(_step)
    return result


async def stream_events(
    obj: Any,
    timeout_ms: int = 5000,
    max_consecutive_timeouts: int = 1,
) -> AsyncGenerator[dict, None]:
    """Async generator yielding events from an Agent, AgentHarness, or WorkflowEngine.

    Wraps the synchronous event iterator, releasing the GIL during each
    ``__next__`` call so the asyncio event loop stays responsive.

    Usage::

        async for event in senza.stream_events(agent, timeout_ms=5000):
            print(event["type"])
    """
    it = _get_event_iterator(obj, timeout_ms, max_consecutive_timeouts)
    while True:
        event = await _next_event(it)
        if event is _STOP:
            break
        yield event


async def stream_prompt(
    obj: Any,
    text: str,
    timeout_ms: int = 5000,
    max_consecutive_timeouts: int = 1,
) -> AsyncGenerator[dict, None]:
    """Send a prompt and yield events as they arrive (Agent / AgentHarness).

    Starts ``obj.prompt(text)`` on a background thread, then yields events
    until a terminal event (``agent_end``, ``settled``, ``aborted``,
    ``error``) is received or the stream is exhausted.

    Args:
        obj: An Agent, AgentHarness, or any object with ``prompt()`` and
            ``events()`` (or ``subscribe()``) methods.
        text: The prompt text to send.
        timeout_ms: Per-poll timeout in milliseconds. Each call to
            ``next(event_iterator)`` blocks for at most this long before
            returning (with or without an event).
        max_consecutive_timeouts: Maximum number of consecutive empty polls
            before the stream is considered exhausted. Set to a large value
            (e.g. ``999999``) when tools may block for a long time — e.g.
            ``ask_user`` waiting for human input. Default ``1`` for backward
            compatibility.

    Usage::

        async for event in senza.stream_prompt(agent, "hello"):
            print(event)

    With long-blocking tools::

        async for event in senza.stream_prompt(
            harness, "ask the user", timeout_ms=30000,
            max_consecutive_timeouts=999999,
        ):
            print(event)
    """
    it = _get_event_iterator(obj, timeout_ms, max_consecutive_timeouts)

    done = _threading.Event()
    errors: list = []

    def _do_prompt() -> None:
        try:
            obj.prompt(text)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            done.set()

    t = _threading.Thread(target=_do_prompt, daemon=True)
    t.start()

    try:
        while True:
            event = await _next_event(it)
            if event is _STOP:
                break
            yield event
            if event.get("type") in _TERMINAL_TYPES:
                break
    finally:
        done.wait(timeout=60)
        t.join(timeout=60)
        if errors:
            raise errors[0]


async def stream_run(
    engine: Any,
    timeout_ms: int = 5000,
    max_consecutive_timeouts: int = 1,
) -> AsyncGenerator[dict, None]:
    """Start ``engine.run()`` on a background thread and yield workflow events.

    Args:
        engine: A WorkflowEngine or any object with ``run()`` and
            ``events()`` (or ``subscribe()``) methods.
        timeout_ms: Per-poll timeout in milliseconds.
        max_consecutive_timeouts: Maximum consecutive empty polls before the
            stream is considered exhausted. See :func:`stream_prompt`.

    Usage::

        async for event in senza.stream_run(engine):
            print(event["type"])
    """
    it = _get_event_iterator(engine, timeout_ms, max_consecutive_timeouts)

    done = _threading.Event()
    errors: list = []

    def _do_run() -> None:
        try:
            engine.run()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            done.set()

    t = _threading.Thread(target=_do_run, daemon=True)
    t.start()

    try:
        while True:
            event = await _next_event(it)
            if event is _STOP:
                break
            yield event
            if event.get("type") in _TERMINAL_TYPES:
                break
    finally:
        done.wait(timeout=120)
        t.join(timeout=120)
        if errors:
            raise errors[0]


# ── extract_text helper ──────────────────────────────────────────────


def extract_text(events):
    """Extract concatenated text from a list of agent events.

    Filters for ``text_delta`` events and concatenates their ``text``
    field. Non-text events are skipped. Missing ``text`` fields are
    treated as empty strings.

    Args:
        events: List of event dicts (e.g. from ``harness.prompt_and_collect()``).

    Returns:
        Concatenated text string.
    """
    return "".join(event.get("text", "") for event in events if event.get("type") == "text_delta")


# ── Event type constants ─────────────────────────────────────────────


class EventType:
    """String constants for event types.

    Use these instead of raw strings to avoid typos::

        if event["type"] == senza.EventType.TEXT_DELTA:
            text += event["text"]
    """

    TEXT_DELTA = "text_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_RESULT = "tool_result"
    MESSAGE_END = "message_end"
    THINKING_DELTA = "thinking_delta"
    ERROR = "error"
    AGENT_END = "agent_end"
    SETTLED = "settled"
    ABORTED = "aborted"
    WORKFLOW_DONE = "workflow_done"
    WORKFLOW_FAILED = "workflow_failed"


# ── create_tool Python wrapper ───────────────────────────────────────
# Rust-layer create_tool already accepts dict schema, but we add a Python
# wrapper to: (1) accept `parameters` as canonical name (alias for
# parameters_schema), and (2) allow single-argument callbacks.

import inspect as _inspect

_create_tool_rust = create_tool


def _wrap_tool_callback(callback):
    """Wrap a tool callback to allow single-argument (args-only) signatures.

    Rust always calls cb(args, ctx). If the user's callback only accepts
    one argument, we wrap it to ignore ctx.
    """
    try:
        sig = _inspect.signature(callback)
        params = [
            p for p in sig.parameters.values() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]
        if len(params) <= 1:
            return lambda args, ctx: callback(args)
    except (ValueError, TypeError):
        pass
    return callback


def create_tool(name, description, parameters=None, parameters_schema=None, callback=None, report_duration=False):
    """Create a Tool from a callback.

    Args:
        name: Tool name.
        description: Tool description.
        parameters: JSON Schema as dict or JSON string (canonical name).
        parameters_schema: Alias for ``parameters`` (backward compat).
            When used positionally as the 4th arg, accepts the callback
            for backward compatibility with the old Rust signature.
        callback: Callable with signature ``(args, ctx)`` or ``(args)``.
            Async callables are supported. May return a str, a dict, an
            ``Attachment``, or a list of str/``Attachment``.
        report_duration: When True, the agent loop appends an execution
            duration annotation (e.g. ``[duration: 812ms]``) to the tool
            result fed back to the model. Only takes effect when hooks
            wrap the tool (the agent loop wraps automatically).
    """
    # Backward compat: old Rust signature was create_tool(name, desc, schema, callback).
    # When called positionally, the 4th arg lands in parameters_schema and callback is None.
    if callback is None and callable(parameters_schema):
        callback = parameters_schema
        parameters_schema = None

    schema = parameters if parameters is not None else parameters_schema
    if schema is None:
        raise TypeError("create_tool() missing required argument: 'parameters'")
    if callback is None:
        raise TypeError("create_tool() missing required argument: 'callback'")
    wrapped = _wrap_tool_callback(callback)
    return _create_tool_rust(name, description, schema, wrapped, report_duration)


# ── @senza.tool decorator ────────────────────────────────────────────

import typing as _typing

_PY_TO_JSON_SCHEMA = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _build_schema_from_hints(func):
    """Build a JSON Schema dict from function type hints."""
    try:
        hints = _typing.get_type_hints(func)
    except Exception:
        hints = {}

    sig = _inspect.signature(func)
    properties = {}
    required = []

    for pname, param in sig.parameters.items():
        annotation = hints.get(pname, str)
        json_type = _PY_TO_JSON_SCHEMA.get(annotation, "string")
        prop = {"type": json_type}

        if param.default is _inspect.Parameter.empty:
            required.append(pname)
        else:
            prop["default"] = param.default

        properties[pname] = prop

    schema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


def _create_tool_from_function(func):
    """Create a Tool from a function with type hints."""
    name = func.__name__
    description = (func.__doc__ or func.__name__).strip()
    schema = _build_schema_from_hints(func)

    is_async = _inspect.iscoroutinefunction(func)
    sig = _inspect.signature(func)
    param_names = list(sig.parameters.keys())

    if is_async:

        async def wrapper(args, ctx):
            kwargs = {k: args.get(k) for k in param_names if k in args}
            return await func(**kwargs)
    else:

        def wrapper(args, ctx):
            kwargs = {k: args.get(k) for k in param_names if k in args}
            return func(**kwargs)

    return create_tool(name, description, schema, wrapper)


def tool(*args, **kwargs):
    """Create a Tool from a function or explicit parameters.

    As a decorator (no parens)::

        @senza.tool
        def search(query: str, limit: int = 10) -> str:
            \"\"\"Search the web.\"\"\"
            return f"Results for {query}"

    As a function call::

        tool = senza.tool(
            name="search",
            description="Search the web",
            parameters={"query": {"type": "string"}},
            callback=lambda args: f"Results for {args['query']}",
        )

    Type hints are used to auto-generate the JSON Schema when used as a
    decorator. The docstring becomes the tool description.
    """
    # Decorator form: @senza.tool (no parentheses)
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return _create_tool_from_function(args[0])

    # Function form: senza.tool(name=..., description=..., parameters=..., callback=...)
    name = kwargs.get("name")
    description = kwargs.get("description")
    parameters = kwargs.get("parameters")
    callback = kwargs.get("callback")

    if name is None or description is None or parameters is None or callback is None:
        raise TypeError("senza.tool() requires name, description, parameters, and callback")

    # Wrap callback to handle both (args) and (args, ctx) signatures
    cb_sig = _inspect.signature(callback)
    cb_nparams = len(cb_sig.parameters)
    if cb_nparams == 1:
        _orig = callback
        if _inspect.iscoroutinefunction(callback):

            async def _wrapped(args, ctx):
                return await _orig(args)
        else:

            def _wrapped(args, ctx):
                return _orig(args)

        callback = _wrapped

    return create_tool(name, description, parameters, callback)


# ── Multimodal attachments ───────────────────────────────────────────

import base64 as _base64
import os as _os


_DOCUMENT_MEDIA_TYPES = {".pdf": "application/pdf", ".txt": "text/plain"}
_RustAttachment = Attachment  # pyo3 class re-exported from the Rust module


def image_url(url: str):
    """Create an image attachment from a public URL."""
    return _RustAttachment("image_url", url, None, None)


def image_base64(data: bytes, mime_type: str = "image/png"):
    """Create an inline image attachment from raw bytes (base64-encoded here)."""
    return _RustAttachment("image_base64", _base64.b64encode(data).decode("ascii"), None, mime_type)


def document_url(url: str, name: str | None = None):
    """Create a document attachment from a URL. Endpoint must support document input.

    Media type is inferred from the URL extension (.pdf -> application/pdf,
    .txt -> text/plain); unknown extensions raise ValueError.
    """
    from urllib.parse import urlparse as _urlparse

    ext = _os.path.splitext(_urlparse(url).path)[1].lower()
    media = _DOCUMENT_MEDIA_TYPES.get(ext)
    if media is None:
        raise ValueError(f"unsupported document extension in URL: {ext!r}")
    return _RustAttachment("document_url", url, name, media)


def document_file(path: str, name: str | None = None):
    """Create a document attachment from a local file.

    Media type is inferred from the extension (.pdf -> application/pdf,
    .txt -> text/plain); unknown extensions raise ValueError.
    """
    ext = _os.path.splitext(path)[1].lower()
    media = _DOCUMENT_MEDIA_TYPES.get(ext)
    if media is None:
        raise ValueError(f"unsupported document extension: {ext!r}")
    with open(path, "rb") as f:
        payload = f.read()
    return _RustAttachment(
        "document_base64",
        _base64.b64encode(payload).decode("ascii"),
        name or _os.path.basename(path),
        media,
    )


# ── Async wrappers for blocking methods ──────────────────────────────


async def _workflow_run_async(self, timeout_ms: int = 300000):
    """Async version of run(). Does not block the event loop.

    Runs ``self.run()`` in a thread pool via ``asyncio.to_thread``.
    For event-streaming async usage, prefer ``senza.stream_run(engine)``.
    """
    return await _asyncio.to_thread(self.run)


async def _harness_prompt_async(self, text: str, timeout_ms: int = 30000, attachments=None):
    """Async version of prompt_and_collect(). Does not block the event loop.

    Runs ``self.prompt_and_collect(text, timeout_ms, attachments)`` in a
    thread pool via ``asyncio.to_thread``. For streaming async usage, prefer
    ``senza.stream_prompt(harness, text)``.
    """
    return await _asyncio.to_thread(self.prompt_and_collect, text, timeout_ms, attachments)


WorkflowEngine.run_async = _workflow_run_async
AgentHarness.prompt_async = _harness_prompt_async


def _harness_chat(self, text: str, timeout_ms: int = 30000, attachments=None) -> str:
    """Send a prompt and return the concatenated text response.

    Convenience wrapper around ``extract_text(prompt_and_collect(text))``.
    For streaming or event-level access, use ``prompt_and_collect()`` or
    ``stream_prompt()`` instead.
    """
    events = self.prompt_and_collect(text, timeout_ms, attachments)
    return extract_text(events)


async def _harness_chat_async(self, text: str, timeout_ms: int = 30000, attachments=None) -> str:
    """Async version of chat(). Does not block the event loop."""
    events = await _asyncio.to_thread(self.prompt_and_collect, text, timeout_ms, attachments)
    return extract_text(events)


AgentHarness.chat = _harness_chat
AgentHarness.chat_async = _harness_chat_async


# ── Debug helpers ────────────────────────────────────────────────────

import logging as _logging


def enable_debug():
    """Enable DEBUG-level logging for the senza logger.

    This sets the Python-side ``senza`` logger to DEBUG. The Rust-side
    tracing filter is controlled by the ``SENZA_LOG`` / ``RUST_LOG``
    environment variable; if you need Rust-side debug output, set
    ``SENZA_LOG=senza=debug`` before importing senza.
    """
    _logging.getLogger("senza").setLevel(_logging.DEBUG)


def disable_debug():
    """Restore INFO-level logging for the senza logger."""
    _logging.getLogger("senza").setLevel(_logging.INFO)


def _harness_inspect(self):
    """Return a snapshot of the harness state for debugging.

    Aggregates phase, message count, token usage, queued messages,
    and active tools into a single dict.
    """
    try:
        messages = self.get_messages()
        msg_count = len(messages) if messages else 0
    except Exception:
        msg_count = 0

    try:
        usage = self.usage()
    except Exception:
        usage = {}

    return {
        "message_count": msg_count,
        "usage": usage,
        "queued_messages": self.has_queued_messages()
        if hasattr(self, "has_queued_messages")
        else False,
    }


def _workflow_inspect(self):
    """Return a snapshot of the workflow engine state for debugging.

    Aggregates state, current step, step count, and total cost.
    """
    try:
        history = self.step_history()
        step_count = len(history) if history else 0
    except Exception:
        step_count = 0

    try:
        cost = self.total_cost()
    except Exception:
        cost = 0.0

    return {
        "state": self.state(),
        "current_step": self.current_step(),
        "step_count": step_count,
        "total_cost": cost,
    }


AgentHarness.inspect = _harness_inspect
WorkflowEngine.inspect = _workflow_inspect
from types import SimpleNamespace as _SimpleNamespace

# ── Grouped submodules ───────────────────────────────────────────────
# Low-frequency create_* factories are grouped into SimpleNamespace
# objects with simplified names (drop create_ prefix, drop _plugin /
# _hook / _provider / _predicate suffixes where the group name already
# conveys the meaning).

_providers = _SimpleNamespace(
    openai=create_openai_provider,
    anthropic=create_anthropic_provider,
)

_hooks = _SimpleNamespace(
    before_turn=create_before_turn_hook,
    after_turn=create_after_turn_hook,
    before_run=create_before_run_hook,
    after_provider_response=create_after_provider_response_hook,
    before_provider_request=create_before_provider_request_hook,
    before_tool_call=create_before_tool_call_hook,
    after_tool_call=create_after_tool_call_hook,
    should_stop=create_should_stop_hook,
    before_compact=create_before_compact_hook,
    transform_context=create_transform_context_hook,
    prepare_next_turn=create_prepare_next_turn_hook,
    final_answer_validator=create_final_answer_validator,
    after_run=create_after_run_hook,
    on_abort=create_on_abort_hook,
    provider_error=create_provider_error_hook,
)

_strategy = _SimpleNamespace(
    safety_defaults=create_safety_defaults_plugin,
    loop_safety=create_loop_safety_plugin,
    tool_output_guard=create_tool_output_guard_plugin,
    vision_degrade=create_vision_degrade_hook,
    observation_shielding=create_observation_shielding_hook,
    status_panel=create_status_panel_plugin,
    memory_defense=create_memory_defense_plugin,
    injection_filter=create_injection_filter_plugin,
    source_tag=create_source_tag_plugin,
    project_instruction=create_project_instruction_plugin,
    audit=create_audit_plugin,
    notify=create_notify_plugin,
    webhook_stream=create_webhook_stream,
    context_aware_compaction_prompt=create_context_aware_compaction_prompt,
)

_knowledge = _SimpleNamespace(
    local_source=create_local_knowledge_source,
    plugin=create_knowledge_plugin,
    memory_store=create_in_memory_store,
    memory_plugin=create_memory_plugin,
    secure_write_policy=create_secure_write_policy,
    allow_all_gate=create_allow_all_gate,
    in_memory_session_recall_index=create_in_memory_session_recall_index,
    sqlite_session_recall_index=create_sqlite_session_recall_index,
    in_memory_session_repo=create_in_memory_session_repo,
    jsonl_session_repo=create_jsonl_session_repo,
    session_recall_knowledge_source=create_session_recall_knowledge_source,
    history_recall_plugin=create_history_recall_plugin,
)

_infra = _SimpleNamespace(
    jsonl_audit_sink=JsonlAuditSink,
    in_memory_trace_exporter=InMemoryTraceExporter,
)

# Platform-specific sandbox factories — only one exists at runtime.
if "create_seatbelt_sandbox" in dir():
    _infra.seatbelt_sandbox = create_seatbelt_sandbox
if "create_bwrap_sandbox" in dir():
    _infra.bwrap_sandbox = create_bwrap_sandbox

_rules = _SimpleNamespace(
    chain=create_rule_chain,
    contains=create_contains_predicate,
    regex_field=create_regex_field_predicate,
    number_range=create_number_range_predicate,
    rate_limit=create_rate_limit_predicate,
    approval_hook=create_rule_approval_hook,
)

# ── Remove low-frequency create_* from top-level namespace ──────────
# These are now only accessible via the submodule groups above.
del create_openai_provider
del create_anthropic_provider
del create_before_turn_hook
del create_after_turn_hook
del create_before_run_hook
del create_after_provider_response_hook
del create_before_provider_request_hook
del create_before_tool_call_hook
del create_after_tool_call_hook
del create_should_stop_hook
del create_before_compact_hook
del create_transform_context_hook
del create_prepare_next_turn_hook
del create_final_answer_validator
del create_after_run_hook
del create_on_abort_hook
del create_provider_error_hook
del create_safety_defaults_plugin
del create_loop_safety_plugin
del create_status_panel_plugin
del create_memory_defense_plugin
del create_injection_filter_plugin
del create_source_tag_plugin
del create_project_instruction_plugin
del create_audit_plugin
del create_notify_plugin
del create_tool_output_guard_plugin
del create_vision_degrade_hook
del create_observation_shielding_hook
del create_webhook_stream
del create_context_aware_compaction_prompt
del create_local_knowledge_source
del create_knowledge_plugin
del create_in_memory_store
del create_memory_plugin
del create_secure_write_policy
del create_allow_all_gate
del create_in_memory_session_recall_index
del create_sqlite_session_recall_index
del create_in_memory_session_repo
del create_jsonl_session_repo
del create_session_recall_knowledge_source
del create_history_recall_plugin
del create_rule_chain
del create_contains_predicate
del create_regex_field_predicate
del create_number_range_predicate
del create_rate_limit_predicate
del create_rule_approval_hook
if "create_seatbelt_sandbox" in dir():
    del create_seatbelt_sandbox
if "create_bwrap_sandbox" in dir():
    del create_bwrap_sandbox

# ── Public submodule aliases ────────────────────────────────────────
# Expose the grouped namespaces without the leading underscore so
# users can call senza.providers.openai(...), senza.hooks.before_turn(...), etc.
providers = _providers
hooks = _hooks
strategy = _strategy
knowledge = _knowledge
infra = _infra
rules = _rules

# ── Public API whitelist ─────────────────────────────────────────────
__all__ = [
    # Classes
    "Attachment",
    "HarnessBuilder",
    "AgentHarness",
    "WorkflowEngine",
    "UsageLedger",
    "Provider",
    "Tool",
    "ToolContext",
    "Plugin",
    "Judge",
    "CompositeJudge",
    "Executor",
    "ExecutionEnv",
    "ResponseFormat",
    "Skill",
    "Hook",
    "KnowledgeSource",
    "MemoryStore",
    "MemoryWritePolicy",
    "MemoryMutationGate",
    "SessionRepo",
    "SessionRecallIndex",
    "SessionRecallKnowledgeSource",
    "JsonlAuditSink",
    "InMemoryTraceExporter",
    "Sandbox",
    "PricingProvider",
    "BudgetExceededHook",
    "Predicate",
    "RuleChain",
    "RuleChainBuilder",
    "McpServerConfig",
    "McpManager",
    "WebhookChannel",
    "EventStream",
    "HeartbeatHandle",
    "ShellMonitorHandle",
    "EventStreamHandle",
    "WaitForExternalEventTool",
    "HarnessEventIterator",
    "WorkflowEventIterator",
    "EventIterator",
    "MemoryDefensePluginBuilder",
    # Factory functions (top-level)
    "create_tool",
    "image_url",
    "image_base64",
    "document_url",
    "document_file",
    "create_sync_tool",
    "create_judge",
    "create_composite_judge",
    "create_plugin",
    "create_fs_tools_plugin",
    "create_os_env",
    "create_event_channel",
    "create_human_approval_channel",
    "create_human_input_channel",
    "create_executor",
    "create_shell_executor",
    "create_http_executor",
    "create_pricing_provider",
    "create_pricing_provider_callback",
    "create_budget_exceeded_hook",
    "create_json_object_format",
    "create_json_schema_format",
    "create_timer_stream",
    "create_heartbeat_stream",
    "create_shell_monitor_stream",
    "load_skills",
    # Submodules
    "providers",
    "hooks",
    "strategy",
    "knowledge",
    "infra",
    "rules",
    # Decorators and helpers
    "tool",
    "extract_text",
    "EventType",
    "stream_events",
    "stream_prompt",
    "stream_run",
    # Debug / utilities
    "enable_debug",
    "disable_debug",
    "version",
    "set_event_loop",
    "to_json",
    "from_json",
    "read_sessions",
    # Exceptions
    "SenzaError",
    "ProviderError",
    "RateLimitError",
    "ProviderTimeoutError",
    "InvalidRequestError",
    "UnauthorizedError",
    "ForbiddenError",
    "OverloadedError",
    "ServerError",
    "StreamError",
    "StreamIncompleteError",
    "NetworkError",
    "DecodeError",
    "ProviderCodeError",
    "ToolError",
    "ToolArgumentError",
    "ToolAbortedError",
    "ToolExecutionError",
    "BudgetExceededError",
    "WorkflowError",
    "StepTimeoutError",
    "StepFailedError",
    "WorkflowPausedError",
    "ValidationError",
    "HarnessStateError",
    "CompactionError",
    "StreamIdleTimeoutError",
    "RustPanicError",
]
