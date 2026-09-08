"""Senza — oh-my-harness runtime Python SDK."""

from __future__ import annotations

from typing import Any, AsyncGenerator, Callable, Iterator, Optional, Union

# ── Exceptions ───────────────────────────────────────────────────────────────

class SenzaError(RuntimeError):
    """Base class for all Senza exceptions."""

class ProviderError(SenzaError):
    """LLM provider error."""

class RateLimitError(ProviderError):
    """Rate limit exceeded. Carries retry_after."""

    retry_after: Optional[float]

class ProviderTimeoutError(ProviderError):
    """Provider request timed out."""

class InvalidRequestError(ProviderError):
    """Invalid request sent to the provider."""

class UnauthorizedError(ProviderError):
    """Provider rejected the request due to missing/invalid credentials."""

class ForbiddenError(ProviderError):
    """Provider rejected the request due to insufficient permissions."""

class OverloadedError(ProviderError):
    """Provider overloaded. Carries retry_after."""

    retry_after: Optional[float]

class ServerError(ProviderError):
    """Provider returned a server error."""

class StreamError(ProviderError):
    """Streaming error from the provider."""

class StreamIncompleteError(ProviderError):
    """Stream ended before completion. Carries received_chunks and finish_reason."""

    received_chunks: int
    finish_reason: Optional[str]

class NetworkError(ProviderError):
    """Network error talking to the provider."""

class DecodeError(ProviderError):
    """Failed to decode the provider response."""

class ProviderCodeError(ProviderError):
    """Provider-returned error code. Carries code."""

    code: str

class ToolError(SenzaError):
    """Tool execution error."""

class ToolArgumentError(ToolError):
    """Invalid tool arguments."""

    tool_name: Optional[str]

class ToolAbortedError(ToolError):
    """Tool was aborted."""

class ToolExecutionError(ToolError):
    """Tool execution failed."""

class BudgetExceededError(SenzaError):
    """Budget limit exceeded. Carries limit and spent."""

    limit: float
    spent: float

class WorkflowError(SenzaError):
    """Workflow execution error."""

class StepTimeoutError(WorkflowError):
    """Step timed out. Carries step_id and timeout_ms."""

    step_id: str
    timeout_ms: int

class StepFailedError(WorkflowError):
    """Step failed. Carries step_id."""

    step_id: str

class WorkflowPausedError(WorkflowError):
    """Workflow was paused."""

class ValidationError(ValueError):
    """Workflow validation error."""

class HarnessStateError(SenzaError):
    """Harness is in wrong state for the operation."""

class CompactionError(SenzaError):
    """Compaction error."""

class StreamIdleTimeoutError(SenzaError):
    """Stream idle timeout."""

class RustPanicError(RuntimeError):
    """Raised when the Rust runtime panics, instead of crashing the process."""
    def add_note(self, note: str) -> None: ...
    def with_traceback(self, tb: Any) -> RustPanicError: ...

# ── Version & utilities ──────────────────────────────────────────────────────

def enable_debug() -> None: ...
def disable_debug() -> None: ...
def version() -> str: ...
def set_event_loop(loop: Any) -> None: ...

# ── Async streaming (issue #11) ──────────────────────────────────────────────

def stream_events(
    obj: Any, timeout_ms: int = ..., max_consecutive_timeouts: int = ...
) -> AsyncGenerator[dict, None]: ...
def stream_prompt(
    obj: Any, text: str, timeout_ms: int = ..., max_consecutive_timeouts: int = ...
) -> AsyncGenerator[dict, None]: ...
def stream_run(
    engine: Any, timeout_ms: int = ..., max_consecutive_timeouts: int = ...
) -> AsyncGenerator[dict, None]: ...
def to_json(obj: Any) -> str: ...
def from_json(json_str: str) -> Any: ...
def read_sessions(dir: str) -> dict: ...
def viewer_html() -> str: ...
def extract_text(events: list[dict]) -> str: ...

class EventType:
    """String constants for event types."""

    TEXT_DELTA: str
    TOOL_CALL_START: str
    TOOL_CALL_END: str
    TOOL_RESULT: str
    MESSAGE_END: str
    THINKING_DELTA: str
    ERROR: str
    AGENT_END: str
    SETTLED: str
    ABORTED: str
    WORKFLOW_DONE: str
    WORKFLOW_FAILED: str

def tool(*args, **kwargs) -> Tool: ...

# ── Provider ─────────────────────────────────────────────────────────────────

class Provider:
    """LLM provider handle (OpenAI-compatible or Anthropic)."""

class providers:
    """Submodule: LLM provider factories."""
    @staticmethod
    def openai(
        api_key: str,
        base_url: Optional[str] = ...,
        chat_path: Optional[str] = ...,
        thinking_scheme: Optional[str] = ...,
        parse_reasoning_content: bool = ...,
        tolerant_keepalive: bool = ...,
        documents: bool = ...,
        documents_inline: bool = ...,
    ) -> Provider: ...
    @staticmethod
    def anthropic(
        api_key: str,
        base_url: Optional[str] = ...,
        messages_path: Optional[str] = ...,
    ) -> Provider: ...

# ── Pricing ─────────────────────────────────────────────────────────────────

class PricingProvider:
    """Pricing provider handle (from create_pricing_provider)."""

def create_pricing_provider(table: dict) -> PricingProvider: ...
def create_pricing_provider_callback(
    callback: Callable[[str, str], Optional[dict]],
) -> PricingProvider: ...

# ── Usage ledger ──────────────────────────────────────────────────────────────

class UsageLedger:
    """Caller-owned usage accounting state, shareable across multiple harnesses."""

    def snapshot(self) -> dict: ...

# ── ResponseFormat ───────────────────────────────────────────────────────────

class ResponseFormat:
    """Response format handle (from create_json_object_format / create_json_schema_format)."""

def create_json_object_format() -> ResponseFormat: ...
def create_json_schema_format(
    name: str,
    schema: Union[dict, str],
    strict: Optional[bool] = ...,
) -> ResponseFormat: ...

# ── Budget control ──────────────────────────────────────────────────────────

class BudgetExceededHook:
    """Budget exceeded hook handle (from create_budget_exceeded_hook)."""

def create_budget_exceeded_hook(
    callback: Callable[[dict, float], bool],
) -> BudgetExceededHook: ...

# ── Rules approval ───────────────────────────────────────────────────────────

class Predicate:
    """Predicate handle (from create_*_predicate)."""

class RuleChain:
    """Rule chain handle (from RuleChainBuilder.build())."""

class RuleChainBuilder:
    """Builder for RuleChain (from create_rule_chain())."""

    def rule(self, tool_name: str, predicate: Predicate, on_match: str) -> RuleChainBuilder: ...
    def fallback(self, decision: str) -> RuleChainBuilder: ...
    def build(self) -> RuleChain: ...

class rules:
    """Submodule: rule chain and predicate factories."""
    @staticmethod
    def chain() -> RuleChainBuilder: ...
    @staticmethod
    def contains(allowed: list[str]) -> Predicate: ...
    @staticmethod
    def regex_field(arg_path: str, pattern: str) -> Predicate: ...
    @staticmethod
    def number_range(arg_path: str, min: float, max: float) -> Predicate: ...
    @staticmethod
    def rate_limit(max: int, window_seconds: float) -> Predicate: ...
    @staticmethod
    def approval_hook(chain: RuleChain) -> Hook: ...

# ── Skills ───────────────────────────────────────────────────────────────────

class Skill:
    """Skill handle (from load_skills). Immutable."""

    @property
    def name(self) -> str: ...
    @property
    def label(self) -> Optional[str]: ...
    @property
    def description(self) -> str: ...
    @property
    def source(self) -> str: ...
    @property
    def base_dir(self) -> str: ...
    @property
    def disable_model_invocation(self) -> bool: ...

def load_skills(path: str) -> list[Skill]: ...

# ── Tool ─────────────────────────────────────────────────────────────────────

class ToolContext:
    """Context passed to a tool callback."""

    def is_cancelled(self) -> bool: ...
    def send_update(self, result: Any) -> None: ...

class Tool:
    """A tool registered on a harness or workflow engine."""

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...

def create_tool(
    name: str,
    description: str,
    parameters: Optional[Union[dict, str]] = ...,
    parameters_schema: Optional[Union[dict, str]] = ...,
    callback: Callable[..., Any] = ...,
    report_duration: bool = ...,
) -> Tool: ...
def create_sync_tool(
    name: str,
    description: str,
    parameters_schema: str,
    callback: Callable[..., Any],
) -> Tool: ...

# ── Plugin ───────────────────────────────────────────────────────────────────

class Plugin:
    """A bundle of tools and hooks."""

def create_plugin(
    name: str,
    tools: Optional[list[Tool]] = ...,
    hooks: Optional[list[Hook]] = ...,
) -> Plugin: ...
def create_fs_tools_plugin() -> Plugin: ...

# ── Native tool wrapper ──────────────────────────────────────────────────────

class NativeTool:
    """Opaque wrapper for a Rust-native Tool."""
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...

# ── Web / Code tools ─────────────────────────────────────────────────────────

def create_web_search_tool(
    config: Optional[dict] = ...,
) -> NativeTool: ...

def create_web_fetch_tool(
    config: Optional[dict] = ...,
) -> NativeTool: ...

def create_web_tools_plugin(
    config: Optional[dict] = ...,
) -> Plugin: ...

def create_code_exec_tool(
    timeout_secs: Optional[int] = ...,
) -> NativeTool: ...

# ── Multimodal attachments ──────────────────────────────────────────

class Attachment:
    """Opaque multimodal attachment (image or document)."""

def image_url(url: str) -> Attachment: ...
def image_base64(data: bytes, mime_type: str = "image/png") -> Attachment: ...
def document_url(url: str, name: str | None = None) -> Attachment: ...
def document_file(path: str, name: str | None = None) -> Attachment: ...
# ── Inspector ────────────────────────────────────────────────────────────────

class Inspector:
    """Agent Inspector Web API handle. Drop to shut down."""
    @property
    def bound_addr(self) -> Optional[str]: ...
    def shutdown(self) -> None: ...

# ── Knowledge ────────────────────────────────────────────────────────────────

class KnowledgeSource:
    """Opaque knowledge source handle (from create_local_knowledge_source)."""

class knowledge:
    """Submodule: knowledge source, memory, and session-recall factories."""
    @staticmethod
    def local_source(
        path: str,
        source_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        domains: Optional[list[str]] = None,
        max_document_bytes: int = 1048576,
    ) -> KnowledgeSource: ...
    @staticmethod
    def plugin(
        sources: list[KnowledgeSource],
        config: Optional[dict] = None,
    ) -> Plugin: ...
    @staticmethod
    def memory_store(read_source_id: str) -> MemoryStore: ...
    @staticmethod
    def memory_plugin(
        source: KnowledgeSource,
        store: MemoryStore,
        policy: MemoryWritePolicy,
        gate: Optional[MemoryMutationGate] = None,
    ) -> Plugin: ...
    @staticmethod
    def secure_write_policy(config: Optional[dict] = None) -> MemoryWritePolicy: ...
    @staticmethod
    def allow_all_gate() -> MemoryMutationGate: ...
    @staticmethod
    def in_memory_session_recall_index() -> SessionRecallIndex: ...
    @staticmethod
    def sqlite_session_recall_index(path: str) -> SessionRecallIndex: ...
    @staticmethod
    def in_memory_session_repo() -> SessionRepo: ...
    @staticmethod
    def jsonl_session_repo(root_dir: str) -> SessionRepo: ...
    @staticmethod
    def session_recall_knowledge_source(
        repo: SessionRepo, index: SessionRecallIndex
    ) -> SessionRecallKnowledgeSource: ...
    @staticmethod
    def history_recall_plugin(
        source: SessionRecallKnowledgeSource,
        config: Optional[dict] = None,
    ) -> Plugin: ...

# ── Memory ──────────────────────────────────────────────────────────────────

class MemoryStore:
    """Opaque memory store handle (from create_in_memory_store)."""

class MemoryWritePolicy:
    """Opaque memory write policy handle (from create_secure_write_policy)."""

class MemoryMutationGate:
    """Opaque memory mutation gate handle (from create_allow_all_gate)."""

# ── Session Recall ──────────────────────────────────────────────────────────

class SessionRecallIndex:
    """Opaque session recall index handle."""

class SessionRepo:
    """Opaque session repo handle."""

class SessionRecallKnowledgeSource:
    """Opaque session recall knowledge source handle."""
    def as_knowledge_source(self) -> KnowledgeSource: ...

# ── Strategy plugins ──────────────────────────────────────────────────────────

class MemoryDefensePluginBuilder:
    """Builder for MemoryDefensePlugin with fluent configuration."""
    def extra_file(self, name: str) -> "MemoryDefensePluginBuilder": ...
    def extra_files(self, names: list[str]) -> "MemoryDefensePluginBuilder": ...
    def build(self) -> Plugin: ...

class strategy:
    """Submodule: strategy plugin factories."""
    @staticmethod
    def safety_defaults() -> Plugin: ...
    @staticmethod
    def loop_safety(config: Optional[dict] = None) -> Plugin: ...
    @staticmethod
    def status_panel() -> Plugin: ...
    @staticmethod
    def memory_defense() -> Plugin: ...
    @staticmethod
    def injection_filter(patterns: Optional[list[str]] = None) -> Plugin: ...
    @staticmethod
    def source_tag(entries: list[dict]) -> Plugin: ...
    @staticmethod
    def project_instruction(env: ExecutionEnv, config: Optional[dict] = None) -> Plugin: ...
    @staticmethod
    def audit(
        sink_path: str, trace_id: Optional[str] = None, task_id: Optional[str] = None
    ) -> Plugin: ...
    @staticmethod
    def notify() -> Plugin: ...
    @staticmethod
    def tool_output_guard(env: ExecutionEnv, config: Optional[dict] = None) -> Plugin: ...
    @staticmethod
    def webhook_stream(buffer: int) -> tuple[WebhookChannel, EventStream]: ...
    @staticmethod
    def context_aware_compaction_prompt() -> tuple[str, str]: ...
    @staticmethod
    def vision_degrade() -> Hook: ...
    @staticmethod
    def observation_shielding(config: Optional[dict] = None) -> Hook: ...

# ── Webhook event stream ──────────────────────────────────────────────────────

class WebhookChannel:
    """Sender side of a webhook event stream. Call push() to inject events."""

    def push(self, payload: Any) -> None: ...

class EventStream:
    """Consumer side of a webhook event stream (opaque)."""

# ── Timer / Heartbeat / Shell monitor event streams ──────────────────────────

class HeartbeatHandle:
    """Handle for a heartbeat stream. Call tick() to reset the watchdog."""

    def tick(self) -> None: ...

class ShellMonitorHandle:
    """Handle for a shell monitor stream. Call kill() to terminate the process."""

    def kill(self) -> None: ...

def create_timer_stream(
    delay_ms: int,
    label: str,
    task_id: str,
) -> tuple[WaitForExternalEventTool]: ...
def create_heartbeat_stream(
    timeout_ms: int,
    label: str,
    task_id: str,
) -> tuple[HeartbeatHandle, WaitForExternalEventTool]: ...
def create_shell_monitor_stream(
    command: str,
    cwd: Optional[str],
    label: str,
    task_id: str,
) -> tuple[ShellMonitorHandle, WaitForExternalEventTool]: ...

# ── Hook (15 types) ──────────────────────────────────────────────────────────

class Hook:
    """Opaque hook wrapper."""

class hooks:
    """Submodule: hook factories (15 lifecycle hooks)."""
    @staticmethod
    def before_turn(callback: Callable[[dict], None]) -> Hook: ...
    @staticmethod
    def after_turn(callback: Callable[[dict], None]) -> Hook: ...
    @staticmethod
    def before_run(callback: Callable[[dict], None]) -> Hook: ...
    @staticmethod
    def after_provider_response(callback: Callable[[dict], None]) -> Hook: ...
    @staticmethod
    def before_provider_request(callback: Callable[[dict], None]) -> Hook: ...
    @staticmethod
    def before_tool_call(callback: Callable[[dict], Optional[str]]) -> Hook: ...
    @staticmethod
    def after_tool_call(callback: Callable[[dict], Any]) -> Hook: ...
    @staticmethod
    def should_stop(callback: Callable[[dict], bool]) -> Hook: ...
    @staticmethod
    def before_compact(callback: Callable[[dict], Any]) -> Hook: ...
    @staticmethod
    def transform_context(callback: Callable[[dict], dict]) -> Hook: ...
    @staticmethod
    def prepare_next_turn(callback: Callable[[dict], Optional[dict]]) -> Hook: ...
    @staticmethod
    def final_answer_validator(
        callback: Callable[[dict], Optional[Union[str, dict]]],
    ) -> Hook: ...
    @staticmethod
    def after_run(callback: Callable[[], None]) -> Hook: ...
    @staticmethod
    def on_abort(callback: Callable[[], None]) -> Hook: ...
    @staticmethod
    def provider_error(callback: Callable[[dict], Optional[str]]) -> Hook: ...

# ── Event channel (human-in-the-loop) ────────────────────────────────────────

class EventStreamHandle:
    """Handle for submitting external events."""

    def submit(self, content: str, details: dict) -> None: ...

class WaitForExternalEventTool:
    """Tool that pauses the LLM to wait for an external event."""

    def name(self) -> str: ...
def create_event_channel(task_id: str) -> tuple[EventStreamHandle, WaitForExternalEventTool]: ...

class HumanResponseHandle:
    """Handle for submitting human responses (auto request_id)."""

    def submit(self, content: str, details: dict) -> None: ...

class HumanApprovalTool:
    """Tool that requests human approval (approve/deny, with fail-safe default)."""

    def name(self) -> str: ...
    def description(self) -> str: ...

class HumanInputTool:
    """Tool that requests human input (free-form value, with default)."""

    def name(self) -> str: ...
    def description(self) -> str: ...

def create_human_approval_channel(
    task_id: str,
    timeout_seconds: float = ...,
    default: str = ...,
) -> tuple[HumanResponseHandle, HumanApprovalTool]: ...

def create_human_input_channel(
    task_id: str,
    timeout_seconds: float = ...,
    default: Any = ...,
) -> tuple[HumanResponseHandle, HumanInputTool]: ...


# ── Judge ────────────────────────────────────────────────────────────────────

class Judge:
    """Opaque judge wrapper (from create_judge)."""

def create_judge(callback: Callable[[dict], str]) -> Judge: ...

class CompositeJudge:
    """Per-step routing judge. Use create_composite_judge() to instantiate."""

    def on(self, step: str, callback: Callable[[dict], str]) -> None: ...
    def fallback(self, callback: Callable[[dict], str]) -> None: ...

def create_composite_judge() -> CompositeJudge: ...

# ── Executor ─────────────────────────────────────────────────────────────────

class Executor:
    """Opaque executor wrapper."""

class ExecutionEnv:
    """Opaque execution environment wrapper (e.g. from create_os_env)."""

def create_executor(callback: Callable[[dict], dict]) -> Executor: ...
def create_shell_executor(
    commands: list[str],
    default_timeout_ms: int = ...,
    max_output_bytes: int = ...,
) -> Executor: ...
def create_http_executor(
    allowed_hosts: list[str],
    allowed_schemes: Optional[list[str]] = ...,
    max_timeout_ms: int = ...,
    allow_private_ip_targets: bool = ...,
) -> Executor: ...
def create_os_env(working_dir: str = ...) -> ExecutionEnv: ...

# ── MCP (Model Context Protocol) ─────────────────────────────────────────────

class McpServerConfig:
    """MCP server configuration (stdio / HTTP / SSE)."""

    @staticmethod
    def stdio(
        command: str,
        args: Optional[list[str]] = ...,
        env: Optional[dict[str, str]] = ...,
        cwd: Optional[str] = ...,
        timeout: Optional[int] = ...,
    ) -> McpServerConfig: ...
    @staticmethod
    def http(
        url: str,
        headers: Optional[dict[str, str]] = ...,
        timeout: Optional[int] = ...,
    ) -> McpServerConfig: ...
    @staticmethod
    def sse(
        url: str,
        headers: Optional[dict[str, str]] = ...,
        timeout: Optional[int] = ...,
    ) -> McpServerConfig: ...

class McpManager:
    """MCP multi-server lifecycle manager."""

    def add_server(self, name: str, config: McpServerConfig) -> None: ...
    def load_config_file(self, path: str) -> None: ...
    def list_tools(self) -> list[str]: ...
    def get_status(self, name: str) -> str: ...
    def reconnect(self, name: str) -> None: ...
    def disconnect_server(self, name: str) -> None: ...
    def disconnect_all(self) -> None: ...
    def errors(self) -> dict[str, str]: ...

# ── Agent layer: HarnessBuilder ──────────────────────────────────────────────

class HarnessBuilder:
    """Fluent builder for AgentHarness."""

    def __init__(self, model: str) -> None: ...
    def provider(self, pattern: str, provider: Provider) -> HarnessBuilder: ...
    def system_prompt(self, prompt: str) -> HarnessBuilder: ...
    def max_tokens(self, tokens: int) -> HarnessBuilder: ...
    def temperature(self, temp: Optional[float]) -> HarnessBuilder: ...
    def thinking_level(self, level: str) -> HarnessBuilder: ...
    def env(self, env: ExecutionEnv) -> HarnessBuilder: ...
    def tool(self, tool: Tool) -> HarnessBuilder: ...
    def tools(self, tools: list[Tool]) -> HarnessBuilder: ...
    def plugin(self, plugin: Plugin) -> HarnessBuilder: ...
    def auto_compact(self, enabled: bool) -> HarnessBuilder: ...
    def compaction_reserve_tokens(self, tokens: Optional[int]) -> HarnessBuilder: ...
    def compaction_keep_recent_tokens(self, tokens: Optional[int]) -> HarnessBuilder: ...
    def should_stop_hook(self, hook: Hook) -> HarnessBuilder: ...
    def after_turn_hook(self, hook: Hook) -> HarnessBuilder: ...
    def response_format(self, fmt: Optional[ResponseFormat]) -> HarnessBuilder: ...
    def knowledge_access(
        self, scope: str = ..., principal: str = ..., kind: str = ...
    ) -> HarnessBuilder: ...
    def hooks(self, hooks_list: list[Hook]) -> HarnessBuilder: ...
    def retry(self, max_retries: int, base_delay_ms: int) -> HarnessBuilder: ...
    def model_info(self, context_window: int, max_tokens: int) -> HarnessBuilder: ...
    def final_answer_mode(self, mode: str) -> HarnessBuilder: ...
    def final_answer_validator(self, validator: Hook) -> HarnessBuilder: ...
    def disable_skill_read_tool(self) -> HarnessBuilder: ...
    def skill(self, skill: Skill) -> HarnessBuilder: ...
    def skills(self, skills: list[Skill]) -> HarnessBuilder: ...
    def compaction_model(
        self,
        model: str,
        context_window: int,
        max_tokens: int,
    ) -> HarnessBuilder: ...
    def compaction_prompt(
        self,
        system_prompt: Optional[str] = ...,
        user_template: Optional[str] = ...,
    ) -> HarnessBuilder: ...
    def compaction_query(self, query: Optional[str] = ...) -> HarnessBuilder: ...
    def pricing(self, provider: PricingProvider) -> HarnessBuilder: ...
    def usage_ledger(self, ledger: UsageLedger) -> HarnessBuilder: ...
    def stream_options(
        self,
        timeout_ms: Optional[int] = ...,
        max_retries: Optional[int] = ...,
    ) -> HarnessBuilder: ...
    def queue_capacity(self, capacity: Optional[int] = ...) -> HarnessBuilder: ...
    def budget(
        self,
        limit: float,
        exceeded_hook: Optional[BudgetExceededHook] = ...,
    ) -> HarnessBuilder: ...
    def mcp_server(self, name: str, config: McpServerConfig) -> HarnessBuilder: ...
    def mcp_config_file(self, path: str) -> HarnessBuilder: ...
    def with_mcp_manager(self, manager: McpManager) -> HarnessBuilder: ...
    def enable_spawn(
        self,
        model: str,
        provider: Provider,
        session_dir: str,
        max_concurrent: Optional[int] = ...,
    ) -> HarnessBuilder: ...
    def session_repo(
        self, repo: SessionRepo, session_id: Optional[str] = ...
    ) -> HarnessBuilder: ...
    def build(self) -> AgentHarness: ...

# ── Agent layer: AgentHarness ────────────────────────────────────────────────

class HarnessEventIterator:
    """Iterator over harness events.

    Event types include:
    - text_delta, thinking_delta: token-level streaming
    - message_start, message_update, message_end: message lifecycle
    - tool_call_start, tool_call_args_delta, tool_call_end: LLM streaming a tool call
    - tool_execution_start, tool_execution_update, tool_execution_end: tool execution lifecycle
    - harness_tool_call_start, harness_tool_call_end: harness-level tool call wrappers
    - phase_change, compaction_start, compaction_end: harness lifecycle
    - settled, aborted, error: terminal events
    """

class AgentHarness:
    """Single-agent LLM harness with tool calling and streaming."""

    # ── Prompting ──
    def prompt(self, text: str, attachments: Optional[list[Attachment]] = ...) -> None: ...
    def prompt_and_collect(
        self, text: str, timeout_ms: int = ..., attachments: Optional[list[Attachment]] = ...
    ) -> list[dict]: ...
    async def prompt_async(
        self, text: str, timeout_ms: int = 30000, attachments: Optional[list[Attachment]] = ...
    ) -> list[dict]: ...
    def chat(self, text: str, timeout_ms: int = 30000, attachments: Optional[list[Attachment]] = ...) -> str: ...
    async def chat_async(
        self, text: str, timeout_ms: int = 30000, attachments: Optional[list[Attachment]] = ...
    ) -> str: ...
    def collect_until_settled(self, timeout_ms: int = ...) -> list[dict]: ...
    def events(
        self, timeout_ms: int = ..., max_consecutive_timeouts: int = ...
    ) -> HarnessEventIterator: ...
    def abort(self) -> None: ...
    def inspect(self) -> dict: ...

    # ── Dynamic config ──
    def set_model(
        self,
        model: str,
        context_window: Optional[int] = ...,
        max_tokens: Optional[int] = ...,
    ) -> None: ...
    def set_system_prompt(self, prompt: Optional[str] = ...) -> None: ...
    def set_temperature(self, temperature: Optional[float] = ...) -> None: ...
    def set_thinking_level(self, level: str) -> None: ...
    def set_max_tokens(self, max_tokens: int) -> None: ...
    def set_tools(self, tools: list[Tool]) -> None: ...
    def set_active_tools(self, tools: Optional[list[str]] = ...) -> None: ...

    # ── Steering ──
    def steer(self, text: str, attachments: Optional[list[Attachment]] = ...) -> None: ...
    def follow_up(self, text: str, attachments: Optional[list[Attachment]] = ...) -> None: ...
    def next_turn(self, text: str, attachments: Optional[list[Attachment]] = ...) -> None: ...
    def continue_run(self) -> None: ...
    def compact(self) -> dict: ...
    def session_metadata(self) -> dict: ...

    # ── Queue management ──
    def clear_steering_queue(self) -> None: ...
    def clear_follow_up_queue(self) -> None: ...
    def clear_all_queues(self) -> None: ...
    def has_queued_messages(self) -> bool: ...

    # ── Inspection ──
    def get_messages(self) -> list[dict]: ...
    def last_response(self) -> str: ...
    def message_count(self) -> int: ...
    def phase(self) -> str: ...
    def usage(self) -> dict: ...
    def usage_ledger(self) -> dict: ...
    def reset_usage(self) -> None: ...

    # ── Waiting ──
    def wait_for_idle(self) -> None: ...
    def wait_for_settled(self) -> None: ...

    # ── Session / Branch ──
    def fork_branch(self, from_entry: str, label: Optional[str] = ...) -> str: ...
    def navigate_tree(self, target: str) -> None: ...
    def list_branches(self) -> list[dict]: ...
    def read_active_path(self) -> list[dict]: ...
    def read_all_entries(self) -> list[dict]: ...
    def delete_branch(self, leaf: str) -> None: ...
    def generate_branch_summary(self, leaf: str) -> dict: ...
    def shutdown(self) -> None: ...
    def mount_inspector(self, port: int = 8080) -> Inspector: ...

    # ── Context manager ──
    def __enter__(self) -> AgentHarness: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool: ...

# ── Runtime layer: WorkflowEngine ────────────────────────────────────────────

class WorkflowEventIterator:
    """Iterator over workflow events (step_started, step_finished, etc.)."""

class WorkflowEngine:
    """Multi-step workflow engine with conditional routing and crash recovery."""

    def __init__(
        self,
        workflow_dict: dict,
        provider: Provider,
        model: str,
        judge: Judge | CompositeJudge,
        session_base_dir: str = ...,
        env: Optional[ExecutionEnv] = ...,
    ) -> None: ...
    def restore(
        cls,
        task_store_dir: str,
        task_id: str,
        provider: Provider,
        model: str,
        judge: Judge | CompositeJudge,
        session_base_dir: str = ...,
        env: Optional[ExecutionEnv] = ...,
    ) -> WorkflowEngine: ...
    @classmethod
    def restore_from_step(
        cls,
        task_store_dir: str,
        task_id: str,
        step: str,
        provider: Provider,
        model: str,
        judge: Judge | CompositeJudge,
        session_base_dir: str = ...,
        env: Optional[ExecutionEnv] = ...,
    ) -> WorkflowEngine: ...

    # ── Registration (chainable) ──
    def with_tool(self, tool: Tool) -> WorkflowEngine: ...
    def with_external_tool(self, tool: WaitForExternalEventTool) -> WorkflowEngine: ...
    def with_executor(self, name: str, executor: Executor) -> WorkflowEngine: ...
    def with_hooks(self, hooks_list: list[Hook]) -> WorkflowEngine: ...
    def with_step_plugin(self, step_id: str, plugin: Plugin) -> WorkflowEngine: ...
    def with_step_builder(
        self, step_id: str, customize: Callable[[HarnessBuilder], HarnessBuilder]
    ) -> WorkflowEngine: ...
    def with_task_store(self, dir: str) -> WorkflowEngine: ...
    @classmethod
    def list_tasks(task_store_dir: str) -> list[dict]: ...
    def with_max_tokens(self, max_tokens: int) -> WorkflowEngine: ...
    def with_max_steps(self, max: int) -> WorkflowEngine: ...
    def with_max_retries(self, max: int) -> WorkflowEngine: ...
    def with_thinking_level(self, level: str) -> WorkflowEngine: ...
    def with_pricing(self, provider: PricingProvider) -> WorkflowEngine: ...

    # ── Context ──
    def set_context_variable(self, key: str, value: Any) -> None: ...
    def get_context_variable(self, key: str) -> Any: ...

    # ── Execution ──
    def run(self) -> None: ...
    async def run_async(self, timeout_ms: int = 300000) -> list[dict]: ...
    def pause(self, reason: str) -> None: ...
    def resume(self) -> None: ...
    def cancel(self, reason: str) -> None: ...

    # ── Inspection ──
    def state(self) -> str: ...
    def current_step(self) -> Optional[str]: ...
    def step_history(self) -> list[dict]: ...
    def task_id(self) -> str: ...
    def total_cost(self) -> dict: ...
    def checkpoint(self, description: str, payload: Any) -> None: ...
    def subscribe(
        self, timeout_ms: int = ..., max_consecutive_timeouts: int = ...
    ) -> WorkflowEventIterator: ...
    def inspect(self) -> dict: ...

    # ── Context manager ──
    def __enter__(self) -> WorkflowEngine: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool: ...

# ── Misc classes (opaque wrappers) ───────────────────────────────────────────

class EventIterator:
    """Generic event iterator."""

# ── Infra: Audit / Trace / Sandbox ──────────────────────────────────────────

class JsonlAuditSink:
    """JSONL file-backed audit sink with SHA-256 hash-chain integrity."""

    def __init__(self, path: str) -> None: ...
    @staticmethod
    def validate(path: str) -> int: ...

class InMemoryTraceExporter:
    """In-memory trace exporter for testing. Accumulates SpanEvent values."""

    def exported_spans(self) -> list[dict]: ...
    def exported_span_count(self) -> int: ...

class Sandbox:
    """Sandbox wrapper (SeatbeltSandbox on macOS, BwrapSandbox on Linux)."""

    def is_running(self) -> bool: ...
    def start(self) -> None: ...

class infra:
    """Submodule: infrastructure (audit sink, trace exporter, sandbox)."""

    jsonl_audit_sink: type[JsonlAuditSink]
    in_memory_trace_exporter: type[InMemoryTraceExporter]
    @staticmethod
    def seatbelt_sandbox(config: Optional[dict] = None) -> Sandbox: ...
    @staticmethod
    def bwrap_sandbox(config: Optional[dict] = None) -> Sandbox: ...
