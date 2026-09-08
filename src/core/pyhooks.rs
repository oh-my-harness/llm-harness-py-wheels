//! 14 个 hook trait 的 Python 回调包装（5 个通知 + 4 个决策 + 2 个变换 + 1 个验证 + 1 个 run 结束 + 1 个 abort）。
//!
//! 统一模式：`Arc<Py<PyAny>>` 持有 callback，`spawn_blocking` +
//! `Python::attach` + `call1` 调用 Python 函数。
//! Context struct 中的 `&'a` 引用字段在进入 `spawn_blocking` 前转换为
//! owned 数据（clone 或序列化），避免跨线程借用。
//!
//! # asyncio 回调调度
//!
//! `async def` callback 通过 [`crate::core::pyloop::run_coro`] 执行。当用户通过
//! `senza.set_event_loop(loop)` 注册了正在运行的 event loop 时，coroutine 会被
//! `asyncio.run_coroutine_threadsafe()` 调度到该 loop 上执行，可安全使用主 loop
//! 的 asyncio 原语（`asyncio.Lock`、`asyncio.Queue` 等）。未注册 loop 时回退到
//! `asyncio.run()`（创建临时 loop）。

use std::str::FromStr;
use std::sync::Arc;

use futures::future::BoxFuture;
use llm_harness_types::{
    AfterProviderResponseCtx, AfterProviderResponseHook, AfterRunHook, AfterToolCallCtx,
    AfterToolCallDecision, AfterToolCallHook, AfterTurnCtx, AfterTurnHook, AgentContext,
    AgentError, AgentMessage, AssistantMessage, BeforeCompactCtx, BeforeCompactDecision,
    BeforeCompactHook, BeforeProviderRequestCtx, BeforeProviderRequestHook, BeforeRunCtx,
    BeforeRunHook, BeforeRunResult, BeforeToolCallCtx, BeforeToolCallDecision, BeforeToolCallHook,
    BeforeTurnCtx, BeforeTurnHook, CompactionResult, DataBlock, ErrorDirective,
    FinalAnswerValidationCtx, FinalAnswerValidationError, FinalAnswerValidator, NextTurnDirective,
    OnAbortHook, PrepareNextTurnCtx, PrepareNextTurnHook, ProviderErrorCtx, ProviderErrorHook,
    RunContext, ShouldStopCtx, ShouldStopHook, StopReason, ToolError, ToolFailure, ToolResult,
    TransformContextCtx, TransformContextHook,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;

use crate::shared::value_conv::{pyobject_to_value, value_to_pyobject};

// ── 共享类型转换器 ────────────────────────────────────────────────────────────

/// 将 `AssistantMessage`（实现了 Serialize）转为 Python dict。
pub fn assistant_message_to_dict(py: Python<'_>, msg: &AssistantMessage) -> PyResult<Py<PyAny>> {
    let json: Value = serde_json::to_value(msg)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    value_to_pyobject(py, &json)
}

/// 将 `&[AgentMessage]`（实现了 Serialize）转为 Python list。
pub fn agent_messages_to_list(py: Python<'_>, msgs: &[AgentMessage]) -> PyResult<Py<PyAny>> {
    let json: Value = serde_json::to_value(msgs)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    value_to_pyobject(py, &json)
}

/// 将 `ToolResult`（未实现 Serialize）手动转为 Python dict。
pub fn tool_result_to_dict(py: Python<'_>, result: &ToolResult) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    let content_list = PyList::empty(py);
    for block in &result.model_content {
        let block_json: Value = serde_json::to_value(block)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        content_list.append(value_to_pyobject(py, &block_json)?)?;
    }
    dict.set_item("content", content_list)?;
    dict.set_item("details", value_to_pyobject(py, &result.details)?)?;
    dict.set_item("terminate", result.terminate)?;
    Ok(dict.into_any().unbind())
}

/// 将 `ToolError` 转为字符串（含 anyhow::Error，不可序列化）。
pub fn tool_error_to_string(err: &ToolError) -> String {
    err.to_string()
}

/// 将 `StopReason` 转为字符串。
pub fn stop_reason_to_str(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndTurn => "end_turn",
        StopReason::MaxTokens => "max_tokens",
        StopReason::StopSequence => "stop_sequence",
        StopReason::ToolUse => "tool_use",
        StopReason::Other => "other",
    }
}

/// 将 `AgentContext`（未实现 Serialize）手动转为 Python dict。
pub fn agent_context_to_dict(py: Python<'_>, ctx: &AgentContext) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    match &ctx.system_prompt {
        Some(s) => dict.set_item("system_prompt", s)?,
        None => dict.set_item("system_prompt", py.None())?,
    }
    dict.set_item("messages", agent_messages_to_list(py, &ctx.messages)?)?;
    Ok(dict.into_any().unbind())
}

/// 从 RunContext 提取 run_id 和 started_at 字符串。
fn run_context_fields(run: &RunContext) -> (String, String) {
    (run.id().to_string(), run.started_at().to_rfc3339())
}

// ── BeforeTurnHook ──────────────────────────────────────────────────────────

/// Python callable 包装为 `BeforeTurnHook`。
///
/// callback 签名：`callback(ctx: dict) -> None`
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
pub struct PyBeforeTurnHook {
    callback: Arc<Py<PyAny>>,
    is_async: bool,
}

impl PyBeforeTurnHook {
    pub fn new(callback: Py<PyAny>) -> Self {
        let is_async = detect_async(&callback);
        Self {
            callback: Arc::new(callback),
            is_async,
        }
    }
}

impl BeforeTurnHook for PyBeforeTurnHook {
    fn before_turn<'a>(&'a self, ctx: BeforeTurnCtx<'a>) -> BoxFuture<'a, ()> {
        let cb = Arc::clone(&self.callback);
        let is_async = self.is_async;
        // 在进入 spawn_blocking 前提取 owned 数据，避免跨线程借用
        let turn_index = ctx.turn_index;
        let model = ctx.snapshot.model.clone();
        let system_prompt = ctx.snapshot.system_prompt.clone();
        let (run_id, started_at) = run_context_fields(ctx.run);

        Box::pin(async move {
            let result = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let dict = PyDict::new(py);
                    dict.set_item("turn_index", turn_index)?;
                    dict.set_item("model", &model)?;
                    match &system_prompt {
                        Some(s) => dict.set_item("system_prompt", s)?,
                        None => dict.set_item("system_prompt", py.None())?,
                    }
                    dict.set_item("run_id", &run_id)?;
                    dict.set_item("started_at", &started_at)?;
                    call_callback_with_mode(py, &cb, (dict,), is_async)?;
                    Ok::<_, PyErr>(())
                })
            })
            .await;
            if let Err(e) = result {
                tracing::warn!("BeforeTurnHook error: {e}");
            }
        })
    }
}

// ── AfterTurnHook ───────────────────────────────────────────────────────────

/// Python callable 包装为 `AfterTurnHook`。
///
/// callback 签名：`callback(ctx: dict) -> None`
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
pub struct PyAfterTurnHook {
    callback: Arc<Py<PyAny>>,
    is_async: bool,
}

impl PyAfterTurnHook {
    pub fn new(callback: Py<PyAny>) -> Self {
        let is_async = detect_async(&callback);
        Self {
            callback: Arc::new(callback),
            is_async,
        }
    }
}

impl AfterTurnHook for PyAfterTurnHook {
    fn after_turn<'a>(&'a self, ctx: AfterTurnCtx<'a>) -> BoxFuture<'a, ()> {
        let cb = Arc::clone(&self.callback);
        let is_async = self.is_async;
        let turn_index = ctx.turn_index;
        let new_messages = ctx.new_messages.to_vec();
        let (run_id, started_at) = run_context_fields(ctx.run);

        Box::pin(async move {
            let result = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let dict = PyDict::new(py);
                    dict.set_item("turn_index", turn_index)?;
                    dict.set_item("new_messages", agent_messages_to_list(py, &new_messages)?)?;
                    dict.set_item("run_id", &run_id)?;
                    dict.set_item("started_at", &started_at)?;
                    call_callback_with_mode(py, &cb, (dict,), is_async)?;
                    Ok::<_, PyErr>(())
                })
            })
            .await;
            if let Err(e) = result {
                tracing::warn!("AfterTurnHook error: {e}");
            }
        })
    }
}

// ── BeforeRunHook ───────────────────────────────────────────────────────────

/// Python callable 包装为 `BeforeRunHook`。
///
/// callback 签名：`callback(ctx: dict) -> dict | None`
/// 返回 dict 可含 `additional_messages`（list[dict]）和 `system_prompt`（str | None）。
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
pub struct PyBeforeRunHook {
    callback: Arc<Py<PyAny>>,
    is_async: bool,
}

impl PyBeforeRunHook {
    pub fn new(callback: Py<PyAny>) -> Self {
        let is_async = detect_async(&callback);
        Self {
            callback: Arc::new(callback),
            is_async,
        }
    }
}

impl BeforeRunHook for PyBeforeRunHook {
    fn before_run<'a>(
        &'a self,
        ctx: BeforeRunCtx<'a>,
    ) -> BoxFuture<'a, Result<BeforeRunResult, AgentError>> {
        let cb = Arc::clone(&self.callback);
        let is_async = self.is_async;
        let prompt_text = ctx.prompt_text.to_string();
        let initial_messages = ctx.initial_messages.clone();
        let system_prompt = ctx.system_prompt.clone();
        let (run_id, started_at) = run_context_fields(ctx.run);

        Box::pin(async move {
            let result = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let dict = PyDict::new(py);
                    dict.set_item("prompt_text", &prompt_text)?;
                    dict.set_item(
                        "initial_messages",
                        agent_messages_to_list(py, &initial_messages)?,
                    )?;
                    match &system_prompt {
                        Some(s) => dict.set_item("system_prompt", s)?,
                        None => dict.set_item("system_prompt", py.None())?,
                    }
                    dict.set_item("run_id", &run_id)?;
                    dict.set_item("started_at", &started_at)?;
                    let raw = call_callback_with_mode(py, &cb, (dict,), is_async)?;
                    if raw.is_none() {
                        return Ok::<BeforeRunResult, PyErr>(BeforeRunResult {
                            additional_messages: vec![],
                            system_prompt: None,
                        });
                    }
                    let result_dict = raw.cast::<PyDict>()?;
                    let additional_messages: Vec<AgentMessage> = result_dict
                        .get_item("additional_messages")?
                        .map(|v| -> PyResult<Vec<AgentMessage>> {
                            let val = pyobject_to_value(&v)?;
                            serde_json::from_value(val).map_err(|e| {
                                pyo3::exceptions::PyTypeError::new_err(format!(
                                    "additional_messages: {e}"
                                ))
                            })
                        })
                        .transpose()?
                        .unwrap_or_default();
                    let system_prompt: Option<String> = result_dict
                        .get_item("system_prompt")?
                        .and_then(|v| v.extract().ok());
                    Ok::<BeforeRunResult, PyErr>(BeforeRunResult {
                        additional_messages,
                        system_prompt,
                    })
                })
            })
            .await;

            match result {
                Ok(Ok(r)) => Ok(r),
                Ok(Err(e)) => Err(AgentError::Internal(e.to_string())),
                Err(e) => Err(AgentError::Internal(format!("hook join failed: {e}"))),
            }
        })
    }
}

// ── AfterProviderResponseHook ───────────────────────────────────────────────

/// Python callable 包装为 `AfterProviderResponseHook`。
///
/// callback 签名：`callback(info: dict) -> None`
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
pub struct PyAfterProviderResponseHook {
    callback: Arc<Py<PyAny>>,
    is_async: bool,
}

impl PyAfterProviderResponseHook {
    pub fn new(callback: Py<PyAny>) -> Self {
        let is_async = detect_async(&callback);
        Self {
            callback: Arc::new(callback),
            is_async,
        }
    }
}

impl AfterProviderResponseHook for PyAfterProviderResponseHook {
    fn after_response<'a>(&'a self, ctx: AfterProviderResponseCtx<'a>) -> BoxFuture<'a, ()> {
        let cb = Arc::clone(&self.callback);
        let is_async = self.is_async;
        let info = ctx.info;
        let status_code = info.status_code;
        let response_headers = info.response_headers.clone();
        let usage = info.usage.clone();
        let latency_ms = info.latency_ms;
        let model = info.model.clone();
        let provider = info.provider.clone();
        let (run_id, started_at) = run_context_fields(ctx.run);
        let turn_index = ctx.turn_index;

        Box::pin(async move {
            let result = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let dict = PyDict::new(py);
                    match status_code {
                        Some(c) => dict.set_item("status_code", c)?,
                        None => dict.set_item("status_code", py.None())?,
                    }
                    let headers_list = PyList::empty(py);
                    for (k, v) in &response_headers {
                        let pair = PyList::empty(py);
                        pair.append(k)?;
                        pair.append(v)?;
                        headers_list.append(pair)?;
                    }
                    dict.set_item("response_headers", headers_list)?;
                    match usage {
                        Some(u) => {
                            let usage_dict = PyDict::new(py);
                            usage_dict.set_item("input_tokens", u.input_tokens)?;
                            usage_dict.set_item("output_tokens", u.output_tokens)?;
                            usage_dict.set_item("cache_read_tokens", u.cache_read_tokens)?;
                            usage_dict
                                .set_item("cache_creation_tokens", u.cache_creation_tokens)?;
                            usage_dict.set_item("reasoning_tokens", u.reasoning_tokens)?;
                            dict.set_item("usage", usage_dict)?;
                        }
                        None => dict.set_item("usage", py.None())?,
                    }
                    dict.set_item("latency_ms", latency_ms)?;
                    match &model {
                        Some(m) => dict.set_item("model", m)?,
                        None => dict.set_item("model", py.None())?,
                    }
                    match &provider {
                        Some(p) => dict.set_item("provider", p)?,
                        None => dict.set_item("provider", py.None())?,
                    }
                    dict.set_item("run_id", &run_id)?;
                    dict.set_item("started_at", &started_at)?;
                    dict.set_item("turn_index", turn_index)?;
                    call_callback_with_mode(py, &cb, (dict,), is_async)?;
                    Ok::<_, PyErr>(())
                })
            })
            .await;
            if let Err(e) = result {
                tracing::warn!("AfterProviderResponseHook error: {e}");
            }
        })
    }
}

// ── BeforeProviderRequestHook ───────────────────────────────────────────────

/// Python callable 包装为 `BeforeProviderRequestHook`。
///
/// callback 签名：`callback(opts: dict) -> None`
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
pub struct PyBeforeProviderRequestHook {
    callback: Arc<Py<PyAny>>,
    is_async: bool,
}

impl PyBeforeProviderRequestHook {
    pub fn new(callback: Py<PyAny>) -> Self {
        let is_async = detect_async(&callback);
        Self {
            callback: Arc::new(callback),
            is_async,
        }
    }
}

impl BeforeProviderRequestHook for PyBeforeProviderRequestHook {
    fn before_request<'a>(&'a self, ctx: BeforeProviderRequestCtx<'a>) -> BoxFuture<'a, ()> {
        let cb = Arc::clone(&self.callback);
        let is_async = self.is_async;
        let opts = ctx.options;
        let timeout_ms = opts.timeout_ms;
        let max_retries = opts.max_retries;
        let max_retry_delay_ms = opts.max_retry_delay_ms;
        let headers = opts.headers.clone();
        let metadata = opts.metadata.clone();
        let cache_config = opts.cache_config.clone();
        let (run_id, started_at) = run_context_fields(ctx.run);
        let turn_index = ctx.turn_index;

        Box::pin(async move {
            let result = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let dict = PyDict::new(py);
                    match timeout_ms {
                        Some(v) => dict.set_item("timeout_ms", v)?,
                        None => dict.set_item("timeout_ms", py.None())?,
                    }
                    match max_retries {
                        Some(v) => dict.set_item("max_retries", v)?,
                        None => dict.set_item("max_retries", py.None())?,
                    }
                    match max_retry_delay_ms {
                        Some(v) => dict.set_item("max_retry_delay_ms", v)?,
                        None => dict.set_item("max_retry_delay_ms", py.None())?,
                    }
                    let headers_list = PyList::empty(py);
                    for (k, v) in &headers {
                        let pair = PyList::empty(py);
                        pair.append(k)?;
                        pair.append(v)?;
                        headers_list.append(pair)?;
                    }
                    dict.set_item("headers", headers_list)?;
                    dict.set_item("metadata", value_to_pyobject(py, &metadata)?)?;
                    match &cache_config {
                        Some(c) => dict.set_item("cache_config", value_to_pyobject(py, c)?)?,
                        None => dict.set_item("cache_config", py.None())?,
                    }
                    dict.set_item("run_id", &run_id)?;
                    dict.set_item("started_at", &started_at)?;
                    dict.set_item("turn_index", turn_index)?;
                    call_callback_with_mode(py, &cb, (dict,), is_async)?;
                    Ok::<_, PyErr>(())
                })
            })
            .await;
            if let Err(e) = result {
                tracing::warn!("BeforeProviderRequestHook error: {e}");
            }
        })
    }
}

// ── BeforeToolCallHook ──────────────────────────────────────────────────────

/// Python callable 包装为 `BeforeToolCallHook`。
///
/// callback 签名：`callback(ctx: dict) -> str | dict`
/// - 返回 `"allow"` → `BeforeToolCallDecision::Allow`
/// - 返回 `{"action": "modify", "args": <json>}` → `Modify`
/// - 返回 `{"action": "deny", "result": <tool_result_dict>}` → `Deny`
///
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
pub struct PyBeforeToolCallHook {
    callback: Arc<Py<PyAny>>,
    is_async: bool,
}

impl PyBeforeToolCallHook {
    pub fn new(callback: Py<PyAny>) -> Self {
        let is_async = detect_async(&callback);
        Self {
            callback: Arc::new(callback),
            is_async,
        }
    }
}

impl BeforeToolCallHook for PyBeforeToolCallHook {
    fn on_call<'a>(&'a self, ctx: BeforeToolCallCtx<'a>) -> BoxFuture<'a, BeforeToolCallDecision> {
        let cb = Arc::clone(&self.callback);
        let is_async = self.is_async;
        let tool_use_id = ctx.tool_use_id.to_string();
        let tool_name = ctx.tool_name.to_string();
        let args = ctx.args.clone();
        let turn_index = ctx.turn_index;
        let assistant_json: Value =
            serde_json::to_value(ctx.assistant_message).unwrap_or(Value::Null);
        let (run_id, started_at) = run_context_fields(ctx.run);

        Box::pin(async move {
            let result = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let dict = PyDict::new(py);
                    dict.set_item("tool_use_id", &tool_use_id)?;
                    dict.set_item("tool_name", &tool_name)?;
                    dict.set_item("args", value_to_pyobject(py, &args)?)?;
                    dict.set_item("turn_index", turn_index)?;
                    dict.set_item("assistant_message", value_to_pyobject(py, &assistant_json)?)?;
                    dict.set_item("run_id", &run_id)?;
                    dict.set_item("started_at", &started_at)?;
                    let raw = call_callback_with_mode(py, &cb, (dict,), is_async)?;
                    parse_before_tool_call_decision(&raw)
                })
            })
            .await;

            match result {
                Ok(Ok(d)) => d,
                Ok(Err(e)) => {
                    tracing::warn!("BeforeToolCallHook error: {e}");
                    // Fail-closed: deny tool call on callback error
                    BeforeToolCallDecision::Deny(ToolFailure::new(
                        "hook_error",
                        "tool call denied: before_tool_call hook error",
                    ))
                }
                Err(e) => {
                    tracing::warn!("BeforeToolCallHook join error: {e}");
                    BeforeToolCallDecision::Deny(ToolFailure::new(
                        "hook_error",
                        "tool call denied: before_tool_call hook join error",
                    ))
                }
            }
        })
    }
}

/// 解析 Python 返回值为 `BeforeToolCallDecision`。
fn parse_before_tool_call_decision(raw: &Bound<'_, PyAny>) -> PyResult<BeforeToolCallDecision> {
    if let Ok(s) = raw.extract::<String>() {
        return match s.as_str() {
            "allow" => Ok(BeforeToolCallDecision::Allow),
            "modify" => Err(pyo3::exceptions::PyValueError::new_err(
                "action 'modify' requires a dict with 'args'",
            )),
            "deny" => Ok(BeforeToolCallDecision::Deny(ToolFailure::new(
                "denied",
                "tool call denied by before_tool_call hook",
            ))),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown decision: {other}"
            ))),
        };
    }
    let d = raw.cast::<PyDict>()?;
    let action: String = d
        .get_item("action")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing 'action'"))?
        .extract()?;
    match action.as_str() {
        "allow" => Ok(BeforeToolCallDecision::Allow),
        "modify" => {
            let new_args = d
                .get_item("args")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing 'args'"))?;
            let new_args_val = pyobject_to_value(&new_args)?;
            Ok(BeforeToolCallDecision::Modify(new_args_val))
        }
        "deny" => {
            // 从 Python dict 构造 ToolFailure
            let result_val = d
                .get_item("result")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing 'result'"))?;
            let result_json = pyobject_to_value(&result_val)?;
            let tool_failure = parse_tool_failure_from_value(&result_json);
            Ok(BeforeToolCallDecision::Deny(tool_failure))
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown action: {other}"
        ))),
    }
}

/// 从 JSON Value 构造 `ToolFailure`（用于 BeforeToolCall deny）。
///
/// Python 侧 `result` dict 的 `content` 文本块被拼接为 `model_message`，
/// `details` 字段原样传递。
fn parse_tool_failure_from_value(val: &Value) -> ToolFailure {
    let model_message = val
        .get("content")
        .and_then(|c| serde_json::from_value::<Vec<DataBlock>>(c.clone()).ok())
        .map(|blocks| {
            blocks
                .into_iter()
                .filter_map(|b| match b {
                    DataBlock::Text { text, .. } => Some(text),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "tool call denied by before_tool_call hook".to_string());
    let details = val.get("details").cloned().unwrap_or(Value::Null);
    let mut failure = ToolFailure::new("denied", model_message);
    failure.details = details;
    failure
}

// ── AfterToolCallHook ───────────────────────────────────────────────────────

/// Python callable 包装为 `AfterToolCallHook`。
///
/// callback 签名：`callback(ctx: dict) -> str | dict`
/// - 返回 `"passthrough"` → `AfterToolCallDecision::Passthrough`
/// - 返回 `{"action": "patch", "content": [...], ...}` → `Patch`
///
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
pub struct PyAfterToolCallHook {
    callback: Arc<Py<PyAny>>,
    is_async: bool,
}

impl PyAfterToolCallHook {
    pub fn new(callback: Py<PyAny>) -> Self {
        let is_async = detect_async(&callback);
        Self {
            callback: Arc::new(callback),
            is_async,
        }
    }
}

impl AfterToolCallHook for PyAfterToolCallHook {
    fn on_complete<'a>(
        &'a self,
        ctx: AfterToolCallCtx<'a>,
    ) -> BoxFuture<'a, AfterToolCallDecision> {
        let cb = Arc::clone(&self.callback);
        let is_async = self.is_async;
        let tool_use_id = ctx.tool_use_id.to_string();
        let tool_name = ctx.tool_name.to_string();
        let args = ctx.args.clone();
        let turn_index = ctx.turn_index;
        let assistant_json: Value =
            serde_json::to_value(ctx.assistant_message).unwrap_or(Value::Null);
        let result_is_ok = ctx.result.is_ok();
        let result_value = match ctx.result {
            Ok(r) => {
                let content_json: Value =
                    serde_json::to_value(&r.model_content).unwrap_or(Value::Null);
                let mut map = serde_json::Map::new();
                map.insert("content".to_string(), content_json);
                map.insert("details".to_string(), r.details.clone());
                map.insert("terminate".to_string(), Value::Bool(r.terminate));
                Value::Object(map)
            }
            Err(e) => Value::String(e.to_string()),
        };
        let (run_id, started_at) = run_context_fields(ctx.run);

        Box::pin(async move {
            let decision = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let dict = PyDict::new(py);
                    dict.set_item("tool_use_id", &tool_use_id)?;
                    dict.set_item("tool_name", &tool_name)?;
                    dict.set_item("args", value_to_pyobject(py, &args)?)?;
                    dict.set_item("turn_index", turn_index)?;
                    dict.set_item("assistant_message", value_to_pyobject(py, &assistant_json)?)?;
                    dict.set_item("result_ok", result_is_ok)?;
                    dict.set_item("result", value_to_pyobject(py, &result_value)?)?;
                    dict.set_item("run_id", &run_id)?;
                    dict.set_item("started_at", &started_at)?;
                    let raw = call_callback_with_mode(py, &cb, (dict,), is_async)?;
                    parse_after_tool_call_decision(&raw)
                })
            })
            .await;

            match decision {
                Ok(Ok(d)) => d,
                Ok(Err(e)) => {
                    tracing::warn!("AfterToolCallHook error: {e}");
                    AfterToolCallDecision::Passthrough
                }
                Err(e) => {
                    tracing::warn!("AfterToolCallHook join error: {e}");
                    AfterToolCallDecision::Passthrough
                }
            }
        })
    }
}

/// 解析 Python 返回值为 `AfterToolCallDecision`。
fn parse_after_tool_call_decision(raw: &Bound<'_, PyAny>) -> PyResult<AfterToolCallDecision> {
    if let Ok(s) = raw.extract::<String>() {
        return match s.as_str() {
            "passthrough" => Ok(AfterToolCallDecision::Passthrough),
            "patch" => Err(pyo3::exceptions::PyValueError::new_err(
                "action 'patch' requires a dict",
            )),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown decision: {other}"
            ))),
        };
    }
    let d = raw.cast::<PyDict>()?;
    let action: String = d
        .get_item("action")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing 'action'"))?
        .extract()?;
    match action.as_str() {
        "passthrough" => Ok(AfterToolCallDecision::Passthrough),
        "patch" => {
            let content: Vec<DataBlock> = d
                .get_item("content")?
                .and_then(|v| {
                    let json = pyobject_to_value(&v).ok()?;
                    serde_json::from_value(json).ok()
                })
                .unwrap_or_default();
            let details: Value = d
                .get_item("details")?
                .and_then(|v| pyobject_to_value(&v).ok())
                .unwrap_or(Value::Null);
            let terminate: bool = d
                .get_item("terminate")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(false);
            Ok(AfterToolCallDecision::Replace(Ok(ToolResult::full(
                content, details, terminate,
            ))))
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown action: {other}"
        ))),
    }
}

// ── ShouldStopHook ──────────────────────────────────────────────────────────

/// Python callable 包装为 `ShouldStopHook`。
///
/// callback 签名：`callback(ctx: dict) -> bool`
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
pub struct PyShouldStopHook {
    callback: Arc<Py<PyAny>>,
    is_async: bool,
}

impl PyShouldStopHook {
    pub fn new(callback: Py<PyAny>) -> Self {
        let is_async = detect_async(&callback);
        Self {
            callback: Arc::new(callback),
            is_async,
        }
    }
}

impl ShouldStopHook for PyShouldStopHook {
    fn should_stop<'a>(&'a self, ctx: ShouldStopCtx<'a>) -> BoxFuture<'a, bool> {
        let cb = Arc::clone(&self.callback);
        let is_async = self.is_async;
        let turn_index = ctx.turn_index;
        let stop_reason = stop_reason_to_str(ctx.stop_reason);
        let last_assistant_json: Value =
            serde_json::to_value(ctx.last_assistant).unwrap_or(Value::Null);
        let (run_id, started_at) = run_context_fields(ctx.run);

        Box::pin(async move {
            let result = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let dict = PyDict::new(py);
                    dict.set_item("turn_index", turn_index)?;
                    dict.set_item("stop_reason", stop_reason)?;
                    dict.set_item(
                        "last_assistant",
                        value_to_pyobject(py, &last_assistant_json)?,
                    )?;
                    dict.set_item("run_id", &run_id)?;
                    dict.set_item("started_at", &started_at)?;
                    let raw = call_callback_with_mode(py, &cb, (dict,), is_async)?;
                    let should: bool = raw.extract()?;
                    Ok::<_, PyErr>(should)
                })
            })
            .await;

            match result {
                Ok(Ok(b)) => b,
                Ok(Err(e)) => {
                    tracing::warn!("ShouldStopHook error: {e}");
                    true // fail-safe: stop on error
                }
                Err(e) => {
                    tracing::warn!("ShouldStopHook join error: {e}");
                    true
                }
            }
        })
    }
}

// ── BeforeCompactHook ───────────────────────────────────────────────────────

/// Python callable 包装为 `BeforeCompactHook`。
/// callback 签名：`callback(ctx: dict) -> str | dict`
/// - 返回 `"proceed"` → `BeforeCompactDecision::Proceed`
/// - 返回 `"skip"` → `BeforeCompactDecision::Skip`
/// - 返回 `"compact"` → `BeforeCompactDecision::Compact`
/// - 返回 `{"action": "override", "summary": <msg_dict>, "first_kept_entry": <str>}` → `Override`
///   `first_kept_entry` 必须是 `ctx["entry_ids"]` 中的一个值。
///   可选字段 `tokens_before` (默认 `ctx["estimated_tokens"]`) 和 `tokens_after` (默认 0)。
///
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
pub struct PyBeforeCompactHook {
    callback: Arc<Py<PyAny>>,
    is_async: bool,
}

impl PyBeforeCompactHook {
    pub fn new(callback: Py<PyAny>) -> Self {
        let is_async = detect_async(&callback);
        Self {
            callback: Arc::new(callback),
            is_async,
        }
    }
}

impl BeforeCompactHook for PyBeforeCompactHook {
    fn before_compact<'a>(
        &'a self,
        ctx: BeforeCompactCtx<'a>,
    ) -> BoxFuture<'a, BeforeCompactDecision> {
        let cb = Arc::clone(&self.callback);
        let is_async = self.is_async;
        let estimated_tokens = ctx.estimated_tokens;
        let messages = ctx.messages.to_vec();
        let entry_ids: Vec<String> = ctx.entry_ids.iter().map(|id| id.to_string()).collect();
        let context_window = ctx.context_window;
        let reserve_tokens = ctx.reserve_tokens;
        let keep_recent_tokens = ctx.keep_recent_tokens;

        Box::pin(async move {
            let result = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let dict = PyDict::new(py);
                    dict.set_item("estimated_tokens", estimated_tokens)?;
                    dict.set_item("messages", agent_messages_to_list(py, &messages)?)?;
                    dict.set_item("entry_ids", entry_ids)?;
                    dict.set_item("context_window", context_window)?;
                    dict.set_item("reserve_tokens", reserve_tokens)?;
                    dict.set_item("keep_recent_tokens", keep_recent_tokens)?;
                    let raw = call_callback_with_mode(py, &cb, (dict,), is_async)?;
                    parse_before_compact_decision(&raw, estimated_tokens)
                })
            })
            .await;

            match result {
                Ok(Ok(d)) => d,
                Ok(Err(e)) => {
                    tracing::warn!("BeforeCompactHook error: {e}");
                    BeforeCompactDecision::Proceed
                }
                Err(e) => {
                    tracing::warn!("BeforeCompactHook join error: {e}");
                    BeforeCompactDecision::Proceed
                }
            }
        })
    }
}

/// 解析 Python 返回值为 `BeforeCompactDecision`。
///
/// `estimated_tokens` is used to fill `tokens_before` for Override decisions
/// when the callback does not provide it explicitly.
fn parse_before_compact_decision(
    raw: &Bound<'_, PyAny>,
    estimated_tokens: usize,
) -> PyResult<BeforeCompactDecision> {
    if let Ok(s) = raw.extract::<String>() {
        return match s.as_str() {
            "proceed" => Ok(BeforeCompactDecision::Proceed),
            "skip" => Ok(BeforeCompactDecision::Skip),
            "compact" => Ok(BeforeCompactDecision::Compact),
            "override" => Err(pyo3::exceptions::PyValueError::new_err(
                "action 'override' requires a dict with 'summary'",
            )),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown decision: {other}"
            ))),
        };
    }
    let d = raw.cast::<PyDict>()?;
    let action: String = d
        .get_item("action")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing 'action'"))?
        .extract()?;
    match action.as_str() {
        "proceed" => Ok(BeforeCompactDecision::Proceed),
        "skip" => Ok(BeforeCompactDecision::Skip),
        "compact" => Ok(BeforeCompactDecision::Compact),
        "override" => {
            let summary_val = d
                .get_item("summary")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing 'summary'"))?;
            let summary_json = pyobject_to_value(&summary_val)?;
            let summary_message: AgentMessage =
                serde_json::from_value(summary_json).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("invalid summary: {e}"))
                })?;
            let first_kept_str: String = d
                .get_item("first_kept_entry")?
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "missing 'first_kept_entry' — must be one of ctx['entry_ids']",
                    )
                })?
                .extract()?;
            let first_kept_entry =
                llm_harness_types::EntryId::from_str(&first_kept_str).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid first_kept_entry: {e}"
                    ))
                })?;
            Ok(BeforeCompactDecision::Override(CompactionResult {
                summary_message,
                first_kept_entry,
                tokens_before: d
                    .get_item("tokens_before")?
                    .map(|v| v.extract::<usize>())
                    .transpose()?
                    .unwrap_or(estimated_tokens),
                tokens_after: d
                    .get_item("tokens_after")?
                    .map(|v| v.extract::<usize>())
                    .transpose()?
                    .unwrap_or(0),
            }))
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown action: {other}"
        ))),
    }
}
// ── TransformContextHook ────────────────────────────────────────────────────

/// Python callable 包装为 `TransformContextHook`。
///
/// callback 签名：`callback(ctx: dict) -> dict`
/// 返回 dict 须含 `system_prompt`（str | None）和 `messages`（list[dict]）。
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
pub struct PyTransformContextHook {
    callback: Arc<Py<PyAny>>,
    is_async: bool,
}

impl PyTransformContextHook {
    pub fn new(callback: Py<PyAny>) -> Self {
        let is_async = detect_async(&callback);
        Self {
            callback: Arc::new(callback),
            is_async,
        }
    }
}

impl TransformContextHook for PyTransformContextHook {
    fn transform<'a>(
        &'a self,
        ctx: TransformContextCtx<'a>,
    ) -> BoxFuture<'a, Result<AgentContext, AgentError>> {
        let cb = Arc::clone(&self.callback);
        let is_async = self.is_async;
        let system_prompt = ctx.context.system_prompt.clone();
        let messages = ctx.context.messages.clone();
        let (run_id, started_at) = run_context_fields(ctx.run);

        Box::pin(async move {
            let result = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let dict = agent_context_to_dict(
                        py,
                        &AgentContext {
                            system_prompt,
                            messages,
                        },
                    )?;
                    let dict_bound = dict.bind(py).cast::<PyDict>()?;
                    dict_bound.set_item("run_id", &run_id)?;
                    dict_bound.set_item("started_at", &started_at)?;
                    let raw = call_callback_with_mode(py, &cb, (dict,), is_async)?;
                    // 期望返回 dict（转换后的 context）
                    let result_dict = raw.cast::<PyDict>()?;
                    let new_system_prompt: Option<String> = result_dict
                        .get_item("system_prompt")?
                        .and_then(|v| v.extract().ok());
                    let new_messages_val: Value = result_dict
                        .get_item("messages")?
                        .map(|v| pyobject_to_value(&v))
                        .transpose()?
                        .unwrap_or(Value::Null);
                    let new_messages: Vec<AgentMessage> = serde_json::from_value(new_messages_val)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                    Ok::<AgentContext, PyErr>(AgentContext {
                        system_prompt: new_system_prompt,
                        messages: new_messages,
                    })
                })
            })
            .await;

            match result {
                Ok(Ok(ctx)) => Ok(ctx),
                Ok(Err(e)) => Err(AgentError::Internal(e.to_string())),
                Err(e) => Err(AgentError::Internal(format!("hook join failed: {e}"))),
            }
        })
    }
}

// ── PrepareNextTurnHook ─────────────────────────────────────────────────────

/// Python callable 包装为 `PrepareNextTurnHook`。
///
/// callback 签名：`callback(ctx: dict) -> dict | None`
/// 返回 dict 可含 `model`（str）、`thinking_level`（str）、`temperature`（float | None）、`active_tools`（list[str]）。返回 `None` 表示沿用当前值。
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
pub struct PyPrepareNextTurnHook {
    callback: Arc<Py<PyAny>>,
    is_async: bool,
}

impl PyPrepareNextTurnHook {
    pub fn new(callback: Py<PyAny>) -> Self {
        let is_async = detect_async(&callback);
        Self {
            callback: Arc::new(callback),
            is_async,
        }
    }
}

/// 将字符串解析为 `ThinkingLevel`。
fn parse_thinking_level(s: &str) -> Option<llm_harness_types::ThinkingLevel> {
    use llm_harness_types::ThinkingLevel;
    match s.to_lowercase().as_str() {
        "off" | "none" => Some(ThinkingLevel::Off),
        "minimal" => Some(ThinkingLevel::Minimal),
        "low" => Some(ThinkingLevel::Low),
        "medium" => Some(ThinkingLevel::Medium),
        "high" => Some(ThinkingLevel::High),
        "xhigh" => Some(ThinkingLevel::XHigh),
        _ => None,
    }
}

impl PrepareNextTurnHook for PyPrepareNextTurnHook {
    fn prepare<'a>(
        &'a self,
        ctx: PrepareNextTurnCtx<'a>,
    ) -> BoxFuture<'a, Result<NextTurnDirective, AgentError>> {
        let cb = Arc::clone(&self.callback);
        let is_async = self.is_async;
        let turn_index = ctx.turn_index;
        let last_message_json: Value =
            serde_json::to_value(ctx.last_message).unwrap_or(Value::Null);
        // last_tool_results: &[(String, Result<ToolResult, ToolError>)]
        // Clone owned data to avoid cross-thread borrows; ToolResult is Clone
        // but not Serialize, so we convert to Python dict inside the GIL scope.
        let last_tool_results: Vec<(String, Result<ToolResult, String>)> = ctx
            .last_tool_results
            .iter()
            .map(|(id, r)| {
                (
                    id.clone(),
                    match r {
                        Ok(tr) => Ok(tr.clone()),
                        Err(e) => Err(e.to_string()),
                    },
                )
            })
            .collect();
        let (run_id, started_at) = run_context_fields(ctx.run);

        Box::pin(async move {
            let result = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let dict = PyDict::new(py);
                    dict.set_item("turn_index", turn_index)?;
                    dict.set_item("last_message", value_to_pyobject(py, &last_message_json)?)?;
                    let results_list = PyList::empty(py);
                    for (id, result_val) in &last_tool_results {
                        let pair = PyDict::new(py);
                        pair.set_item("tool_use_id", id)?;
                        match result_val {
                            Ok(tr) => {
                                pair.set_item("result", tool_result_to_dict(py, tr)?)?;
                            }
                            Err(err_str) => {
                                pair.set_item("error", err_str)?;
                            }
                        }
                        results_list.append(pair)?;
                    }
                    dict.set_item("last_tool_results", results_list)?;
                    dict.set_item("run_id", &run_id)?;
                    dict.set_item("started_at", &started_at)?;
                    let raw = call_callback_with_mode(py, &cb, (dict,), is_async)?;
                    // 返回 None 表示沿用当前值
                    if raw.is_none() {
                        return Ok::<NextTurnDirective, PyErr>(NextTurnDirective {
                            context: None,
                            model: None,
                            thinking_level: None,
                            temperature: None,
                            tools: None,
                            active_tools: None,
                            response_format: None,
                        });
                    }
                    let d = raw.cast::<PyDict>()?;
                    // 解析 model
                    let model: Option<String> = d.get_item("model")?.and_then(|v| v.extract().ok());
                    // 解析 thinking_level
                    let thinking_level = d.get_item("thinking_level")?.and_then(|v| {
                        v.extract::<String>()
                            .ok()
                            .and_then(|s| parse_thinking_level(&s))
                    });
                    // 解析 temperature：None 表示清除，Some(float) 表示设置
                    let temperature: Option<Option<f32>> = match d.get_item("temperature")? {
                        None => None,                         // 字段缺失 → 沿用当前值
                        Some(v) if v.is_none() => Some(None), // 显式 None → 清除
                        Some(v) => {
                            let temp = v.extract::<f32>().map_err(|_| {
                                let type_name = v
                                    .get_type()
                                    .name()
                                    .map(|n| n.to_string())
                                    .unwrap_or_else(|_| "unknown".to_string());
                                pyo3::exceptions::PyTypeError::new_err(format!(
                                    "temperature must be a float, got {type_name}"
                                ))
                            })?;
                            Some(Some(temp))
                        }
                    };
                    // 解析 active_tools
                    let active_tools: Option<std::collections::HashSet<String>> =
                        d.get_item("active_tools")?.and_then(|v| {
                            v.extract::<Vec<String>>()
                                .ok()
                                .map(|names| names.into_iter().collect())
                        });
                    Ok(NextTurnDirective {
                        context: None,
                        model,
                        thinking_level,
                        temperature,
                        tools: None,
                        active_tools,
                        response_format: None,
                    })
                })
            })
            .await;

            match result {
                Ok(Ok(d)) => Ok(d),
                Ok(Err(e)) => Err(AgentError::Internal(e.to_string())),
                Err(e) => Err(AgentError::Internal(format!("hook join failed: {e}"))),
            }
        })
    }
}

// ── 共享 callback 调用工具函数 ───────────────────────────────────────────────

/// 检测 Python callable 是否为 coroutine function（`async def`）。
pub(crate) fn detect_async(callback: &Py<PyAny>) -> bool {
    Python::attach(|py| {
        let inspect = pyo3::types::PyModule::import(py, "inspect")?;
        let is_coro: bool = inspect
            .call_method1("iscoroutinefunction", (callback.bind(py),))?
            .extract()?;
        Ok::<_, PyErr>(is_coro)
    })
    .unwrap_or(false)
}

/// 与 `call_callback` 相同，但使用预先检测的 `is_async` 标志，避免每次调用
/// 都执行 `inspect.iscoroutinefunction`。
pub(crate) fn call_callback_with_mode<'py>(
    py: Python<'py>,
    cb: &Py<PyAny>,
    args: impl pyo3::call::PyCallArgs<'py>,
    is_async: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let bound = cb.bind(py);
    let result = bound.call1(args)?;
    if is_async {
        // Schedule the coroutine on the user's main event loop when
        // possible (issue #13), falling back to asyncio.run().
        Ok(crate::core::pyloop::run_coro(py, &result)?)
    } else {
        Ok(result)
    }
}

// ── FinalAnswerValidator ─────────────────────────────────────────────────────

/// Python callable wrapper for `FinalAnswerValidator`.
///
/// callback signature: `callback(ctx: dict) -> None | str | dict`
/// - None → accept the answer
/// - str → reject with code="rejected", message=<returned str>
/// - dict → reject with code=dict["code"], message=dict["message"]
pub struct PyFinalAnswerValidatorWrapper {
    callback: Arc<Py<PyAny>>,
    is_async: bool,
}

impl PyFinalAnswerValidatorWrapper {
    pub fn new(callback: Py<PyAny>) -> Self {
        let is_async = detect_async(&callback);
        Self {
            callback: Arc::new(callback),
            is_async,
        }
    }
}

impl FinalAnswerValidator for PyFinalAnswerValidatorWrapper {
    fn validate<'a>(
        &'a self,
        ctx: FinalAnswerValidationCtx<'a>,
    ) -> BoxFuture<'a, Result<(), FinalAnswerValidationError>> {
        let cb = Arc::clone(&self.callback);
        let is_async = self.is_async;
        // Serialize candidate to owned data (avoids borrowing across threads).
        let candidate_json: Value = serde_json::to_value(ctx.candidate).unwrap_or(Value::Null);
        let turn_index = ctx.turn_index;
        let (run_id, started_at) = run_context_fields(ctx.run);

        Box::pin(async move {
            let result = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let dict = PyDict::new(py);
                    dict.set_item("candidate", value_to_pyobject(py, &candidate_json)?)?;
                    dict.set_item("turn_index", turn_index)?;
                    dict.set_item("run_id", &run_id)?;
                    dict.set_item("started_at", &started_at)?;
                    let result = call_callback_with_mode(py, &cb, (dict,), is_async)?;
                    Ok::<_, PyErr>(result.unbind())
                })
            })
            .await;

            match result {
                Ok(Ok(obj)) => {
                    // Callback succeeded — interpret return value.
                    Python::attach(|py| -> Result<(), FinalAnswerValidationError> {
                        if obj.is_none(py) {
                            Ok(())
                        } else if let Ok(s) = obj.extract::<String>(py) {
                            Err(FinalAnswerValidationError::new("rejected", s))
                        } else {
                            // Try dict with "code" and "message"
                            let dict = obj.bind(py).cast::<PyDict>().map_err(|e| {
                                FinalAnswerValidationError::new("validator_error", e.to_string())
                            })?;
                            let code: String = dict
                                .get_item("code")
                                .map_err(|e: PyErr| {
                                    FinalAnswerValidationError::new(
                                        "validator_error",
                                        e.to_string(),
                                    )
                                })?
                                .ok_or_else(|| {
                                    FinalAnswerValidationError::new(
                                        "validator_error",
                                        "validator dict must have 'code' key",
                                    )
                                })?
                                .extract()
                                .map_err(|e: PyErr| {
                                    FinalAnswerValidationError::new(
                                        "validator_error",
                                        e.to_string(),
                                    )
                                })?;
                            let message: String = dict
                                .get_item("message")
                                .map_err(|e: PyErr| {
                                    FinalAnswerValidationError::new(
                                        "validator_error",
                                        e.to_string(),
                                    )
                                })?
                                .ok_or_else(|| {
                                    FinalAnswerValidationError::new(
                                        "validator_error",
                                        "validator dict must have 'message' key",
                                    )
                                })?
                                .extract()
                                .map_err(|e: PyErr| {
                                    FinalAnswerValidationError::new(
                                        "validator_error",
                                        e.to_string(),
                                    )
                                })?;
                            Err(FinalAnswerValidationError::new(code, message))
                        }
                    })
                }
                Ok(Err(e)) => {
                    // Python exception in callback → treat as rejection.
                    Err(FinalAnswerValidationError::new(
                        "validator_error",
                        e.to_string(),
                    ))
                }
                Err(e) => {
                    // JoinError (panic) → treat as rejection.
                    Err(FinalAnswerValidationError::new(
                        "validator_error",
                        e.to_string(),
                    ))
                }
            }
        })
    }
}

// ── AfterRunHook ────────────────────────────────────────────────────────────

/// Python callable 包装为 `AfterRunHook`。
///
/// callback 签名：`callback() -> None`
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
pub struct PyAfterRunHook {
    callback: Arc<Py<PyAny>>,
    is_async: bool,
}

impl PyAfterRunHook {
    pub fn new(callback: Py<PyAny>) -> Self {
        let is_async = detect_async(&callback);
        Self {
            callback: Arc::new(callback),
            is_async,
        }
    }
}

impl AfterRunHook for PyAfterRunHook {
    fn after_run(&self) -> BoxFuture<'_, ()> {
        let cb = Arc::clone(&self.callback);
        let is_async = self.is_async;

        Box::pin(async move {
            let result = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    call_callback_with_mode(py, &cb, (), is_async)?;
                    Ok::<_, PyErr>(())
                })
            })
            .await;
            if let Err(e) = result {
                tracing::warn!("AfterRunHook error: {e}");
            }
        })
    }
}

// ── OnAbortHook ─────────────────────────────────────────────────────────────

/// Python callable 包装为 `OnAbortHook`。
///
/// callback 签名：`callback() -> None`
/// 同步调用：在 abort 路径上直接执行，不经过 `spawn_blocking`。
/// 若 callback 为 `async def`，其 coroutine 将通过 `pyloop::run_coro`
/// 在当前线程上阻塞执行（与 AfterRunHook 的 async 调度一致）。
pub struct PyOnAbortHook {
    callback: Arc<Py<PyAny>>,
    is_async: bool,
}

impl PyOnAbortHook {
    pub fn new(callback: Py<PyAny>) -> Self {
        let is_async = detect_async(&callback);
        Self {
            callback: Arc::new(callback),
            is_async,
        }
    }
}

impl OnAbortHook for PyOnAbortHook {
    fn on_abort(&self) {
        let cb = Arc::clone(&self.callback);
        let is_async = self.is_async;
        let result = Python::attach(|py| {
            call_callback_with_mode(py, &cb, (), is_async)?;
            Ok::<_, PyErr>(())
        });
        if let Err(e) = result {
            tracing::warn!("OnAbortHook error: {e}");
        }
    }
}

// ── ProviderErrorHook ───────────────────────────────────────────────────────

/// Python callable 包装为 `ProviderErrorHook`。
///
/// callback 签名：`callback(ctx: dict) -> str | None`
/// 返回 `"retry"`（同轮重试）/ `"surface"` 或 `None`（默认原样上抛）。
/// ctx 中的 `context` / `new_messages` 是只读快照——Python 回调不回写历史；
/// 需要修复历史时使用 Rust 侧 preset hook（如 vision_degrade）。
pub struct PyProviderErrorHook {
    callback: Arc<Py<PyAny>>,
    is_async: bool,
}

impl PyProviderErrorHook {
    pub fn new(callback: Py<PyAny>) -> Self {
        let is_async = detect_async(&callback);
        Self {
            callback: Arc::new(callback),
            is_async,
        }
    }
}

impl ProviderErrorHook for PyProviderErrorHook {
    fn on_provider_error<'a>(&'a self, ctx: ProviderErrorCtx<'a>) -> BoxFuture<'a, ErrorDirective> {
        let cb = Arc::clone(&self.callback);
        let is_async = self.is_async;
        let (run_id, started_at) = run_context_fields(ctx.run);
        let turn_index = ctx.turn_index;
        let error = ctx.error.to_string();
        // 只读快照：在进入 async block 前提取 owned 数据（ctx 是 &mut 借用）。
        let system_prompt = ctx.context.system_prompt.clone();
        let messages = ctx.context.messages.clone();
        let new_messages = ctx.new_messages.to_vec();

        Box::pin(async move {
            let result = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let dict = PyDict::new(py);
                    dict.set_item("run_id", &run_id)?;
                    dict.set_item("started_at", &started_at)?;
                    dict.set_item("turn_index", turn_index)?;
                    dict.set_item("error", &error)?;
                    dict.set_item(
                        "context",
                        agent_context_to_dict(
                            py,
                            &AgentContext {
                                system_prompt,
                                messages,
                            },
                        )?,
                    )?;
                    dict.set_item("new_messages", agent_messages_to_list(py, &new_messages)?)?;
                    let raw = call_callback_with_mode(py, &cb, (dict,), is_async)?;
                    // "retry" → Retry；其余（"surface"、None、其他）→ Surface。
                    let directive = raw
                        .extract::<String>()
                        .ok()
                        .is_some_and(|s| s.eq_ignore_ascii_case("retry"));
                    Ok::<bool, PyErr>(directive)
                })
            })
            .await;
            match result {
                Ok(Ok(true)) => ErrorDirective::Retry,
                _ => ErrorDirective::Surface,
            }
        })
    }
}

// ── Python 包装类 ───────────────────────────────────────────────────────────

/// 所有 hook trait 的枚举包装。
#[derive(Clone)]
pub enum HookKind {
    BeforeTurn(Arc<dyn BeforeTurnHook>),
    AfterTurn(Arc<dyn AfterTurnHook>),
    BeforeRun(Arc<dyn BeforeRunHook>),
    AfterProviderResponse(Arc<dyn AfterProviderResponseHook>),
    BeforeProviderRequest(Arc<dyn BeforeProviderRequestHook>),
    BeforeToolCall(Arc<dyn BeforeToolCallHook>),
    AfterToolCall(Arc<dyn AfterToolCallHook>),
    ShouldStop(Arc<dyn ShouldStopHook>),
    BeforeCompact(Arc<dyn BeforeCompactHook>),
    TransformContext(Arc<dyn TransformContextHook>),
    PrepareNextTurn(Arc<dyn PrepareNextTurnHook>),
    FinalAnswerValidator(Arc<dyn FinalAnswerValidator>),
    AfterRun(Arc<dyn AfterRunHook>),
    OnAbort(Arc<dyn OnAbortHook>),
    ProviderError(Arc<dyn ProviderErrorHook>),
}

/// 持有任意 hook trait 对象的不透明 Python 包装。
#[pyclass(name = "Hook")]
pub struct PyHookWrapper {
    pub kind: HookKind,
}

impl PyHookWrapper {
    /// 若包装的是 `ShouldStopHook`，返回其 `Arc`；否则返回 `PyTypeError`。
    ///
    /// 调用方（`HarnessBuilder::should_stop_hook`）应仅在 `ShouldStop` variant 上调用，
    /// 但若用户误传其他 kind 的 hook，这里返回 Python 异常而非 panic，避免无 traceback 的进程崩溃。
    pub fn as_should_stop_hook(&self) -> PyResult<Arc<dyn ShouldStopHook>> {
        match &self.kind {
            HookKind::ShouldStop(h) => Ok(h.clone()),
            other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "expected a ShouldStop hook, got {}",
                other.kind_name()
            ))),
        }
    }

    /// 提取 `AfterTurnHook`，类型不匹配时返回 Python 异常。
    pub fn as_after_turn_hook(&self) -> PyResult<Arc<dyn AfterTurnHook>> {
        match &self.kind {
            HookKind::AfterTurn(h) => Ok(h.clone()),
            other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "expected an AfterTurn hook, got {}",
                other.kind_name()
            ))),
        }
    }

    /// 提取 `ProviderErrorHook`，类型不匹配时返回 Python 异常。
    pub fn as_provider_error_hook(&self) -> PyResult<Arc<dyn ProviderErrorHook>> {
        match &self.kind {
            HookKind::ProviderError(h) => Ok(h.clone()),
            other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "expected a ProviderError hook, got {}",
                other.kind_name()
            ))),
        }
    }

    /// 将内部 hook 按 kind 推入对应的 `HarnessHooks` 向量。
    pub fn push_into(&self, hooks: &mut llm_harness_agent::HarnessHooks) {
        match &self.kind {
            HookKind::BeforeTurn(h) => hooks.before_turn.push(h.clone()),
            HookKind::AfterTurn(h) => hooks.after_turn.push(h.clone()),
            HookKind::BeforeRun(h) => hooks.before_run.push(h.clone()),
            HookKind::AfterProviderResponse(h) => hooks.after_provider_response.push(h.clone()),
            HookKind::BeforeProviderRequest(h) => hooks.before_provider_request.push(h.clone()),
            HookKind::BeforeToolCall(h) => hooks.before_tool_call.push(h.clone()),
            HookKind::AfterToolCall(h) => hooks.after_tool_call.push(h.clone()),
            HookKind::ShouldStop(h) => hooks.should_stop.push(h.clone()),
            HookKind::BeforeCompact(h) => hooks.before_compact.push(h.clone()),
            HookKind::TransformContext(h) => hooks.transform_context.push(h.clone()),
            HookKind::PrepareNextTurn(h) => hooks.prepare_next_turn.push(h.clone()),
            HookKind::FinalAnswerValidator(h) => hooks.final_answer_validator.push(h.clone()),
            HookKind::AfterRun(h) => hooks.after_run.push(h.clone()),
            HookKind::OnAbort(h) => hooks.on_abort.push(h.clone()),
            HookKind::ProviderError(h) => hooks.provider_error.push(h.clone()),
        }
    }
}

impl HookKind {
    /// 返回 hook kind 的可读名称，用于诊断信息。
    pub fn kind_name(&self) -> &'static str {
        match self {
            HookKind::BeforeTurn(_) => "BeforeTurn",
            HookKind::AfterTurn(_) => "AfterTurn",
            HookKind::BeforeRun(_) => "BeforeRun",
            HookKind::AfterProviderResponse(_) => "AfterProviderResponse",
            HookKind::BeforeProviderRequest(_) => "BeforeProviderRequest",
            HookKind::BeforeToolCall(_) => "BeforeToolCall",
            HookKind::AfterToolCall(_) => "AfterToolCall",
            HookKind::ShouldStop(_) => "ShouldStop",
            HookKind::BeforeCompact(_) => "BeforeCompact",
            HookKind::TransformContext(_) => "TransformContext",
            HookKind::PrepareNextTurn(_) => "PrepareNextTurn",
            HookKind::FinalAnswerValidator(_) => "FinalAnswerValidator",
            HookKind::AfterRun(_) => "AfterRun",
            HookKind::OnAbort(_) => "OnAbort",
            HookKind::ProviderError(_) => "ProviderError",
        }
    }
}
