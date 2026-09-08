//! `PyAgentHarness` — 包装 `AgentHarness`。
//!
//! 暴露 `prompt()`、`events()`、`message_count()`、`phase()`。
//! 使用与 `PyAgent` 相同的 GIL+tokio 模式：`py.detach()` + `runtime.block_on()`。
//!
//! # 事件类型说明
//!
//! `AgentHarness::subscribe()` 返回 `broadcast::Receiver<Arc<AgentHarnessEvent>>`，
//! 与 `Agent::subscribe()` 返回的 `Receiver<Arc<AgentEvent>>` 不同。
//! `AgentHarnessEvent` 是 `AgentEvent` 的超集：`Agent(AgentEvent)` 变体包装底层
//! agent 事件，其余变体是 harness 级事件（phase change、compaction 等）。
//! 本模块实现独立的 `PyHarnessEventIterator` 处理此类型。

use std::collections::HashSet;
use std::str::FromStr;
use std::sync::Arc;

use llm_harness_agent::{AgentHarness, AgentHarnessEvent};
use llm_harness_knowledge::{KnowledgeAccessContext, KnowledgeScope, PrincipalRef};
use llm_harness_mcp::builder::McpAgentHarness;
use llm_harness_types::{HarnessPhase, RunRequest, ThinkingLevel, Tool};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::sync::broadcast;

use crate::core::pyagent::runtime;
use crate::core::pyinspector::PyInspector;
use crate::runtime::pyworkflow::cost_aggregate_to_dict;
use crate::shared::event_stream::agent_event_to_dict;
use crate::shared::pyerror::harness_error_to_pyerr;
use crate::shared::value_conv::value_to_pyobject;

/// 将 `HarnessPhase` 转为字符串标识。
fn phase_str(phase: HarnessPhase) -> &'static str {
    match phase {
        HarnessPhase::Idle => "idle",
        HarnessPhase::Turning => "turning",
        HarnessPhase::Compacting => "compacting",
        HarnessPhase::Branching => "branching",
        _ => "unknown",
    }
}

/// 从 text + 可选 Python 附件构造 `UserMessage`。
///
/// content 顺序：`[Text(text), *attachment_blocks]`。
pub(crate) fn user_message_from_py(
    text: &str,
    attachments: Option<Vec<Bound<'_, crate::core::pytool::PyAttachment>>>,
) -> llm_harness_types::AgentMessage {
    use llm_harness_types::content::ContentBlock;
    let mut content = Vec::with_capacity(1 + attachments.as_ref().map_or(0, Vec::len));
    content.push(ContentBlock::Text {
        text: text.to_string(),
    });
    if let Some(list) = attachments {
        for att in &list {
            content.push(att.borrow().block().clone());
        }
    }
    llm_harness_types::AgentMessage::User(llm_harness_types::UserMessage {
        content,
        timestamp: chrono::Utc::now(),
    })
}

/// 从 text + 可选 Python 附件构造 `RunRequest`。
pub(crate) fn run_request_from_py(
    text: &str,
    attachments: Option<Vec<Bound<'_, crate::core::pytool::PyAttachment>>>,
) -> RunRequest {
    RunRequest::new(vec![user_message_from_py(text, attachments)])
}

/// 将 `AgentHarnessEvent` 转换为 Python dict。
///
/// `Agent(AgentEvent)` 变体委托 `agent_event_to_dict`，其余变体生成
/// 带 `"type"` 字段的简单 dict。
fn harness_event_to_dict(py: Python<'_>, event: &AgentHarnessEvent) -> PyResult<Py<PyAny>> {
    match event {
        AgentHarnessEvent::Agent(agent_event) => agent_event_to_dict(py, agent_event),
        AgentHarnessEvent::PhaseChange { from, to } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "phase_change")?;
            dict.set_item("from", phase_str(*from))?;
            dict.set_item("to", phase_str(*to))?;
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::ModelUpdate { from, to } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "model_update")?;
            dict.set_item("from", from.clone())?;
            dict.set_item("to", to.clone())?;
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::ThinkingLevelUpdate { from, to } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "thinking_level_update")?;
            dict.set_item("from", format!("{from:?}"))?;
            dict.set_item("to", format!("{to:?}"))?;
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::ToolsUpdate { added, removed } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "tools_update")?;
            dict.set_item("added", added.clone())?;
            dict.set_item("removed", removed.clone())?;
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::ActiveToolsUpdate { active } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "active_tools_update")?;
            match active {
                Some(tools) => dict.set_item("active", tools.clone())?,
                None => dict.set_item("active", py.None())?,
            }
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::ResourcesUpdate {
            skills,
            templates,
            diagnostics,
        } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "resources_update")?;
            dict.set_item("skills", *skills)?;
            dict.set_item("templates", *templates)?;
            let diag_strs: Vec<String> = diagnostics.iter().map(|d| format!("{d:?}")).collect();
            dict.set_item("diagnostics", diag_strs)?;
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::SessionInfoUpdate { name } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "session_info_update")?;
            dict.set_item("name", name.clone())?;
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::CompactionStart { estimated_tokens } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "compaction_start")?;
            dict.set_item("estimated_tokens", *estimated_tokens)?;
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::CompactionEnd { stats, error } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "compaction_end")?;
            match stats {
                Some(s) => {
                    dict.set_item("stats", format!("{s:?}"))?;
                }
                None => dict.set_item("stats", py.None())?,
            }
            match error {
                Some(e) => dict.set_item("error", e.clone())?,
                None => dict.set_item("error", py.None())?,
            }
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::QueueUpdate {
            steer_len,
            follow_up_len,
        } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "queue_update")?;
            dict.set_item("steer_len", *steer_len)?;
            dict.set_item("follow_up_len", *follow_up_len)?;
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::SavePoint { entries_flushed } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "savepoint")?;
            dict.set_item("entries_flushed", *entries_flushed)?;
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::BranchForked {
            from,
            new_leaf,
            label,
        } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "branch_forked")?;
            dict.set_item("from", format!("{from:?}"))?;
            dict.set_item("new_leaf", format!("{new_leaf:?}"))?;
            match label {
                Some(l) => dict.set_item("label", l.clone())?,
                None => dict.set_item("label", py.None())?,
            }
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::BranchSwitched { from, to } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "branch_switched")?;
            dict.set_item("from", format!("{from:?}"))?;
            dict.set_item("to", format!("{to:?}"))?;
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::BranchDeleted { leaf } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "branch_deleted")?;
            dict.set_item("leaf", format!("{leaf:?}"))?;
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::BranchSummarized { leaf, summary } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "branch_summarized")?;
            dict.set_item("leaf", format!("{leaf:?}"))?;
            dict.set_item("summary", summary.clone())?;
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::ToolCallStart {
            tool_use_id,
            tool_name,
            args,
        } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "harness_tool_call_start")?;
            dict.set_item("tool_use_id", tool_use_id.clone())?;
            dict.set_item("tool_name", tool_name.clone())?;
            dict.set_item("args", value_to_pyobject(py, args)?)?;
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::ToolCallEnd {
            tool_use_id,
            tool_name,
            result,
        } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "harness_tool_call_end")?;
            dict.set_item("tool_use_id", tool_use_id.clone())?;
            dict.set_item("tool_name", tool_name.clone())?;
            let result_dict = PyDict::new(py);
            result_dict.set_item("details", value_to_pyobject(py, &result.details)?)?;
            result_dict.set_item("is_error", result.is_error)?;
            dict.set_item("result", result_dict)?;
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::Settled => {
            let dict = PyDict::new(py);
            dict.set_item("type", "settled")?;
            Ok(dict.into_any().unbind())
        }
        AgentHarnessEvent::Aborted => {
            let dict = PyDict::new(py);
            dict.set_item("type", "aborted")?;
            Ok(dict.into_any().unbind())
        }
    }
}

/// Python 迭代器，包装 `broadcast::Receiver<Arc<AgentHarnessEvent>>`。
///
/// 与 `PyEventIterator` 对称，但处理 harness 级事件类型。
#[pyclass(name = "HarnessEventIterator")]
pub struct PyHarnessEventIterator {
    rx: Option<broadcast::Receiver<Arc<AgentHarnessEvent>>>,
    timeout_ms: u64,
    max_consecutive_timeouts: u32,
    consecutive_timeouts: u32,
    handle: tokio::runtime::Handle,
}

impl PyHarnessEventIterator {
    pub fn new(
        rx: broadcast::Receiver<Arc<AgentHarnessEvent>>,
        timeout_ms: u64,
        max_consecutive_timeouts: u32,
        handle: tokio::runtime::Handle,
    ) -> Self {
        Self {
            rx: Some(rx),
            timeout_ms,
            max_consecutive_timeouts,
            consecutive_timeouts: 0,
            handle,
        }
    }
}

#[pymethods]
impl PyHarnessEventIterator {
    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// 阻塞等待下一个事件，channel 关闭时返回 None。超时不终止，发出 timeout 事件后继续。
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let rx = match &mut self.rx {
            Some(rx) => rx,
            None => return Ok(None),
        };

        let timeout = std::time::Duration::from_millis(self.timeout_ms);
        let handle = self.handle.clone();

        let recv_result = crate::shared::pyerror::detach_catch_panic(py, move || {
            handle.block_on(async move { tokio::time::timeout(timeout, rx.recv()).await })
        })?;

        match recv_result {
            Ok(Ok(event)) => {
                self.consecutive_timeouts = 0;
                let dict = harness_event_to_dict(py, &event)?;
                Ok(Some(dict))
            }
            Ok(Err(broadcast::error::RecvError::Lagged(n))) => {
                let warning = PyDict::new(py);
                warning.set_item("type", "lagged")?;
                warning.set_item("skipped", n)?;
                Ok(Some(warning.into_any().unbind()))
            }
            Ok(Err(broadcast::error::RecvError::Closed)) => Ok(None),
            // timeout elapsed：达到 max_consecutive_timeouts 则终止，否则发出 timeout 事件继续
            Err(_) => {
                self.consecutive_timeouts += 1;
                if self.consecutive_timeouts >= self.max_consecutive_timeouts {
                    Ok(None)
                } else {
                    let timeout_event = PyDict::new(py);
                    timeout_event.set_item("type", "timeout")?;
                    Ok(Some(timeout_event.into_any().unbind()))
                }
            }
        }
    }
}

/// 将字符串解析为 `ThinkingLevel`。
///
/// 接受: "off", "minimal", "low", "medium", "high", "xhigh", 或 "budget:<tokens>"。
pub(crate) fn parse_thinking_level(s: &str) -> PyResult<ThinkingLevel> {
    match s.to_lowercase().as_str() {
        "off" => Ok(ThinkingLevel::Off),
        "minimal" => Ok(ThinkingLevel::Minimal),
        "low" => Ok(ThinkingLevel::Low),
        "medium" => Ok(ThinkingLevel::Medium),
        "high" => Ok(ThinkingLevel::High),
        "xhigh" => Ok(ThinkingLevel::XHigh),
        s if s.starts_with("budget:") => {
            let n: u32 = s[7..].parse().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid budget value: {e}"))
            })?;
            Ok(ThinkingLevel::Budget(n))
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid thinking level: '{s}'. Use: off, minimal, low, medium, high, xhigh, or budget:<tokens>"
        ))),
    }
}

/// 从 broadcast channel 接收事件，同时每 200ms 检查 Python 信号（SIGINT）。
///
/// 返回 `Ok(Some(event))` 表示收到事件；`Ok(None)` 表示超时或 channel 已关闭；
/// `Err(PyErr)` 表示信号中断（`KeyboardInterrupt`）。
///
/// 用 `tokio::select!` 同时等待 `rx.recv()` 和定时信号检查，确保即使
/// `timeout` 很长（如 30s），Ctrl+C 也能在 ~200ms 内被打断。
async fn recv_event_with_signal_check(
    rx: &mut broadcast::Receiver<Arc<AgentHarnessEvent>>,
    timeout: std::time::Duration,
) -> PyResult<Option<Arc<AgentHarnessEvent>>> {
    let signal_interval = std::time::Duration::from_millis(200);
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Ok(None);
        }
        let wait = remaining.min(signal_interval);
        tokio::select! {
            biased;
            result = rx.recv() => match result {
                Ok(event) => return Ok(Some(event)),
                Err(_) => return Ok(None),
            },
            _ = tokio::time::sleep(wait) => {
                Python::attach(|py| py.check_signals())?;
            }
        }
    }
}

/// 内部 harness 引用：普通或 MCP。
///
/// `Mcp` 变体通过 `Deref` 链（`Arc<McpAgentHarness>` → `McpAgentHarness` →
/// `AgentHarness`）暴露所有 `AgentHarness` 方法。
#[derive(Clone)]
pub(crate) enum HarnessRef {
    Base(Arc<AgentHarness>),
    Mcp(Arc<McpAgentHarness>),
}

impl std::ops::Deref for HarnessRef {
    type Target = AgentHarness;

    fn deref(&self) -> &Self::Target {
        match self {
            HarnessRef::Base(h) => h,
            HarnessRef::Mcp(h) => h,
        }
    }
}

/// Trusted single-user access context used as the Senza SDK default.
///
/// The knowledge/memory/recall tools fail closed unless the run carries a
/// [`KnowledgeAccessContext`] extension. The Senza SDK is a single-user trusted
/// layer (its plugins use `AllowAllAuthorizer`), so it defaults every run to
/// this context; applications may override it via
/// `HarnessBuilder.knowledge_access(scope, principal)`.
fn default_knowledge_access() -> KnowledgeAccessContext {
    KnowledgeAccessContext::new(
        KnowledgeScope::new("senza"),
        PrincipalRef::new("python-sdk", "user"),
    )
}

impl HarnessRef {
    /// Run a request, injecting the given knowledge access context.
    async fn run_with_extension(
        &self,
        request: RunRequest,
        access: KnowledgeAccessContext,
    ) -> Result<(), llm_harness_types::HarnessError> {
        let request = request.with_extension(access);
        match self {
            HarnessRef::Base(h) => h.run(request).await,
            HarnessRef::Mcp(h) => h.run(request).await,
        }
    }
}

/// Python 侧的 `AgentHarness` 包装类。
#[pyclass(name = "AgentHarness")]
pub struct PyAgentHarness {
    pub(crate) harness: HarnessRef,
    /// Per-harness knowledge access context; `None` → `default_knowledge_access()`.
    knowledge_access: Option<KnowledgeAccessContext>,
}

impl PyAgentHarness {
    /// 创建普通 harness 包装。
    pub fn new_base(harness: Arc<AgentHarness>) -> Self {
        Self {
            harness: HarnessRef::Base(harness),
            knowledge_access: None,
        }
    }

    /// 创建普通 harness 包装，携带显式的知识访问上下文。
    pub fn new_base_with_access(
        harness: Arc<AgentHarness>,
        knowledge_access: KnowledgeAccessContext,
    ) -> Self {
        Self {
            harness: HarnessRef::Base(harness),
            knowledge_access: Some(knowledge_access),
        }
    }

    /// 创建 MCP harness 包装。
    pub fn new_mcp(mcp_harness: Arc<McpAgentHarness>) -> Self {
        Self {
            harness: HarnessRef::Mcp(mcp_harness),
            knowledge_access: None,
        }
    }

    /// 创建 MCP harness 包装，携带显式的知识访问上下文。
    pub fn new_mcp_with_access(
        mcp_harness: Arc<McpAgentHarness>,
        knowledge_access: KnowledgeAccessContext,
    ) -> Self {
        Self {
            harness: HarnessRef::Mcp(mcp_harness),
            knowledge_access: Some(knowledge_access),
        }
    }

    /// The effective knowledge access context for this harness.
    fn effective_access(&self) -> KnowledgeAccessContext {
        self.knowledge_access
            .clone()
            .unwrap_or_else(default_knowledge_access)
    }
}

#[pymethods]
impl PyAgentHarness {
    /// 同步执行 prompt，阻塞直到完成。
    ///
    /// `AgentHarness::prompt()` 返回 `Result<(), HarnessError>`——
    /// 不返回回复文本（回复通过事件流或 `build_context()` 获取）。
    /// 成功时返回 None。
    ///
    /// **警告**: `prompt()` 完全阻塞直到 LLM 完成，所有事件在这期间发出。
    /// 在 `prompt()` 之后再调用 `collect_until_settled()` 会拿到空列表——
    /// broadcast channel 不回放已发送的事件。如需同时发送 prompt 并收集事件，
    /// 请使用 `prompt_and_collect()`。
    #[pyo3(signature = (text, attachments=None))]
    fn prompt(
        &self,
        py: Python<'_>,
        text: &str,
        attachments: Option<Vec<Bound<'_, crate::core::pytool::PyAttachment>>>,
    ) -> PyResult<()> {
        let harness = self.harness.clone();
        let request = run_request_from_py(text, attachments);
        let access = self.effective_access();
        let rt = runtime(py);
        crate::shared::pyerror::block_on_with_signal_check(
            py,
            rt,
            async move {
                harness
                    .run_with_extension(request, access)
                    .await
                    .map_err(harness_error_to_pyerr)
            },
            200,
        )?;
        Ok(())
    }

    /// 获取当前会话中的消息数量。
    ///
    /// 通过 `build_context()` 获取消息列表后计数。需释放 GIL 执行 async 调用。
    fn message_count(&self, py: Python<'_>) -> PyResult<usize> {
        let harness = self.harness.clone();
        let rt = runtime(py);
        let ctx = crate::shared::pyerror::detach_catch_panic_result(py, move || {
            rt.block_on(async move { harness.build_context().await })
        })?;
        Ok(ctx.messages.len())
    }

    /// 返回当前会话的完整消息列表（user / assistant / tool_result 等）。
    ///
    /// 每条消息是一个 dict，包含 `role` 字段和对应的消息内容。
    /// 可用于检查对话历史、调试 prompt、提取 LLM 回复文本。
    fn get_messages(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let harness = self.harness.clone();
        let rt = runtime(py);
        let ctx = crate::shared::pyerror::detach_catch_panic_result(py, move || {
            rt.block_on(async move { harness.build_context().await })
        })?;
        let mut messages = Vec::new();
        for msg in &ctx.messages {
            let json = serde_json::to_value(msg)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let py_msg = value_to_pyobject(py, &json)?;
            messages.push(py_msg);
        }
        Ok(messages)
    }

    /// 返回最近一条 assistant 消息的文本内容。
    ///
    /// 从 `build_context()` 的消息列表中反向查找最后一条 `role=assistant` 消息，
    /// 提取其 `content` 中的 Text 块拼接返回。
    /// 若没有 assistant 消息则返回空字符串。
    fn last_response(&self, py: Python<'_>) -> PyResult<String> {
        let harness = self.harness.clone();
        let rt = runtime(py);
        let ctx = crate::shared::pyerror::detach_catch_panic_result(py, move || {
            rt.block_on(async move { harness.build_context().await })
        })?;
        {
            for msg in ctx.messages.iter().rev() {
                let json = serde_json::to_value(msg)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                if json.get("role").and_then(|r| r.as_str()) == Some("assistant")
                    && let Some(content) = json.get("content").and_then(|c| c.as_array())
                {
                    let text: String = content
                        .iter()
                        .filter_map(|block| {
                            if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                                block.get("text").and_then(|t| t.as_str()).map(String::from)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("");
                    return Ok(text);
                }
            }
            Ok(String::new())
        }
    }

    /// 获取当前 phase（"idle" / "turning" / "compacting" / "branching"）。
    fn phase(&self) -> &'static str {
        phase_str(self.harness.state().phase)
    }

    /// 返回 harness 事件迭代器。`timeout_ms` 为单次 `__next__` 等待超时（毫秒）。
    #[pyo3(signature = (timeout_ms=5000, max_consecutive_timeouts=1))]
    fn events(
        &self,
        py: Python<'_>,
        timeout_ms: u64,
        max_consecutive_timeouts: u32,
    ) -> PyResult<Py<PyHarnessEventIterator>> {
        let rx = self.harness.subscribe();
        let handle = runtime(py).handle().clone();
        Py::new(
            py,
            PyHarnessEventIterator::new(rx, timeout_ms, max_consecutive_timeouts, handle),
        )
    }

    /// 取消当前正在运行的 prompt（如果有）。不阻塞。
    ///
    /// 发出取消信号后，正在执行的 `prompt()` 会在最近的取消点抛出
    /// `RuntimeError`。可以安全地在非 prompt 期间调用（无操作）。
    fn abort(&self) {
        self.harness.abort();
    }

    /// 收集事件直到 harness 进入 settled 状态（phase=idle 且无更多事件）。
    ///
    /// 便捷方法，等价于：
    /// ```python
    /// events = []
    /// for event in harness.events(timeout_ms=timeout_ms):
    ///     events.append(event)
    ///     if event.get("type") in ("settled", "aborted"):
    ///         break
    /// ```
    ///
    /// 返回事件列表。若超时未收到 settled/aborted，返回已收集的事件。
    ///
    /// **警告**: 仅在 `subscribe()` 之后、`prompt()` 之前调用才有效。
    /// 如果先调用 `prompt()`（阻塞）再调用本方法，所有事件已发出，
    /// 将拿到空列表。请改用 `prompt_and_collect()` 一步完成。
    #[pyo3(signature = (timeout_ms=30000))]
    fn collect_until_settled(&self, py: Python<'_>, timeout_ms: u64) -> PyResult<Vec<Py<PyAny>>> {
        let rx = self.harness.subscribe();
        let handle = runtime(py).handle().clone();
        let timeout = std::time::Duration::from_millis(timeout_ms);

        let events: Vec<Py<PyAny>> =
            crate::shared::pyerror::detach_catch_panic_pyresult(py, move || {
                let mut events = Vec::new();
                let mut rx = rx;
                loop {
                    let recv = handle.block_on(recv_event_with_signal_check(&mut rx, timeout));
                    match recv {
                        Ok(Some(event)) => {
                            let is_settled = matches!(
                                &*event,
                                AgentHarnessEvent::Settled | AgentHarnessEvent::Aborted
                            );
                            Python::attach(|py| {
                                if let Ok(dict) = harness_event_to_dict(py, &event) {
                                    events.push(dict);
                                }
                            });
                            if is_settled {
                                break;
                            }
                        }
                        Ok(None) => break,
                        Err(e) => return Err(e),
                    }
                }
                Ok(events)
            })?;

        Ok(events)
    }

    /// 一步到位：发送 prompt 并收集事件，直到 settled/aborted。
    ///
    /// 等价于先订阅事件再在后台执行 `prompt()`，避免了 `prompt()` 阻塞
    /// 完毕后 `collect_until_settled()` 拿不到事件的时序问题。
    ///
    /// 若 LLM 调用失败（网络错误、API key 无效等），抛出 `RuntimeError`。
    /// 若超时未收到 settled/aborted，中止 LLM 任务并返回已收集的事件。
    #[pyo3(signature = (text, timeout_ms=30000, attachments=None))]
    fn prompt_and_collect(
        &self,
        py: Python<'_>,
        text: &str,
        timeout_ms: u64,
        attachments: Option<Vec<Bound<'_, crate::core::pytool::PyAttachment>>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let harness = self.harness.clone();
        let request = run_request_from_py(text, attachments);
        let rx = self.harness.subscribe();
        let access = self.effective_access();
        let handle = runtime(py).handle().clone();
        let timeout = std::time::Duration::from_millis(timeout_ms);

        let events: PyResult<Vec<Py<PyAny>>> =
            crate::shared::pyerror::detach_catch_panic_pyresult(py, move || {
                // Spawn prompt as a background task so we can collect events concurrently.
                let prompt_harness = harness.clone();
                let prompt_task = handle
                    .spawn(async move { prompt_harness.run_with_extension(request, access).await });

                let mut events = Vec::new();
                let mut rx = rx;
                let mut got_terminal = false;
                loop {
                    let recv = handle.block_on(recv_event_with_signal_check(&mut rx, timeout));
                    match recv {
                        Ok(Some(event)) => {
                            let is_settled = matches!(
                                &*event,
                                AgentHarnessEvent::Settled | AgentHarnessEvent::Aborted
                            );
                            Python::attach(|py| {
                                if let Ok(dict) = harness_event_to_dict(py, &event) {
                                    events.push(dict);
                                }
                            });
                            if is_settled {
                                got_terminal = true;
                                break;
                            }
                        }
                        Ok(None) => break,
                        Err(e) => {
                            // Abort the prompt task before returning so we
                            // don't leak a background LLM call.
                            prompt_task.abort();
                            return Err(e);
                        }
                    }
                }

                // If the event loop timed out without a terminal event, abort
                // the prompt task so we don't block indefinitely waiting for
                // the LLM to finish.
                if !got_terminal {
                    prompt_task.abort();
                }

                // Join the prompt task. If it completed naturally (after
                // Settled/Aborted) this returns immediately. If we aborted
                // it, we get a cancelled JoinError and return partial events.
                match handle.block_on(prompt_task) {
                    Ok(Ok(())) => {
                        // prompt() returns Ok even when the LLM call fails —
                        // the error is stored in state.error_message. Check
                        // it here so callers see failures instead of empty
                        // event lists. (issue #58)
                        if let Some(msg) = &harness.state().error_message {
                            Err(pyo3::exceptions::PyRuntimeError::new_err(msg.clone()))
                        } else {
                            Ok(events)
                        }
                    }
                    Ok(Err(e)) => Err(harness_error_to_pyerr(e)),
                    Err(join_err) if join_err.is_cancelled() => Ok(events),
                    Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
                }
            });

        events
    }

    // ── Dynamic configuration ──────────────────────────────────────────────

    /// 动态切换模型。可选提供 `context_window` 和 `max_tokens`。
    #[pyo3(signature = (model, context_window=None, max_tokens=None))]
    fn set_model(
        &self,
        py: Python<'_>,
        model: &str,
        context_window: Option<u32>,
        max_tokens: Option<u32>,
    ) -> PyResult<()> {
        let harness = self.harness.clone();
        let model = model.to_string();
        let info = Some(llm_harness_agent::ModelInfo {
            context_window: context_window.unwrap_or(0),
            max_tokens: max_tokens.unwrap_or(0),
        });
        let rt = runtime(py);
        crate::shared::pyerror::detach_catch_panic_result(py, move || {
            rt.block_on(async move { harness.set_model(model, info).await })
        })
    }

    /// 设置或清除系统提示。下一轮生效。
    #[pyo3(signature = (prompt=None))]
    fn set_system_prompt(&self, prompt: Option<&str>) {
        self.harness
            .set_system_prompt(prompt.map(|s| s.to_string()));
    }

    /// 设置采样温度。`None` 重置为 provider 默认值。下一轮生效。
    #[pyo3(signature = (temperature=None))]
    fn set_temperature(&self, py: Python<'_>, temperature: Option<f32>) -> PyResult<()> {
        let harness = self.harness.clone();
        let rt = runtime(py);
        crate::shared::pyerror::detach_catch_panic_result(py, move || {
            rt.block_on(async move { harness.set_temperature(temperature).await })
        })
    }

    /// 设置 thinking level。接受: "off", "minimal", "low", "medium", "high", "xhigh", "budget:<tokens>"。
    fn set_thinking_level(&self, py: Python<'_>, level: &str) -> PyResult<()> {
        let level = parse_thinking_level(level)?;
        let harness = self.harness.clone();
        let rt = runtime(py);
        crate::shared::pyerror::detach_catch_panic_result(py, move || {
            rt.block_on(async move { harness.set_thinking_level(level).await })
        })
    }

    /// 设置每次 provider 调用的最大输出 token 数。下一轮生效。
    fn set_max_tokens(&self, max_tokens: u32) {
        self.harness.set_max_tokens(max_tokens);
    }

    /// 替换已注册的工具列表。
    fn set_tools(&self, py: Python<'_>, tools: &Bound<'_, PyList>) -> PyResult<()> {
        let mut tool_vec: Vec<Arc<dyn Tool>> = Vec::with_capacity(tools.len());
        for item in tools.iter() {
            let wrapper = item.cast::<crate::core::pytool::PyToolWrapper>()?;
            let t: Arc<dyn Tool> = wrapper.borrow().tool.clone();
            tool_vec.push(t);
        }
        let harness = self.harness.clone();
        let rt = runtime(py);
        crate::shared::pyerror::detach_catch_panic_result(py, move || {
            rt.block_on(async move { harness.set_tools(tool_vec).await })
        })
    }

    // ── Steering / Follow-up ────────────────────────────────────────────────

    /// 向正在运行的 harness 插入一条 steering 消息。
    /// 当前 turn 完成后，steering 消息会被加入下一轮的 context。
    #[pyo3(signature = (text, attachments=None))]
    fn steer(
        &self,
        text: &str,
        attachments: Option<Vec<Bound<'_, crate::core::pytool::PyAttachment>>>,
    ) {
        let msg = user_message_from_py(text, attachments);
        // 与 runtime `steer(text)` 契约一致：Idle 阶段 channel 关闭，消息
        // 静默丢失（错误被忽略）。需要检测丢失时先检查 phase() 再调用。
        let _ = self.harness.steer_message(msg);
    }

    /// 向正在运行的 harness 插入一条 follow-up 消息。
    /// Follow-up 消息在当前 turn 结束后立即触发新一轮。
    #[pyo3(signature = (text, attachments=None))]
    fn follow_up(
        &self,
        text: &str,
        attachments: Option<Vec<Bound<'_, crate::core::pytool::PyAttachment>>>,
    ) {
        let msg = user_message_from_py(text, attachments);
        // runtime 契约：Idle 阶段 follow_up channel 关闭，消息静默丢失。
        let _ = self.harness.follow_up_message(msg);
    }

    /// 继续上一次运行（不附加新消息）。
    fn continue_run(&self, py: Python<'_>) -> PyResult<()> {
        let harness = self.harness.clone();
        let rt = runtime(py);
        crate::shared::pyerror::block_on_with_signal_check(
            py,
            rt,
            async move { harness.continue_run().await.map_err(harness_error_to_pyerr) },
            200,
        )
    }

    /// 发送下一条 user 消息并继续运行。
    #[pyo3(signature = (text, attachments=None))]
    fn next_turn(
        &self,
        text: &str,
        attachments: Option<Vec<Bound<'_, crate::core::pytool::PyAttachment>>>,
    ) {
        let msg = user_message_from_py(text, attachments);
        self.harness.next_turn_message(msg);
    }
    // ── Compaction ──────────────────────────────────────────────────────────

    /// Manually trigger compaction. Returns a dict with:
    /// `tokens_before`, `tokens_after`, `compressed_entries`.
    ///
    /// Uses the compaction prompt and query set via builder or runtime setters.
    /// Not subject to the circuit breaker — manual compaction always runs.
    fn compact(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let harness = self.harness.clone();
        let rt = runtime(py);
        let stats = crate::shared::pyerror::block_on_with_signal_check(
            py,
            rt,
            async move { harness.compact().await.map_err(harness_error_to_pyerr) },
            200,
        )?;
        let dict = PyDict::new(py);
        dict.set_item("tokens_before", stats.tokens_before)?;
        dict.set_item("tokens_after", stats.tokens_after)?;
        dict.set_item("compressed_entries", stats.compressed_entries)?;
        Ok(dict.into_any().unbind())
    }

    /// 返回当前会话的元数据 dict（id/name/created_at/updated_at/model/...）。
    fn session_metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let harness = self.harness.clone();
        let rt = runtime(py);
        let md = crate::shared::pyerror::block_on_with_signal_check(
            py,
            rt,
            async move {
                harness
                    .session_metadata()
                    .await
                    .map_err(harness_error_to_pyerr)
            },
            200,
        )?;
        let json = serde_json::to_value(&md)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        value_to_pyobject(py, &json)
    }

    // ── Cost / Usage ────────────────────────────────────────────────────────

    /// 返回累计 token/成本统计，dict。
    fn usage(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cost = self.harness.usage();
        cost_aggregate_to_dict(py, &cost)
    }

    /// 返回共享 UsageLedger 的成本快照，dict。
    ///
    /// 与 `usage()` 返回相同的 `CostAggregate` 快照——当 harness 通过
    /// `HarnessBuilder.usage_ledger()` 注入了 caller-owned ledger 时，
    /// 此方法返回的是该共享 ledger 的当前累计值。
    #[pyo3(text_signature = "($self)")]
    fn usage_ledger(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cost = self.harness.usage();
        cost_aggregate_to_dict(py, &cost)
    }

    /// 重置累计成本统计。
    fn reset_usage(&self) {
        self.harness.reset_usage();
    }

    // ── Waiting ─────────────────────────────────────────────────────────────

    /// 阻塞直到 harness 进入 idle 状态。
    fn wait_for_idle(&self, py: Python<'_>) -> PyResult<()> {
        let harness = self.harness.clone();
        let rt = runtime(py);
        crate::shared::pyerror::block_on_with_signal_check(
            py,
            rt,
            async move {
                harness.wait_for_idle().await;
                Ok(())
            },
            200,
        )
    }

    /// 阻塞直到 harness 进入 settled 状态（idle 且无待处理事件）。
    fn wait_for_settled(&self, py: Python<'_>) -> PyResult<()> {
        let harness = self.harness.clone();
        let rt = runtime(py);
        crate::shared::pyerror::block_on_with_signal_check(
            py,
            rt,
            async move {
                harness.wait_for_settled().await;
                Ok(())
            },
            200,
        )
    }

    // ── Queue management ────────────────────────────────────────────────────

    /// Drain the steering queue. Only callable when Idle.
    fn clear_steering_queue(&self) -> PyResult<()> {
        self.harness
            .clear_steering_queue()
            .map_err(harness_error_to_pyerr)
    }

    /// Drain the follow-up queue. Only callable when Idle.
    fn clear_follow_up_queue(&self) -> PyResult<()> {
        self.harness
            .clear_follow_up_queue()
            .map_err(harness_error_to_pyerr)
    }

    /// Drain all queues (steer, follow-up, next_turn). Only callable when Idle.
    fn clear_all_queues(&self) -> PyResult<()> {
        self.harness
            .clear_all_queues()
            .map_err(harness_error_to_pyerr)
    }

    /// Returns True if any queue is non-empty.
    fn has_queued_messages(&self) -> bool {
        self.harness.has_queued_messages()
    }

    // ── Active tools ────────────────────────────────────────────────────────

    /// Limit the active tool subset for the next turn. `None` enables all tools.
    #[pyo3(signature = (tools=None))]
    fn set_active_tools(&self, py: Python<'_>, tools: Option<Vec<String>>) -> PyResult<()> {
        let harness = self.harness.clone();
        let active = tools.map(|v| v.into_iter().collect::<HashSet<String>>());
        let rt = runtime(py);
        crate::shared::pyerror::detach_catch_panic_result(py, move || {
            rt.block_on(async move { harness.set_active_tools(active).await })
        })
    }

    // ── Session / Branch management ─────────────────────────────────────────

    /// Fork the session at a given entry. Returns the new leaf entry ID.
    #[pyo3(signature = (from_entry, label=None))]
    fn fork_branch(
        &self,
        py: Python<'_>,
        from_entry: &str,
        label: Option<String>,
    ) -> PyResult<String> {
        let harness = self.harness.clone();
        let entry = llm_harness_types::EntryId::from_str(from_entry)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let rt = runtime(py);
        crate::shared::pyerror::detach_catch_panic_result(py, move || {
            rt.block_on(async move { harness.fork_branch(entry, label).await })
        })
        .map(|id| id.to_string())
    }

    /// Switch the active cursor to a target entry.
    fn navigate_tree(&self, py: Python<'_>, target: &str) -> PyResult<()> {
        let harness = self.harness.clone();
        let entry = llm_harness_types::EntryId::from_str(target)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let rt = runtime(py);
        crate::shared::pyerror::detach_catch_panic_result(py, move || {
            rt.block_on(async move { harness.navigate_tree(entry).await })
        })
    }

    /// List all branches (leaves) in the session.
    fn list_branches(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let harness = self.harness.clone();
        let rt = runtime(py);
        let branches = crate::shared::pyerror::detach_catch_panic_result(py, move || {
            rt.block_on(async move { harness.list_branches().await })
        })?;
        let mut result = Vec::new();
        for b in branches {
            let dict = PyDict::new(py);
            dict.set_item("leaf_id", b.leaf_id.to_string())?;
            dict.set_item("label", b.label)?;
            dict.set_item("message_count", b.message_count)?;
            dict.set_item("last_activity", b.last_activity.to_rfc3339())?;
            dict.set_item("summary", b.summary)?;
            result.push(dict.into_any().unbind());
        }
        Ok(result)
    }

    /// Return all entries on the active cursor's path (root-first).
    fn read_active_path(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let harness = self.harness.clone();
        let rt = runtime(py);
        let entries = crate::shared::pyerror::detach_catch_panic_result(py, move || {
            rt.block_on(async move { harness.read_active_path().await })
        })?;
        Ok(session_entries_to_list(py, &entries))
    }

    /// Return all session entries (every node in the tree).
    fn read_all_entries(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let harness = self.harness.clone();
        let rt = runtime(py);
        let entries = crate::shared::pyerror::detach_catch_panic_result(py, move || {
            rt.block_on(async move { harness.read_all_entries().await })
        })?;
        Ok(session_entries_to_list(py, &entries))
    }

    /// Delete the branch ending at a leaf entry.
    fn delete_branch(&self, py: Python<'_>, leaf: &str) -> PyResult<()> {
        let harness = self.harness.clone();
        let entry = llm_harness_types::EntryId::from_str(leaf)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let rt = runtime(py);
        crate::shared::pyerror::detach_catch_panic_result(py, move || {
            rt.block_on(async move { harness.delete_branch(entry).await })
        })
    }

    /// Generate an AI summary for the branch ending at a leaf entry.
    fn generate_branch_summary(&self, py: Python<'_>, leaf: &str) -> PyResult<Py<PyAny>> {
        let harness = self.harness.clone();
        let entry = llm_harness_types::EntryId::from_str(leaf)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let rt = runtime(py);
        let summary = crate::shared::pyerror::detach_catch_panic_result(py, move || {
            rt.block_on(async move { harness.generate_branch_summary(entry).await })
        })?;
        let dict = PyDict::new(py);
        dict.set_item("leaf_id", summary.leaf_id.to_string())?;
        dict.set_item("from_entry", summary.from_entry.to_string())?;
        dict.set_item("summary", &summary.summary)?;
        dict.set_item("token_count", summary.token_count)?;
        Ok(dict.into_any().unbind())
    }

    // ── Context manager ─────────────────────────────────────────────────────

    /// 返回 harness 状态摘要。
    fn __repr__(&self) -> String {
        let state = self.harness.state();
        format!(
            "AgentHarness(model={}, phase={})",
            state.model,
            phase_str(state.phase)
        )
    }

    /// Context manager entry: returns self for use in `with` statements.
    ///
    /// ```python
    /// with HarnessBuilder("gpt-4o").provider("gpt-*", provider).build() as h:
    ///     h.prompt("Hello!")
    ///     for event in h.collect_until_settled():
    ///         ...
    /// ```
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// 关闭 MCP runtime（仅 MCP harness 有效）。
    ///
    /// 断开所有 MCP server 连接。对普通 harness 无操作。
    fn shutdown(&self, py: Python<'_>) -> PyResult<()> {
        if let HarnessRef::Mcp(mcp_harness) = &self.harness {
            let mcp_harness = mcp_harness.clone();
            let rt = runtime(py);
            crate::shared::pyerror::detach_catch_panic_result(py, move || {
                rt.block_on(async move {
                    mcp_harness
                        .shutdown()
                        .await
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
                })
            })?;
        }
        Ok(())
    }
    /// Mount the Agent Inspector Web API on the given port.
    ///
    /// Returns an `Inspector` handle. Dropping it shuts down the server.
    /// Call this after `build()`, before `prompt()`.
    #[pyo3(signature = (port = 8080))]
    fn mount_inspector(&self, py: Python<'_>, port: u16) -> PyResult<Py<PyInspector>> {
        let harness: Arc<AgentHarness> = match &self.harness {
            HarnessRef::Base(h) => h.clone(),
            HarnessRef::Mcp(_) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "mount_inspector is not supported on MCP harnesses",
                ));
            }
        };
        let rt = runtime(py);
        let inspector = crate::shared::pyerror::detach_catch_panic_result(py, move || {
            rt.block_on(async move { PyInspector::mount(harness, port).await })
        })?;
        Py::new(py, inspector)
    }

    /// Context manager exit: aborts any in-progress prompt and returns.
    /// Ensures clean shutdown when used with `with`.
    fn __exit__(
        &mut self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        self.harness.abort();
        Ok(false) // don't suppress exceptions
    }
}

/// Convert a Vec<SessionEntry> to a list of Python dicts.
fn session_entries_to_list(
    py: Python<'_>,
    entries: &[llm_harness_agent::session::SessionEntry],
) -> Vec<Py<PyAny>> {
    let mut result = Vec::new();
    for entry in entries {
        let dict = PyDict::new(py);
        dict.set_item("id", entry.id.to_string()).ok();
        dict.set_item("parent_id", entry.parent_id.map(|id| id.to_string()))
            .ok();
        dict.set_item("timestamp", entry.timestamp.to_rfc3339())
            .ok();
        // Serialize payload to JSON then convert to Python
        if let Ok(payload_val) = serde_json::to_value(&entry.payload) {
            dict.set_item(
                "payload",
                crate::shared::value_conv::value_to_pyobject(py, &payload_val).unwrap_or(py.None()),
            )
            .ok();
        }
        result.push(dict.into_any().unbind());
    }
    result
}
