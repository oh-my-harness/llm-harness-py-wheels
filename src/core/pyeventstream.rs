//! EventStream + WaitForExternalEventTool 的 Python 包装。
//!
//! 用于 human-in-the-loop 场景：Python 侧创建 channel，将 tool 注册到 engine，
//! 通过 handle 在外部推送事件，阻塞等待的 tool 被唤醒后返回结果。

use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use llm_harness_types::{DataBlock, Tool};
use llm_harness_workflow::lifecycle::event::{Event, EventStream, WaitForExternalEventTool};
use llm_harness_workflow::lifecycle::human::{
    HumanApprovalDefault, RequestHumanApprovalTool, RequestHumanInputTool,
};
use llm_harness_workflow::lifecycle::task::TaskId;
use pyo3::prelude::*;
use tokio::sync::{Mutex, mpsc};

use crate::shared::value_conv::pyobject_to_value;

/// mpsc-backed EventStream。
struct ChannelStream {
    rx: mpsc::Receiver<Event>,
}

impl EventStream for ChannelStream {
    fn next<'a>(&'a mut self) -> BoxFuture<'a, Option<Event>> {
        Box::pin(async { self.rx.recv().await })
    }
}

/// 持有 sender 侧，供 Python 外部推送事件。
#[pyclass(name = "EventStreamHandle")]
pub struct PyEventStreamHandle {
    tx: mpsc::Sender<Event>,
}

#[pymethods]
impl PyEventStreamHandle {
    fn submit(&self, content: &str, details: &Bound<'_, PyAny>) -> PyResult<()> {
        let details_val = pyobject_to_value(details)?;
        let event = Event {
            content: vec![DataBlock::text(content.to_string())],
            details: details_val,
        };
        self.tx
            .try_send(event)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("submit failed: {e}")))
    }
}

#[pyclass(name = "WaitForExternalEventTool")]
pub struct PyWaitForExternalEventTool {
    pub(crate) tool: Arc<dyn Tool>,
}

#[pymethods]
impl PyWaitForExternalEventTool {
    fn name(&self) -> &str {
        self.tool.name()
    }
    fn description(&self) -> &str {
        self.tool.description()
    }
}

// ── Human interaction channels ──────────────────────────────────────────────

/// 当前挂起请求的 request_id（tool 发起等待时写入，handle submit 时读取注入）。
///
/// 上游 human tool 通过 `Event.details["request_id"] == ctx.tool_use_id`
/// 过滤事件；`tool_use_id` 由 LLM 调用时生成，Python 侧不可预知。此共享
/// 单元让 handle 自动补上正确的 id——支持单挂起请求（human-in-the-loop
/// 的典型形态），不支持多并发挂起请求。
type PendingRequestId = Arc<parking_lot::Mutex<Option<String>>>;

/// human interaction channel 的 handle。
///
/// `submit` 时自动注入当前挂起请求的 `request_id`（覆盖用户误传的同名
/// 字段），调用方只需提供 `decision` / `value`。tool 尚未发起请求时
/// submit 返回 RuntimeError。
#[pyclass(name = "HumanResponseHandle")]
pub struct PyHumanResponseHandle {
    tx: mpsc::Sender<Event>,
    pending: PendingRequestId,
}

#[pymethods]
impl PyHumanResponseHandle {
    fn submit(&self, content: &str, details: &Bound<'_, PyAny>) -> PyResult<()> {
        let request_id = self.pending.lock().clone().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "no pending human request: the tool has not been called yet",
            )
        })?;
        let mut details_val = pyobject_to_value(details)?;
        if let serde_json::Value::Object(map) = &mut details_val {
            map.insert(
                "request_id".to_string(),
                serde_json::Value::String(request_id),
            );
        }
        let event = Event {
            content: vec![DataBlock::text(content.to_string())],
            details: details_val,
        };
        self.tx
            .try_send(event)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("submit failed: {e}")))
    }
}

/// 包装上游 human tool：execute 前记录当前 `tool_use_id` 供 handle 注入。
struct HumanToolWrapper {
    inner: Arc<dyn Tool>,
    pending: PendingRequestId,
}

impl Tool for HumanToolWrapper {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn level(&self) -> llm_harness_types::ToolLevel {
        // 透传 BuiltIn 级别——上游 human tool 是内置 tool，名字在保留列表；
        // 用 trait 默认的 User 级会被 tool name conflict 检查拒绝。
        self.inner.level()
    }
    fn description(&self) -> &str {
        self.inner.description()
    }
    fn parameters_schema(&self) -> &serde_json::Value {
        self.inner.parameters_schema()
    }
    fn execution_mode(&self) -> llm_harness_types::ToolExecutionMode {
        self.inner.execution_mode()
    }
    fn execute<'a>(
        &'a self,
        args: serde_json::Value,
        ctx: &'a llm_harness_types::ToolContext,
    ) -> BoxFuture<'a, Result<llm_harness_types::ToolResult, llm_harness_types::ToolFailure>> {
        *self.pending.lock() = Some(ctx.tool_use_id.as_str().to_string());
        self.inner.execute(args, ctx)
    }
}

/// human approval tool 的 Python 包装（不透明，供 engine 注册）。
#[pyclass(name = "HumanApprovalTool")]
pub struct PyHumanApprovalTool {
    pub(crate) tool: Arc<dyn Tool>,
}

#[pymethods]
impl PyHumanApprovalTool {
    fn name(&self) -> &str {
        self.tool.name()
    }
    fn description(&self) -> &str {
        self.tool.description()
    }
}

/// human input tool 的 Python 包装（不透明，供 engine 注册）。
#[pyclass(name = "HumanInputTool")]
pub struct PyHumanInputTool {
    pub(crate) tool: Arc<dyn Tool>,
}

#[pymethods]
impl PyHumanInputTool {
    fn name(&self) -> &str {
        self.tool.name()
    }
    fn description(&self) -> &str {
        self.tool.description()
    }
}

/// Create a human approval channel.
///
/// Returns `(handle, tool)`. Register `tool` on the engine via
/// `.with_external_tool(tool)`; when the LLM calls `request_human_approval`,
/// execution pauses until `handle.submit("approved", {"decision": "approve"})`
/// is called. On timeout, the fail-safe `default` ("approve"/"deny") applies.
#[pyfunction]
#[pyo3(signature = (task_id, timeout_seconds = 300.0, default = "deny"))]
pub fn create_human_approval_channel(
    py: Python<'_>,
    task_id: &str,
    timeout_seconds: f64,
    default: &str,
) -> PyResult<(Py<PyHumanResponseHandle>, Py<PyHumanApprovalTool>)> {
    let default = match default {
        "approve" => HumanApprovalDefault::Approve,
        "deny" => HumanApprovalDefault::Deny,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "default must be 'approve' or 'deny', got {other:?}"
            )));
        }
    };
    let (tx, rx) = mpsc::channel::<Event>(16);
    let stream: Arc<Mutex<Box<dyn EventStream>>> =
        Arc::new(Mutex::new(Box::new(ChannelStream { rx })));
    let pending: PendingRequestId = Arc::new(parking_lot::Mutex::new(None));
    let inner = RequestHumanApprovalTool::new(Arc::clone(&stream))
        .with_timeout(Duration::from_secs_f64(timeout_seconds))
        .with_default(default);
    let tool: Arc<dyn Tool> = Arc::new(HumanToolWrapper {
        inner: Arc::new(inner),
        pending: Arc::clone(&pending),
    });
    let _ = task_id; // 仅用于日志/区分 channel；stream 本身不按 task_id 过滤。
    let handle = Py::new(py, PyHumanResponseHandle { tx, pending })?;
    let tool_wrapper = Py::new(py, PyHumanApprovalTool { tool })?;
    Ok((handle, tool_wrapper))
}

/// Create a human input channel.
///
/// Returns `(handle, tool)`. Register `tool` via `.with_external_tool(tool)`;
/// when the LLM calls `request_human_input`, execution pauses until
/// `handle.submit("42", {"value": 42})` is called. On timeout, `default`
/// is returned to the model.
#[pyfunction]
#[pyo3(signature = (task_id, timeout_seconds = 300.0, default = None))]
pub fn create_human_input_channel(
    py: Python<'_>,
    task_id: &str,
    timeout_seconds: f64,
    default: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Py<PyHumanResponseHandle>, Py<PyHumanInputTool>)> {
    let default_val = match default {
        Some(v) => pyobject_to_value(v)?,
        None => serde_json::Value::Null,
    };
    let (tx, rx) = mpsc::channel::<Event>(16);
    let stream: Arc<Mutex<Box<dyn EventStream>>> =
        Arc::new(Mutex::new(Box::new(ChannelStream { rx })));
    let pending: PendingRequestId = Arc::new(parking_lot::Mutex::new(None));
    let inner = RequestHumanInputTool::new(Arc::clone(&stream))
        .with_timeout(Duration::from_secs_f64(timeout_seconds))
        .with_default(default_val);
    let tool: Arc<dyn Tool> = Arc::new(HumanToolWrapper {
        inner: Arc::new(inner),
        pending: Arc::clone(&pending),
    });
    let _ = task_id;
    let handle = Py::new(py, PyHumanResponseHandle { tx, pending })?;
    let tool_wrapper = Py::new(py, PyHumanInputTool { tool })?;
    Ok((handle, tool_wrapper))
}

/// Create a human-in-the-loop event channel.
///
/// Returns `(handle, wait_tool)`. Register `wait_tool` on the engine
/// or harness; when the LLM calls it, execution pauses until
/// `handle.submit(event_type, payload)` is called from another thread.
#[pyfunction]
pub fn create_event_channel(
    py: Python<'_>,
    task_id: &str,
) -> PyResult<(Py<PyEventStreamHandle>, Py<PyWaitForExternalEventTool>)> {
    let (tx, rx) = mpsc::channel::<Event>(16);
    let stream: Arc<Mutex<Box<dyn EventStream>>> =
        Arc::new(Mutex::new(Box::new(ChannelStream { rx })));
    let tid = TaskId(task_id.to_string());
    let descriptor = serde_json::json!({ "review_id": tid.0.clone() });
    let tool: Arc<dyn Tool> =
        Arc::new(WaitForExternalEventTool::new(stream, descriptor, None, tid));
    let handle = Py::new(py, PyEventStreamHandle { tx })?;
    let tool_wrapper = Py::new(py, PyWaitForExternalEventTool { tool })?;
    Ok((handle, tool_wrapper))
}
