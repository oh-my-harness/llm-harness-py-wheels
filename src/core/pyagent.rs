//! `PyAgent` — 包装 `Agent` + 全局 tokio runtime。
//!
//! 验证风险点：
//! - 全局 tokio runtime 使用 `PyOnceLock` 初始化（非 `Lazy`/`LazyLock`，
//!   后者在 pytest 下可能产生双向死锁）。
//! - `Agent::prompt()` 的 async 驱动：`py.detach()` 释放 GIL 后，
//!   `runtime.block_on()` 运行 agent loop。
//! - 事件流从 Rust broadcast 到 Python 消费者的链路（通过 `subscribe`）。

#[cfg(feature = "test-utils")]
use std::sync::Arc;

#[cfg(feature = "test-utils")]
use llm_harness_agent::Agent;
#[cfg(feature = "test-utils")]
use llm_harness_agent::AgentOptions;
#[cfg(feature = "test-utils")]
use llm_harness_loop::test_utils::{MockLlmClient, MockResponse};
#[cfg(feature = "test-utils")]
use llm_harness_types::Tool;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;

/// 全局 tokio runtime——所有 `PyAgent` 实例共享。
///
/// 使用 `PyOnceLock`（PyO3 0.29 中 `GILOnceCell` 的替代品）确保初始化
/// 在持有 GIL 时进行，避免 `Lazy`/`LazyLock` 在 pytest 下可能产生的
/// 双向死锁（线程 A 持有静态初始化锁等待 GIL，线程 B 持有 GIL 等待
/// 静态初始化锁）。
static RT: PyOnceLock<tokio::runtime::Runtime> = PyOnceLock::new();
/// 获取或初始化全局 tokio runtime。
pub(crate) fn runtime(py: Python<'_>) -> &'static tokio::runtime::Runtime {
    RT.get_or_init(py, || {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build tokio runtime")
    })
}

/// Python 侧的 `Agent` 包装类。
///
/// 整个类（struct + impl + pymethods）门控在 `test-utils` 后：
/// `#[new]` 用 `MockLlmClient`（test-only），生产 wheel 不应暴露。
/// 门控 `#[pyclass]` 本身（而非仅 `add_class`）确保 stub 生成器
/// 在生产构建中看不到 `Agent` 类，避免 .pyi 与运行时漂移。
#[cfg(feature = "test-utils")]
#[pyclass(name = "Agent")]
pub struct PyAgent {
    agent: Arc<Agent>,
}
#[cfg(feature = "test-utils")]
impl PyAgent {
    /// 从已有 `Arc<Agent>` 构造 `PyAgent`（供 builder 等内部路径使用）。
    pub fn from_agent(agent: Arc<Agent>) -> Self {
        Self { agent }
    }
}

#[cfg(feature = "test-utils")]
#[pymethods]
impl PyAgent {
    /// 创建一个使用 `MockLlmClient` 的 Agent（仅供测试）。
    ///
    /// `model` 参数仅用于标识，不影响 mock 响应。
    /// `responses` 为可选的自定义响应列表，每个元素是 dict：
    ///   - {"type": "text", "text": "..."}
    ///   - {"type": "tool_use", "id": "...", "name": "...", "args": "..."}
    ///   - {"type": "tool_use_end_turn", "id": "...", "name": "...", "args": "..."}
    /// `tools` 为可选的 Tool 列表（来自 `create_tool`）。
    /// 不传 `responses` 时使用默认的 4 条 "hello from mock" 文本响应。
    /// 生产环境请使用 `HarnessBuilder` 构建真实 provider 的 Agent。
    #[cfg(feature = "test-utils")]
    #[new]
    #[pyo3(signature = (model="mock-model", responses=None, tools=None))]
    fn new(
        model: &str,
        responses: Option<Vec<Bound<'_, PyAny>>>,
        tools: Option<Vec<Bound<'_, crate::core::pytool::PyToolWrapper>>>,
    ) -> PyResult<Self> {
        let mock_responses: Vec<MockResponse> = match responses {
            Some(list) => list
                .iter()
                .map(mock_response_from_py)
                .collect::<PyResult<Vec<_>>>()?,
            None => vec![
                MockResponse::text("hello from mock"),
                MockResponse::text("hello from mock"),
                MockResponse::text("hello from mock"),
                MockResponse::text("hello from mock"),
            ],
        };
        let client = Arc::new(MockLlmClient::new(mock_responses));
        let mut opts = AgentOptions::new(model.to_string());
        if let Some(tool_list) = tools {
            opts.tools = tool_list
                .iter()
                .map(|t| {
                    let wrapper = t.extract::<PyRef<'_, crate::core::pytool::PyToolWrapper>>()?;
                    Ok::<Arc<dyn Tool>, PyErr>(wrapper.tool.clone())
                })
                .collect::<PyResult<Vec<Arc<dyn Tool>>>>()?;
        }
        let agent = Arc::new(Agent::new(client, opts));
        Ok(Self { agent })
    }

    /// 同步执行 prompt，阻塞直到完成。
    ///
    /// 通过 `py.detach()` 释放 GIL，让 tokio worker 线程在需要时
    /// 能 acquire GIL（例如执行 Python tool callback）。
    #[pyo3(signature = (text, attachments=None))]
    fn prompt(
        &self,
        py: Python<'_>,
        text: &str,
        attachments: Option<Vec<Bound<'_, crate::core::pytool::PyAttachment>>>,
    ) -> PyResult<String> {
        let agent = self.agent.clone();
        let has_attachments = attachments.as_ref().is_some_and(|l| !l.is_empty());
        let text = text.to_string();
        let message = has_attachments
            .then(|| crate::core::pyharness::user_message_from_py(&text, attachments));
        let rt = runtime(py);

        // 释放 GIL + panic 隔离 + 信号检查（Ctrl+C 可打断）。
        crate::shared::pyerror::block_on_with_signal_check(
            py,
            rt,
            async move {
                let result = if let Some(message) = message {
                    // Agent 只有 replace 语义的 prompt_with_messages；带附件时
                    // 用它（调用方自行负责 transcript 预期）。
                    agent.prompt_with_messages(vec![message]).await
                } else {
                    agent.prompt(text).await
                };
                result.map_err(|e| crate::shared::pyerror::agent_error_to_pyerr(e))
            },
            200,
        )?;

        // 返回最后一条 assistant 消息的文本内容。
        let state = self.agent.state();
        let last = state.messages.last();
        let response = match last {
            Some(llm_harness_types::AgentMessage::Assistant(msg)) => msg.text_content(),
            _ => String::new(),
        };
        Ok(response)
    }

    /// 获取当前 agent 状态中的消息数量。
    fn message_count(&self) -> usize {
        self.agent.state().messages.len()
    }

    /// 获取当前 phase（"idle" / "running"）。
    fn phase(&self) -> &'static str {
        match self.agent.state().phase {
            llm_harness_agent::AgentPhase::Idle => "idle",
            llm_harness_agent::AgentPhase::Running => "running",
        }
    }

    /// 返回事件迭代器。`timeout_ms` 为单次 `__next__` 等待超时（毫秒）。
    ///
    /// 典型用法：`for event in agent.events(timeout_ms=5000): ...`
    #[pyo3(signature = (timeout_ms=5000, max_consecutive_timeouts=1))]
    fn events(
        &self,
        py: Python<'_>,
        timeout_ms: u64,
        max_consecutive_timeouts: u32,
    ) -> PyResult<Py<crate::shared::event_stream::PyEventIterator>> {
        let rx = self.agent.subscribe();
        let handle = runtime(py).handle().clone();
        let iter = crate::shared::event_stream::PyEventIterator::new(
            rx,
            timeout_ms,
            max_consecutive_timeouts,
            handle,
        );
        Py::new(py, iter)
    }

    /// 取消当前正在运行的 prompt（如果有）。不阻塞。
    fn abort(&self) {
        self.agent.abort();
    }

    /// 返回最近一次运行的错误消息（如果有）。
    #[getter]
    fn error_message(&self) -> Option<String> {
        self.agent.state().error_message.clone()
    }
}

/// Convert a Python dict describing a mock response into a `MockResponse`.
#[cfg(feature = "test-utils")]
fn mock_response_from_py(obj: &Bound<'_, PyAny>) -> PyResult<MockResponse> {
    use pyo3::types::PyDict;
    let dict = obj.cast::<PyDict>()?;
    let rtype: String = dict
        .get_item("type")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'type' key"))?
        .extract()?;
    match rtype.as_str() {
        "text" => {
            let text: String = dict
                .get_item("text")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'text' key"))?
                .extract()?;
            Ok(MockResponse::text(&text))
        }
        "tool_use" => {
            let id: String = dict
                .get_item("id")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'id' key"))?
                .extract()?;
            let name: String = dict
                .get_item("name")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'name' key"))?
                .extract()?;
            let args: String = dict
                .get_item("args")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'args' key"))?
                .extract()?;
            Ok(MockResponse::tool_use(&id, &name, &args))
        }
        "tool_use_end_turn" => {
            let id: String = dict
                .get_item("id")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'id' key"))?
                .extract()?;
            let name: String = dict
                .get_item("name")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'name' key"))?
                .extract()?;
            let args: String = dict
                .get_item("args")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'args' key"))?
                .extract()?;
            Ok(MockResponse::tool_use_end_turn(&id, &name, &args))
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown mock response type: {other}"
        ))),
    }
}
