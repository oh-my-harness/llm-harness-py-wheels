//! PyO3 SDK 验证 crate。

use std::sync::Arc;

use llm_harness_sandbox::os::OsEnv;
use llm_harness_workflow::workflow::executor::{HttpCallExecutor, HttpCallPolicy, ShellExecutor};
use pyo3::prelude::*;

pub mod core;
pub mod infra;
pub mod knowledge;
pub mod runtime;
pub mod shared;
pub mod strategy;

/// PyO3 module entry point.
#[pymodule]
fn senza(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 桥接 Rust tracing → Python logging：用户 `logging.basicConfig(level=DEBUG)`
    // 即可看到 Rust 底座日志，级别/handler/格式完全由 Python 侧控制。
    crate::shared::pylogging::init_logging();
    m.add(
        "RustPanicError",
        py.get_type::<crate::shared::pyerror::RustPanicError>(),
    )?;
    m.add(
        "SenzaError",
        py.get_type::<crate::shared::pyerror::SenzaError>(),
    )?;
    m.add(
        "ProviderError",
        py.get_type::<crate::shared::pyerror::ProviderError>(),
    )?;
    m.add(
        "RateLimitError",
        py.get_type::<crate::shared::pyerror::RateLimitError>(),
    )?;
    m.add(
        "ProviderTimeoutError",
        py.get_type::<crate::shared::pyerror::ProviderTimeoutError>(),
    )?;
    m.add(
        "InvalidRequestError",
        py.get_type::<crate::shared::pyerror::InvalidRequestError>(),
    )?;
    m.add(
        "UnauthorizedError",
        py.get_type::<crate::shared::pyerror::UnauthorizedError>(),
    )?;
    m.add(
        "ForbiddenError",
        py.get_type::<crate::shared::pyerror::ForbiddenError>(),
    )?;
    m.add(
        "OverloadedError",
        py.get_type::<crate::shared::pyerror::OverloadedError>(),
    )?;
    m.add(
        "ServerError",
        py.get_type::<crate::shared::pyerror::ServerError>(),
    )?;
    m.add(
        "StreamError",
        py.get_type::<crate::shared::pyerror::StreamError>(),
    )?;
    m.add(
        "StreamIncompleteError",
        py.get_type::<crate::shared::pyerror::StreamIncompleteError>(),
    )?;
    m.add(
        "NetworkError",
        py.get_type::<crate::shared::pyerror::NetworkError>(),
    )?;
    m.add(
        "DecodeError",
        py.get_type::<crate::shared::pyerror::DecodeError>(),
    )?;
    m.add(
        "ProviderCodeError",
        py.get_type::<crate::shared::pyerror::ProviderCodeError>(),
    )?;
    m.add(
        "ToolError",
        py.get_type::<crate::shared::pyerror::ToolError>(),
    )?;
    m.add(
        "ToolArgumentError",
        py.get_type::<crate::shared::pyerror::ToolArgumentError>(),
    )?;
    m.add(
        "ToolAbortedError",
        py.get_type::<crate::shared::pyerror::ToolAbortedError>(),
    )?;
    m.add(
        "ToolExecutionError",
        py.get_type::<crate::shared::pyerror::ToolExecutionError>(),
    )?;
    m.add(
        "BudgetExceededError",
        py.get_type::<crate::shared::pyerror::BudgetExceededError>(),
    )?;
    m.add(
        "WorkflowError",
        py.get_type::<crate::shared::pyerror::WorkflowError>(),
    )?;
    m.add(
        "StepTimeoutError",
        py.get_type::<crate::shared::pyerror::StepTimeoutError>(),
    )?;
    m.add(
        "StepFailedError",
        py.get_type::<crate::shared::pyerror::StepFailedError>(),
    )?;
    m.add(
        "WorkflowPausedError",
        py.get_type::<crate::shared::pyerror::WorkflowPausedError>(),
    )?;
    m.add(
        "ValidationError",
        py.get_type::<crate::shared::pyerror::ValidationError>(),
    )?;
    m.add(
        "HarnessStateError",
        py.get_type::<crate::shared::pyerror::HarnessStateError>(),
    )?;
    m.add(
        "CompactionError",
        py.get_type::<crate::shared::pyerror::CompactionError>(),
    )?;
    m.add(
        "StreamIdleTimeoutError",
        py.get_type::<crate::shared::pyerror::StreamIdleTimeoutError>(),
    )?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(set_event_loop, m)?)?;
    m.add_function(wrap_pyfunction!(to_json, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::pyviewer::read_sessions, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::pyviewer::viewer_html, m)?)?;
    m.add_function(wrap_pyfunction!(from_json, m)?)?;
    // `PyAgent`'s `#[new]` uses `MockLlmClient` (test-only). Gating the
    // class registration behind `test-utils` keeps it out of production
    // wheels, where it would be visible via `dir(senza)` yet raise
    // `TypeError: cannot create 'Agent' instances`. Production callers
    // use `HarnessBuilder` → `AgentHarness` instead.
    #[cfg(feature = "test-utils")]
    m.add_class::<crate::core::pyagent::PyAgent>()?;
    m.add_class::<crate::shared::event_stream::PyEventIterator>()?;
    m.add_class::<crate::runtime::pyworkflow::PyJudgeWrapper>()?;
    m.add_class::<crate::runtime::pyworkflow::PyCompositeJudge>()?;
    m.add_class::<crate::runtime::pyworkflow::PyExecutorWrapper>()?;
    m.add_class::<crate::runtime::pyworkflow::PyEnvWrapper>()?;
    m.add_class::<crate::core::pyhooks::PyHookWrapper>()?;
    m.add_class::<crate::core::pytool::PyToolWrapper>()?;
    m.add_class::<crate::core::pytool::PyToolContext>()?;
    m.add_class::<crate::core::pytool::PyAttachment>()?;
    m.add_function(wrap_pyfunction!(create_sync_tool, m)?)?;
    m.add_function(wrap_pyfunction!(create_tool, m)?)?;
    m.add_function(wrap_pyfunction!(create_judge, m)?)?;
    m.add_function(wrap_pyfunction!(create_composite_judge, m)?)?;
    m.add_function(wrap_pyfunction!(create_executor, m)?)?;
    m.add_function(wrap_pyfunction!(create_shell_executor, m)?)?;
    m.add_function(wrap_pyfunction!(create_http_executor, m)?)?;
    m.add_function(wrap_pyfunction!(create_os_env, m)?)?;
    m.add_function(wrap_pyfunction!(create_fs_tools_plugin, m)?)?;
    m.add_class::<crate::core::pywebtools::PyNativeTool>()?;
    m.add_function(wrap_pyfunction!(
        crate::core::pywebtools::create_web_search_tool,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::core::pywebtools::create_web_fetch_tool,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::core::pywebtools::create_web_tools_plugin,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::core::pywebtools::create_code_exec_tool,
        m
    )?)?;
    m.add_class::<crate::core::pyinspector::PyInspector>()?;
    m.add_function(wrap_pyfunction!(
        strategy::pysafety::create_safety_defaults_plugin,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        strategy::pyloopsafety::create_loop_safety_plugin,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        strategy::pyvision::create_vision_degrade_hook,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        strategy::pyvision::create_observation_shielding_hook,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        strategy::pystatuspanel::create_status_panel_plugin,
        m
    )?)?;
    m.add_class::<crate::strategy::pymemorydefense::PyMemoryDefensePluginBuilder>()?;
    m.add_function(wrap_pyfunction!(
        strategy::pymemorydefense::create_memory_defense_plugin,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        strategy::pyinjection::create_injection_filter_plugin,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        strategy::pysourcetag::create_source_tag_plugin,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        strategy::pyprojectinstr::create_project_instruction_plugin,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(strategy::pyaudit::create_audit_plugin, m)?)?;
    m.add_function(wrap_pyfunction!(
        strategy::pynotify::create_notify_plugin,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        strategy::pytoolguard::create_tool_output_guard_plugin,
        m
    )?)?;
    m.add_class::<crate::strategy::pyeventstreams::PyWebhookChannel>()?;
    m.add_class::<crate::strategy::pyeventstreams::PyEventStream>()?;
    m.add_function(wrap_pyfunction!(
        strategy::pyeventstreams::create_webhook_stream,
        m
    )?)?;
    m.add_class::<crate::strategy::pyeventstreams::PyHeartbeatHandle>()?;
    m.add_class::<crate::strategy::pyeventstreams::PyShellMonitorHandle>()?;
    m.add_function(wrap_pyfunction!(
        strategy::pyeventstreams::create_timer_stream,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        strategy::pyeventstreams::create_heartbeat_stream,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        strategy::pyeventstreams::create_shell_monitor_stream,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        strategy::pycompaction::create_context_aware_compaction_prompt,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(create_before_turn_hook, m)?)?;
    m.add_class::<crate::core::pyeventstream::PyEventStreamHandle>()?;
    m.add_class::<crate::core::pyeventstream::PyWaitForExternalEventTool>()?;
    m.add_function(wrap_pyfunction!(
        crate::core::pyeventstream::create_event_channel,
        m
    )?)?;
    m.add_class::<crate::core::pyeventstream::PyHumanResponseHandle>()?;
    m.add_class::<crate::core::pyeventstream::PyHumanApprovalTool>()?;
    m.add_class::<crate::core::pyeventstream::PyHumanInputTool>()?;
    m.add_function(wrap_pyfunction!(
        crate::core::pyeventstream::create_human_approval_channel,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::core::pyeventstream::create_human_input_channel,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(create_after_turn_hook, m)?)?;
    m.add_function(wrap_pyfunction!(create_before_run_hook, m)?)?;
    m.add_function(wrap_pyfunction!(create_after_provider_response_hook, m)?)?;
    m.add_function(wrap_pyfunction!(create_before_provider_request_hook, m)?)?;
    m.add_function(wrap_pyfunction!(create_before_tool_call_hook, m)?)?;
    m.add_function(wrap_pyfunction!(create_after_tool_call_hook, m)?)?;
    m.add_function(wrap_pyfunction!(create_should_stop_hook, m)?)?;
    m.add_function(wrap_pyfunction!(create_before_compact_hook, m)?)?;
    m.add_function(wrap_pyfunction!(create_transform_context_hook, m)?)?;
    m.add_function(wrap_pyfunction!(create_prepare_next_turn_hook, m)?)?;
    m.add_function(wrap_pyfunction!(create_final_answer_validator, m)?)?;
    m.add_function(wrap_pyfunction!(create_after_run_hook, m)?)?;
    m.add_function(wrap_pyfunction!(create_on_abort_hook, m)?)?;
    m.add_function(wrap_pyfunction!(create_provider_error_hook, m)?)?;
    m.add_class::<crate::core::pybuilder::PyHarnessBuilder>()?;
    m.add_class::<crate::core::pybuilder::PyUsageLedger>()?;
    m.add_class::<crate::core::pyplugin::PyPluginWrapper>()?;
    m.add_function(wrap_pyfunction!(create_plugin, m)?)?;
    m.add_class::<crate::knowledge::pylocalsource::PyKnowledgeSource>()?;
    m.add_function(wrap_pyfunction!(
        knowledge::pylocalsource::create_local_knowledge_source,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        knowledge::pyknowledge::create_knowledge_plugin,
        m
    )?)?;
    m.add_class::<crate::knowledge::pymemory::PyMemoryStore>()?;
    m.add_class::<crate::knowledge::pymemory::PyMemoryWritePolicy>()?;
    m.add_class::<crate::knowledge::pymemory::PyMemoryMutationGate>()?;
    m.add_function(wrap_pyfunction!(
        knowledge::pymemory::create_in_memory_store,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        knowledge::pymemory::create_secure_write_policy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        knowledge::pymemory::create_allow_all_gate,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        knowledge::pymemory::create_memory_plugin,
        m
    )?)?;
    m.add_class::<crate::knowledge::pysessionrecall::PySessionRecallIndex>()?;
    m.add_class::<crate::knowledge::pysessionrecall::PySessionRepo>()?;
    m.add_class::<crate::knowledge::pysessionrecall::PySessionRecallKnowledgeSource>()?;
    m.add_function(wrap_pyfunction!(
        knowledge::pysessionrecall::create_in_memory_session_recall_index,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        knowledge::pysessionrecall::create_sqlite_session_recall_index,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        knowledge::pysessionrecall::create_in_memory_session_repo,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        knowledge::pysessionrecall::create_jsonl_session_repo,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        knowledge::pysessionrecall::create_session_recall_knowledge_source,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        knowledge::pysessionrecall::create_history_recall_plugin,
        m
    )?)?;
    m.add_class::<crate::core::pyresponseformat::PyResponseFormat>()?;
    m.add_function(wrap_pyfunction!(
        crate::core::pyresponseformat::create_json_object_format,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::core::pyresponseformat::create_json_schema_format,
        m
    )?)?;
    m.add_class::<crate::core::pyprovider::PyProvider>()?;
    m.add_function(wrap_pyfunction!(
        crate::core::pyprovider::create_openai_provider,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::core::pyprovider::create_anthropic_provider,
        m
    )?)?;
    m.add_class::<crate::runtime::pypricing::PyPricingProvider>()?;
    m.add_function(wrap_pyfunction!(
        crate::runtime::pypricing::create_pricing_provider,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::runtime::pypricing::create_pricing_provider_callback,
        m
    )?)?;
    m.add_class::<crate::runtime::pybudget::PyBudgetExceededHook>()?;
    m.add_function(wrap_pyfunction!(
        crate::runtime::pybudget::create_budget_exceeded_hook,
        m
    )?)?;
    m.add_class::<crate::runtime::pyrules::PyPredicate>()?;
    m.add_class::<crate::runtime::pyrules::PyRuleChain>()?;
    m.add_class::<crate::runtime::pyrules::PyRuleChainBuilder>()?;
    m.add_function(wrap_pyfunction!(
        crate::runtime::pyrules::create_rule_chain,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::runtime::pyrules::create_contains_predicate,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::runtime::pyrules::create_regex_field_predicate,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::runtime::pyrules::create_number_range_predicate,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::runtime::pyrules::create_rate_limit_predicate,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::runtime::pyrules::create_rule_approval_hook,
        m
    )?)?;
    m.add_class::<crate::runtime::pyskills::PySkill>()?;
    m.add_function(wrap_pyfunction!(crate::runtime::pyskills::load_skills, m)?)?;
    m.add_class::<crate::core::pyharness::PyAgentHarness>()?;
    m.add_class::<crate::core::pyharness::PyHarnessEventIterator>()?;
    m.add_class::<crate::runtime::pyworkflow::PyWorkflowEngine>()?;
    m.add_class::<crate::runtime::pyworkflow::PyWorkflowEventIterator>()?;
    m.add_class::<crate::runtime::pymcp::PyMcpServerConfig>()?;
    m.add_class::<crate::runtime::pymcp::PyMcpManager>()?;
    m.add_class::<crate::infra::pyaudit::PyJsonlAuditSink>()?;
    m.add_class::<crate::infra::pytrace::PyInMemoryTraceExporter>()?;
    m.add_class::<crate::infra::pysandbox::PySandbox>()?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(
        crate::infra::pysandbox::create_seatbelt_sandbox,
        m
    )?)?;
    #[cfg(target_os = "linux")]
    m.add_function(wrap_pyfunction!(
        crate::infra::pysandbox::create_bwrap_sandbox,
        m
    )?)?;
    Ok(())
}

/// Return the SDK version string.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Register the user's asyncio event loop for async callback scheduling.
///
/// When set, `async def` tool/hook/budget callbacks are scheduled onto the
/// registered loop via `asyncio.run_coroutine_threadsafe`, instead of
/// `asyncio.run()` (which creates a throwaway loop).  This lets callbacks
/// share loop-bound resources (sessions, locks, queues) with the caller.
///
/// The loop must be running on another thread; otherwise a deadlock will
/// occur because the blocking thread waits for a result the loop cannot
/// produce.
#[pyfunction]
#[pyo3(text_signature = "(loop)")]
fn set_event_loop(loop_obj: Py<PyAny>) {
    crate::core::pyloop::set_event_loop(loop_obj);
}

/// Convert a Python object to a JSON string.
#[pyfunction]
fn to_json(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let value = crate::shared::value_conv::pyobject_to_value(obj)?;
    Ok(value.to_string())
}

/// Parse a JSON string into a Python object.
#[pyfunction]
fn from_json(py: Python<'_>, json_str: &str) -> PyResult<Py<PyAny>> {
    let value: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    crate::shared::value_conv::value_to_pyobject(py, &value)
}

/// 从 Python callable 创建一个同步 `Tool`。
///
/// 此函数是 `create_tool` 的别名——`create_tool` 已自动检测 `async def`
/// 回调并正确处理。保留此名称以简化从旧 API 的迁移。
#[pyfunction]
fn create_sync_tool<'py>(
    py: Python<'py>,
    name: &str,
    description: &str,
    parameters_schema: &Bound<'py, PyAny>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::core::pytool::PyToolWrapper>> {
    create_tool(py, name, description, parameters_schema, callback, false)
}

/// 从 Python callable 创建一个 `Tool`（统一入口，支持 sync 与 async 回调）。
///
/// 若 `callback` 是 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行——`select()` 内部释放 GIL，无需独立事件循环线程。
#[pyfunction]
#[pyo3(signature = (name, description, parameters_schema, callback, report_duration=false))]
fn create_tool<'py>(
    py: Python<'py>,
    name: &str,
    description: &str,
    parameters_schema: &Bound<'py, PyAny>,
    callback: Py<PyAny>,
    report_duration: bool,
) -> PyResult<Bound<'py, crate::core::pytool::PyToolWrapper>> {
    // Accept dict or str for parameters_schema.
    // If dict, convert to serde_json::Value directly via pyobject_to_value.
    // If str, parse as JSON string (existing behavior).
    use pyo3::types::PyDict;
    let schema: serde_json::Value = if parameters_schema.is_instance_of::<PyDict>() {
        crate::shared::value_conv::pyobject_to_value(parameters_schema)?
    } else {
        let s: String = parameters_schema.extract()?;
        serde_json::from_str(&s)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };
    let tool = crate::core::pytool::PyTool::new(
        name.to_string(),
        description.to_string(),
        schema,
        callback,
        report_duration,
    );
    let wrapper = crate::core::pytool::PyToolWrapper {
        tool: Arc::new(tool),
    };
    Py::new(py, wrapper).map(|p| p.into_bound(py))
}

/// 从 Python callable 创建一个 `StepTransitionJudge`。
///
/// callback 签名：`callback(ctx: dict) -> str`
/// 返回值编码：`"to:<step_id>"`, `"retry"`, `"fail:<reason>"`, `"abort:<reason>"`
#[pyfunction]
fn create_judge<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::runtime::pyworkflow::PyJudgeWrapper>> {
    let judge = crate::runtime::pyworkflow::PyJudge::new(callback);
    let wrapper = crate::runtime::pyworkflow::PyJudgeWrapper {
        judge: Arc::new(judge)
            as Arc<dyn llm_harness_workflow::workflow::judge::StepTransitionJudge>,
    };
    Py::new(py, wrapper).map(|p| p.into_bound(py))
}

/// 创建一个 CompositeJudge，支持按节点注册独立路由函数。
///
/// 用法：
/// ```python
/// judge = senza.create_composite_judge()
/// judge.on("step1", lambda ctx: "to:step2")
/// judge.on("step2", lambda ctx: "abort:done" if ctx["output"] else "retry")
/// judge.fallback(lambda ctx: "abort:done")  # 可选
/// engine = senza.WorkflowEngine(workflow, provider, model, judge)
/// ```
///
/// 未注册 `.on()` 的 step 会依次尝试：用户 fallback → 声明式边 (Expr/Label) → Abort。
/// 如果 workflow 有声明式条件边 (Expr 或 Label)，引擎会自动注入 EdgeConditionJudge 作为 fallback。
#[pyfunction]
fn create_composite_judge<'py>(
    py: Python<'py>,
) -> PyResult<Bound<'py, crate::runtime::pyworkflow::PyCompositeJudge>> {
    Py::new(py, crate::runtime::pyworkflow::PyCompositeJudge::new()).map(|p| p.into_bound(py))
}

/// 从 Python callable 创建一个 `StepExecutor`。
///
/// callback 签名：`callback(ctx: dict) -> dict`
/// 返回 dict 须含 `"output"` (str)，可选 `"structured"` (dict)。
#[pyfunction]
fn create_executor<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::runtime::pyworkflow::PyExecutorWrapper>> {
    let executor = crate::runtime::pyworkflow::PyExecutor::new(callback);
    let wrapper = crate::runtime::pyworkflow::PyExecutorWrapper {
        executor: Arc::new(executor),
    };
    Py::new(py, wrapper).map(|p| p.into_bound(py))
}

/// Create a ShellExecutor with a command allowlist.
///
/// `commands` is a list of allowed command names (e.g. ["echo", "python"]).
/// `default_timeout_ms` overrides the default timeout per shell call (default 30000).
/// `max_output_bytes` caps stdout/stderr capture (default 1 MiB).
///
/// The executor is NOT registered by default — register with
/// `engine.with_executor("shell", shell_executor)`.
#[pyfunction]
#[pyo3(signature = (commands, default_timeout_ms=30000, max_output_bytes=1048576))]
fn create_shell_executor<'py>(
    py: Python<'py>,
    commands: Vec<String>,
    default_timeout_ms: u64,
    max_output_bytes: usize,
) -> PyResult<Bound<'py, crate::runtime::pyworkflow::PyExecutorWrapper>> {
    let exec = ShellExecutor::new(commands)
        .with_default_timeout(std::time::Duration::from_millis(default_timeout_ms))
        .with_max_output_bytes(max_output_bytes);
    let wrapper = crate::runtime::pyworkflow::PyExecutorWrapper {
        executor: Arc::new(exec),
    };
    Py::new(py, wrapper).map(|p| p.into_bound(py))
}

/// Create an HttpCallExecutor with a host allowlist policy.
///
/// `allowed_hosts` is a list of allowed hostnames (e.g. ["api.example.com"]).
/// `allowed_schemes` defaults to ["https"]; pass ["http", "https"] to allow HTTP.
/// `max_timeout_ms` caps request duration (default 30000).
/// `allow_private_ip_targets` defaults to False (blocks localhost/10.x/172.16.x/192.168.x).
#[pyfunction]
#[pyo3(signature = (allowed_hosts, allowed_schemes=None, max_timeout_ms=30000, allow_private_ip_targets=false))]
fn create_http_executor<'py>(
    py: Python<'py>,
    allowed_hosts: Vec<String>,
    allowed_schemes: Option<Vec<String>>,
    max_timeout_ms: u64,
    allow_private_ip_targets: bool,
) -> PyResult<Bound<'py, crate::runtime::pyworkflow::PyExecutorWrapper>> {
    let mut policy = HttpCallPolicy::new(allowed_hosts)
        .with_max_timeout(std::time::Duration::from_millis(max_timeout_ms));
    if let Some(schemes) = allowed_schemes {
        policy = policy.with_allowed_schemes(schemes);
    }
    policy = policy.allow_private_ip_targets(allow_private_ip_targets);
    let exec = HttpCallExecutor::new(policy);
    let wrapper = crate::runtime::pyworkflow::PyExecutorWrapper {
        executor: Arc::new(exec),
    };
    Py::new(py, wrapper).map(|p| p.into_bound(py))
}

/// Create an OS-backed `ExecutionEnv` rooted at `working_dir`.
///
/// The returned env exposes the real filesystem and shell of the host.
/// Pass it to `WorkflowEngine(..., env=...)` so that executors such as
/// `create_shell_executor` can run real commands (subject to their own
/// allowlists). Without an env, the engine uses `UnsupportedEnv`, whose
/// `execute_shell` always returns an error.
///
/// SECURITY: This env executes real shell commands on the host. The
/// `ShellExecutor` command allowlist is the first line of defense, but
/// callers are responsible for the security of `working_dir` and the
/// surrounding runtime.
#[pyfunction]
#[pyo3(signature = (working_dir="."))]
fn create_os_env<'py>(
    py: Python<'py>,
    working_dir: &str,
) -> PyResult<Bound<'py, crate::runtime::pyworkflow::PyEnvWrapper>> {
    let env: Arc<dyn llm_harness_types::ExecutionEnv> =
        Arc::new(OsEnv::new(std::path::PathBuf::from(working_dir)));
    Py::new(py, crate::runtime::pyworkflow::PyEnvWrapper::new(env)).map(|p| p.into_bound(py))
}

/// 创建一个聚合 `bash`/`read`/`write`/`edit` 四件套的 `FsToolsPlugin`。
///
/// 四个工具通过共享的 `FileSnapshotStore` 耦合：`read` 记录文件快照并
/// 在输出中附加 `[PATH#TAG]` 锚点，`edit` 据此检测 stale 内容并拒绝
/// 对已过期快照的编辑；`write` 在覆写后使对应快照失效。
///
/// 这些工具通过 `ExecutionEnv` 执行真实文件系统 / shell 操作——
/// 必须在 `HarnessBuilder.env(create_os_env(...))` 或
/// `WorkflowEngine(..., env=create_os_env(...))` 提供真实 env 时才有意义。
/// 在 `UnsupportedEnv`（默认）下，`bash`/`read`/`write`/`edit` 会返回错误。
///
/// 用法：
/// ```python
/// plugin = lh.create_fs_tools_plugin()
/// harness = lh.HarnessBuilder("gpt-4o").plugin(plugin).env(lh.create_os_env()).build()
/// ```
#[pyfunction]
fn create_fs_tools_plugin<'py>(
    py: Python<'py>,
) -> PyResult<Bound<'py, crate::core::pyplugin::PyPluginWrapper>> {
    let store = Arc::new(parking_lot::RwLock::new(
        llm_harness_tools::FileSnapshotStore::new(),
    ));
    let plugin: Arc<dyn llm_harness_agent::Plugin> =
        Arc::new(llm_harness_tools::FsToolsPlugin::new(Some(store)));
    Py::new(py, crate::core::pyplugin::PyPluginWrapper::new(plugin)).map(|p| p.into_bound(py))
}

/// 从 Python callable 创建一个 `BeforeTurnHook`。
///
/// callback 签名：`callback(ctx: dict) -> None`
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
#[pyfunction]
fn create_before_turn_hook<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::core::pyhooks::PyHookWrapper>> {
    let hook = crate::core::pyhooks::PyBeforeTurnHook::new(callback);
    Py::new(
        py,
        crate::core::pyhooks::PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::BeforeTurn(Arc::new(hook)),
        },
    )
    .map(|p| p.into_bound(py))
}

/// 从 Python callable 创建一个 `AfterTurnHook`。
///
/// callback 签名：`callback(ctx: dict) -> None`
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
#[pyfunction]
fn create_after_turn_hook<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::core::pyhooks::PyHookWrapper>> {
    let hook = crate::core::pyhooks::PyAfterTurnHook::new(callback);
    Py::new(
        py,
        crate::core::pyhooks::PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::AfterTurn(Arc::new(hook)),
        },
    )
    .map(|p| p.into_bound(py))
}

/// 从 Python callable 创建一个 `BeforeRunHook`。
///
/// callback 签名：`callback(ctx: dict) -> dict | None`
/// 返回 dict 可含 `additional_messages`（list[dict]）和 `system_prompt`（str | None）。
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
#[pyfunction]
fn create_before_run_hook<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::core::pyhooks::PyHookWrapper>> {
    let hook = crate::core::pyhooks::PyBeforeRunHook::new(callback);
    Py::new(
        py,
        crate::core::pyhooks::PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::BeforeRun(Arc::new(hook)),
        },
    )
    .map(|p| p.into_bound(py))
}

/// 从 Python callable 创建一个 `AfterProviderResponseHook`。
///
/// callback 签名：`callback(info: dict) -> None`
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
#[pyfunction]
fn create_after_provider_response_hook<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::core::pyhooks::PyHookWrapper>> {
    let hook = crate::core::pyhooks::PyAfterProviderResponseHook::new(callback);
    Py::new(
        py,
        crate::core::pyhooks::PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::AfterProviderResponse(Arc::new(hook)),
        },
    )
    .map(|p| p.into_bound(py))
}

/// 从 Python callable 创建一个 `BeforeProviderRequestHook`。
///
/// callback 签名：`callback(opts: dict) -> None`
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
#[pyfunction]
fn create_before_provider_request_hook<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::core::pyhooks::PyHookWrapper>> {
    let hook = crate::core::pyhooks::PyBeforeProviderRequestHook::new(callback);
    Py::new(
        py,
        crate::core::pyhooks::PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::BeforeProviderRequest(Arc::new(hook)),
        },
    )
    .map(|p| p.into_bound(py))
}

/// 从 Python callable 创建一个 `BeforeToolCallHook`。
///
/// callback 签名：`callback(ctx: dict) -> str | dict`
/// 返回 `"allow"` 或 `{"action": "modify", "args": ...}` 或 `{"action": "deny", "result": ...}`。
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
#[pyfunction]
fn create_before_tool_call_hook<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::core::pyhooks::PyHookWrapper>> {
    let hook = crate::core::pyhooks::PyBeforeToolCallHook::new(callback);
    Py::new(
        py,
        crate::core::pyhooks::PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::BeforeToolCall(Arc::new(hook)),
        },
    )
    .map(|p| p.into_bound(py))
}

/// 从 Python callable 创建一个 `AfterToolCallHook`。
///
/// callback 签名：`callback(ctx: dict) -> str | dict`
/// 返回 `"passthrough"` 或 `{"action": "patch", "content": ...}`。
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
#[pyfunction]
fn create_after_tool_call_hook<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::core::pyhooks::PyHookWrapper>> {
    let hook = crate::core::pyhooks::PyAfterToolCallHook::new(callback);
    Py::new(
        py,
        crate::core::pyhooks::PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::AfterToolCall(Arc::new(hook)),
        },
    )
    .map(|p| p.into_bound(py))
}

/// 从 Python callable 创建一个 `ShouldStopHook`。
///
/// callback 签名：`callback(ctx: dict) -> bool`
/// 返回 `True` 停止 loop，`False` 强制再跑一轮。
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
#[pyfunction]
fn create_should_stop_hook<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::core::pyhooks::PyHookWrapper>> {
    let hook = crate::core::pyhooks::PyShouldStopHook::new(callback);
    Py::new(
        py,
        crate::core::pyhooks::PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::ShouldStop(Arc::new(hook)),
        },
    )
    .map(|p| p.into_bound(py))
}

/// 从 Python callable 创建一个 `BeforeCompactHook`。
///
/// callback 签名：`callback(ctx: dict) -> str | dict`
/// 返回 `"proceed"` / `"skip"` / `"compact"` 或 `{"action": "override", "summary": <msg_dict>, "first_kept_entry": <str>}`。
/// `first_kept_entry` 必须是 `ctx["entry_ids"]` 中的一个值。
/// 可选字段 `tokens_before` (默认 `ctx["estimated_tokens"]`) 和 `tokens_after` (默认 0)。
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
#[pyfunction]
fn create_before_compact_hook<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::core::pyhooks::PyHookWrapper>> {
    let hook = crate::core::pyhooks::PyBeforeCompactHook::new(callback);
    Py::new(
        py,
        crate::core::pyhooks::PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::BeforeCompact(Arc::new(hook)),
        },
    )
    .map(|p| p.into_bound(py))
}

/// 从 Python callable 创建一个 `TransformContextHook`。
///
/// callback 签名：`callback(ctx: dict) -> dict`
/// 返回 dict 须含 `system_prompt`（str | None）和 `messages`（list[dict]）。
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
#[pyfunction]
fn create_transform_context_hook<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::core::pyhooks::PyHookWrapper>> {
    let hook = crate::core::pyhooks::PyTransformContextHook::new(callback);
    Py::new(
        py,
        crate::core::pyhooks::PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::TransformContext(Arc::new(hook)),
        },
    )
    .map(|p| p.into_bound(py))
}

/// 从 Python callable 创建一个 `PrepareNextTurnHook`。
///
/// callback 签名：`callback(ctx: dict) -> dict | None`
/// 返回 dict 可含 `model`（str）、`thinking_level`（str）、`temperature`（float | None）、
/// `active_tools`（list[str]）。返回 `None` 表示沿用当前值。
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
#[pyfunction]
fn create_prepare_next_turn_hook<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::core::pyhooks::PyHookWrapper>> {
    let hook = crate::core::pyhooks::PyPrepareNextTurnHook::new(callback);
    Py::new(
        py,
        crate::core::pyhooks::PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::PrepareNextTurn(Arc::new(hook)),
        },
    )
    .map(|p| p.into_bound(py))
}

/// Create a `FinalAnswerValidator` from a Python callable.
///
/// callback signature: `callback(ctx: dict) -> None | str | dict`
/// - None → accept the candidate answer
/// - str → reject with code="rejected", message=<returned str>
/// - dict → reject with code=dict["code"], message=dict["message"]
#[pyfunction]
fn create_final_answer_validator<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::core::pyhooks::PyHookWrapper>> {
    let wrapper = crate::core::pyhooks::PyFinalAnswerValidatorWrapper::new(callback);
    Py::new(
        py,
        crate::core::pyhooks::PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::FinalAnswerValidator(Arc::new(wrapper)),
        },
    )
    .map(|p| p.into_bound(py))
}

/// 从 Python callable 创建一个 `AfterRunHook`。
///
/// callback 签名：`callback() -> None`
/// 在 run 结束、Harness 回到 Idle 后调用。
/// 若 callback 为 `async def`，其 coroutine 将在 `spawn_blocking` 线程上
/// 通过 `asyncio.run()` 执行。
#[pyfunction]
fn create_after_run_hook<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::core::pyhooks::PyHookWrapper>> {
    let hook = crate::core::pyhooks::PyAfterRunHook::new(callback);
    Py::new(
        py,
        crate::core::pyhooks::PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::AfterRun(Arc::new(hook)),
        },
    )
    .map(|p| p.into_bound(py))
}

/// 从 Python callable 创建一个 `OnAbortHook`。
///
/// callback 签名：`callback() -> None`
/// 在 `harness.abort()` 时同步调用。用于置标志或 cancel token，不阻塞。
#[pyfunction]
fn create_on_abort_hook<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::core::pyhooks::PyHookWrapper>> {
    let hook = crate::core::pyhooks::PyOnAbortHook::new(callback);
    Py::new(
        py,
        crate::core::pyhooks::PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::OnAbort(Arc::new(hook)),
        },
    )
    .map(|p| p.into_bound(py))
}

/// 从 Python callable 创建一个 `ProviderErrorHook`。
///
/// callback 签名：`callback(ctx: dict) -> str | None`
/// 返回 `"retry"`（同轮重试）/ `"surface"` / `None`（默认原样上抛）。
/// ctx 中的 `context` / `new_messages` 为只读快照。
#[pyfunction]
fn create_provider_error_hook<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, crate::core::pyhooks::PyHookWrapper>> {
    let hook = crate::core::pyhooks::PyProviderErrorHook::new(callback);
    Py::new(
        py,
        crate::core::pyhooks::PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::ProviderError(Arc::new(hook)),
        },
    )
    .map(|p| p.into_bound(py))
}

/// 从 Python 侧配置创建一个 `Plugin`。
///
/// `tools` 为 `create_tool` 创建的 Tool 列表；
/// `hooks` 为 `create_*_hook` 创建的 Hook 列表。
#[pyfunction]
#[pyo3(signature = (name, tools=None, hooks=None))]
fn create_plugin<'py>(
    py: Python<'py>,
    name: &str,
    tools: Option<Vec<Bound<'py, crate::core::pytool::PyToolWrapper>>>,
    hooks: Option<Vec<Bound<'py, crate::core::pyhooks::PyHookWrapper>>>,
) -> PyResult<Bound<'py, crate::core::pyplugin::PyPluginWrapper>> {
    let mut tool_vec: Vec<Arc<dyn llm_harness_types::Tool>> = vec![];
    if let Some(tools) = tools {
        for t in tools {
            let borrowed = t.try_borrow()?;
            tool_vec.push(borrowed.tool.clone());
        }
    }
    let mut hook_vec: Vec<crate::core::pyhooks::HookKind> = vec![];
    if let Some(hooks) = hooks {
        for h in hooks {
            let borrowed = h.try_borrow()?;
            hook_vec.push(borrowed.kind.clone());
        }
    }
    let plugin: Arc<dyn llm_harness_agent::Plugin> = Arc::new(
        crate::core::pyplugin::PyPlugin::new(name.to_string(), tool_vec, hook_vec),
    );
    Py::new(py, crate::core::pyplugin::PyPluginWrapper::new(plugin)).map(|p| p.into_bound(py))
}
