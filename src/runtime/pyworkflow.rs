//! Python callable 包装为 `StepTransitionJudge` 和 `StepExecutor` trait。
//!
//! 验证风险点：workflow trait callback 可从 Python 驱动。
//! 使用与 `PyTool` 相同的模式：`spawn_blocking` + `Python::attach` + `call1`。
//! `StepCtx`/`ExecutorCtx` 的借用字段在进入 `spawn_blocking` 前序列化为
//! owned 数据，避免跨线程借用和 GIL 下 `.await` 死锁。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use futures::future::BoxFuture;
use llm_harness_agent::HarnessBuilder;
use llm_harness_agent::{HarnessHooks, Plugin};
use llm_harness_subagents::{EnvFactory, spawner::JsonlSessionFactory};
use llm_harness_types::{AgentError, CostAggregate, ExecutionEnv, Tool, UnsupportedEnv};
use llm_harness_workflow::lifecycle::task::TaskId;
use llm_harness_workflow::lifecycle::task_store::{JsonlTaskStore, TaskStore, TaskSummary};
use llm_harness_workflow::workflow::engine::{WorkflowEngine, WorkflowEngineConfig, WorkflowEvent};
use llm_harness_workflow::workflow::error::WorkflowError;
use llm_harness_workflow::workflow::executor::{ExecutorCtx, StepExecutor};
use llm_harness_workflow::workflow::judge::{EdgeConditionJudge, StepCtx, StepTransitionJudge};
use llm_harness_workflow::workflow::model::{
    Edge, EdgeCondition, LoopConfig, Step, StepExecutionPolicy, StepRecord, StepResult, Transition,
    Workflow, WorkflowStatus,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

use crate::core::pyagent::runtime;
use crate::core::pybuilder::PyHarnessBuilder;
use crate::core::pyeventstream::PyWaitForExternalEventTool;
use crate::core::pyplugin::PyPluginWrapper;
use crate::core::pyprovider::PyProvider;
use crate::core::pytool::PyToolWrapper;
use crate::runtime::pypricing::PyPricingProvider;
use crate::shared::value_conv::{pyobject_to_value, value_to_pyobject};

// ── PyJudge ─────────────────────────────────────────────────────────────────

/// Python callable 包装为 `StepTransitionJudge`。
///
/// callback 签名：`callback(ctx: dict) -> str`
/// 返回值编码：
/// - `"retry"` → Retry
/// - `"to:<step_id>"` → To(step_id)
/// - `"fail:<reason>"` → Fail { reason }
/// - `"abort:<reason>"` → Abort { reason }
pub struct PyJudge {
    callback: Arc<Py<PyAny>>,
}

impl PyJudge {
    pub fn new(callback: Py<PyAny>) -> Self {
        Self {
            callback: Arc::new(callback),
        }
    }
}

impl StepTransitionJudge for PyJudge {
    fn decide<'a>(&'a self, ctx: &StepCtx<'a>) -> BoxFuture<'a, Transition> {
        let callback = Arc::clone(&self.callback);
        let step_id = ctx.current_step.id().to_string();
        let structured = ctx.last_result.structured.clone();
        let structured_status = ctx.last_result.structured_status.clone();
        let output = ctx.last_result.output.clone();
        let step_count = ctx.step_history.len();
        let retry_count = count_consecutive_retries(ctx.step_history, ctx.current_step.id());
        let tool_calls_count = ctx.last_result.tool_calls_count;

        Box::pin(async move {
            call_python_judge(
                &callback,
                &step_id,
                &output,
                &structured,
                &structured_status,
                step_count,
                retry_count,
                tool_calls_count,
            )
            .await
        })
    }

    fn is_noop(&self) -> bool {
        false
    }
}

/// 将字符串解析为 `Transition`。
fn parse_transition(s: &str) -> Transition {
    let s = s.trim();
    match s {
        "retry" => Transition::Retry,
        s if s.starts_with("to:") => {
            let target = s[3..].trim().to_string();
            Transition::To(target)
        }
        s if s.starts_with("pause:") => Transition::Pause {
            reason: s[6..].trim().to_string(),
        },
        s if s.starts_with("fail:") => Transition::Fail {
            reason: s[5..].trim().to_string(),
        },
        s if s.starts_with("abort:") => Transition::Abort {
            reason: s[6..].trim().to_string(),
        },
        _ => Transition::Abort {
            reason: format!("invalid transition: {s}"),
        },
    }
}

/// 统计 `step_history` 末尾连续属于 `step_id` 的 Retry 记录数。
///
/// 与引擎 `apply_transition` 的 `max_retries` 统计口径一致：从末尾向前
/// 取连续的 `step_id == step_id && transition == Retry` 记录。
///
/// judge 调用时当前步的记录尚未 push（engine 在 `apply_transition`
/// 之后才 push），故：
/// - 首次执行后调用：返回 0
/// - 第一次 Retry 重跑后调用：返回 1
fn count_consecutive_retries(history: &[StepRecord], step_id: &str) -> usize {
    history
        .iter()
        .rev()
        .take_while(|r| r.step_id.as_str() == step_id && matches!(r.transition, Transition::Retry))
        .count()
}

// ── PyExecutor ──────────────────────────────────────────────────────────────

/// 传给 executor 回调 `ctx["emit"]` 的事件发射器。callback 在返回最终结果前
/// 可多次调用 `.text_delta(text)`，用于把 LLM streaming token 实时推送到
/// engine 的事件广播（`WorkflowEngine.subscribe()`），而不必等 callback 整体
/// 执行完毕才有输出。
#[pyclass(name = "ExecutorEmitter")]
pub struct PyExecutorEmitter {
    event_tx: tokio::sync::broadcast::Sender<WorkflowEvent>,
    step_id: String,
}

#[pymethods]
impl PyExecutorEmitter {
    fn text_delta(&self, text: String) {
        let _ = self.event_tx.send(WorkflowEvent::TextDelta {
            step_id: self.step_id.clone(),
            text,
        });
    }
}

/// Python callable 包装为 `StepExecutor`。
///
/// callback 签名：`callback(ctx: dict) -> dict`
/// 返回 dict 期望包含：
/// - `"output"`: str（必填）
/// - `"structured"`: dict（可选，将转为 `serde_json::Value`）
pub struct PyExecutor {
    callback: Arc<Py<PyAny>>,
}

impl PyExecutor {
    pub fn new(callback: Py<PyAny>) -> Self {
        Self {
            callback: Arc::new(callback),
        }
    }
}

impl StepExecutor for PyExecutor {
    fn execute<'a>(&'a self, ctx: &ExecutorCtx<'a>) -> BoxFuture<'a, anyhow::Result<StepResult>> {
        let callback = Arc::clone(&self.callback);
        let step_id = ctx.current_step.id().to_string();
        let step_name = ctx.current_step.name().to_string();
        let config = ctx.current_step.executor_config().cloned();
        let prev_output = ctx.prev_result.map(|r| r.output.clone());
        // clone Arc<Mutex<WorkflowContext>>，在 async 上下文里 lock
        let context = ctx.context.clone();
        let event_tx = ctx.event_tx.clone();

        Box::pin(async move {
            // 在进入 spawn_blocking 前读取 context 快照，避免 GIL 下 .await 死锁
            let context_snapshot = {
                let guard = context.lock().await;
                guard.variables.clone()
            };

            let result = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let cb = callback.bind(py);
                    let dict = PyDict::new(py);
                    dict.set_item("step_id", &step_id)?;
                    dict.set_item("step_name", &step_name)?;
                    if let Some(c) = &config {
                        dict.set_item("config", value_to_pyobject(py, c)?)?;
                    } else {
                        dict.set_item("config", py.None())?;
                    }
                    if let Some(o) = &prev_output {
                        dict.set_item("prev_output", o)?;
                    } else {
                        dict.set_item("prev_output", py.None())?;
                    }

                    let ctx_dict = PyDict::new(py);
                    for (k, v) in &context_snapshot {
                        ctx_dict.set_item(k, value_to_pyobject(py, v)?)?;
                    }
                    dict.set_item("context", ctx_dict)?;

                    let emitter = PyExecutorEmitter {
                        event_tx: event_tx.clone(),
                        step_id: step_id.clone(),
                    };
                    dict.set_item("emit", Py::new(py, emitter)?)?;

                    let raw = cb.call1((dict,))?;
                    let result_dict = raw.cast::<PyDict>()?;
                    let output: String = result_dict
                        .get_item("output")?
                        .ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err("missing 'output' key")
                        })?
                        .extract()?;
                    let structured = result_dict
                        .get_item("structured")
                        .ok()
                        .flatten()
                        .filter(|v| !v.is_none())
                        .map(|v| pyobject_to_value(&v))
                        .transpose()?;

                    Ok::<_, PyErr>((output, structured))
                })
            })
            .await;

            match result {
                Ok(Ok((output, structured))) => Ok(StepResult {
                    output,
                    structured,
                    // 自定义 executor 不参与引擎的结构化提取（StructuredOutputCoordinator），
                    // status 恒为 NotRequired；Python 回调可自行通过 `structured` 字段返回结构化结果。
                    structured_status: Default::default(),
                    tool_calls_count: 0,
                    session_id: String::new(),
                    cost: Default::default(),
                    started_at: None,
                    ended_at: None,
                }),
                Ok(Err(e)) => Err(anyhow::anyhow!("executor callback error: {e}")),
                Err(e) => Err(anyhow::anyhow!("executor join failed: {e}")),
            }
        })
    }
}

// ── Python 包装类 ───────────────────────────────────────────────────────────

/// 持有 `StepTransitionJudge` 的不透明 Python 包装，供 Python 侧引用已注册的 judge。
///
/// 内部以 `Arc<dyn StepTransitionJudge>` 存储，因此可以包装 `PyJudge`（单 callback）、
/// `PyCompositeJudgeInner`（多 handler 分发）等任何实现了 `StepTransitionJudge` 的类型。
#[pyclass(name = "Judge")]
pub struct PyJudgeWrapper {
    pub judge: Arc<dyn StepTransitionJudge>,
}

// ── PyCompositeJudge ────────────────────────────────────────────────────────

/// Python 多 handler 分发 judge。
///
/// 内部维护 `HashMap<step_id, callback>` + 可选 fallback callback。
/// 未注册的 step 如果 workflow 有声明式边 (Expr 或 Label)，引擎会自动注入 `EdgeConditionJudge`
/// 作为 edge_fallback。
pub struct PyCompositeJudgeInner {
    handlers: std::sync::Mutex<HashMap<String, Arc<Py<PyAny>>>>,
    fallback: std::sync::Mutex<Option<Arc<Py<PyAny>>>>,
    edge_fallback: std::sync::Mutex<Option<EdgeConditionJudge>>,
}

impl Default for PyCompositeJudgeInner {
    fn default() -> Self {
        Self::new()
    }
}

impl PyCompositeJudgeInner {
    pub fn new() -> Self {
        Self {
            handlers: std::sync::Mutex::new(HashMap::new()),
            fallback: std::sync::Mutex::new(None),
            edge_fallback: std::sync::Mutex::new(None),
        }
    }

    pub fn set_edge_fallback(&self, judge: EdgeConditionJudge) {
        *self.edge_fallback.lock().unwrap() = Some(judge);
    }
}

impl StepTransitionJudge for PyCompositeJudgeInner {
    fn decide<'a>(&'a self, ctx: &StepCtx<'a>) -> BoxFuture<'a, Transition> {
        let step_id = ctx.current_step.id().to_string();

        // 1. Try registered handler (async — calls Python)
        //    用 Arc::clone 避免 Py<PyAny>::clone（需 GIL attached）。
        if let Some(cb) = self.handlers.lock().unwrap().get(&step_id).cloned() {
            let structured = ctx.last_result.structured.clone();
            let structured_status = ctx.last_result.structured_status.clone();
            let output = ctx.last_result.output.clone();
            let step_count = ctx.step_history.len();
            let retry_count = count_consecutive_retries(ctx.step_history, ctx.current_step.id());
            let tool_calls_count = ctx.last_result.tool_calls_count;
            return Box::pin(async move {
                call_python_judge(
                    &cb,
                    &step_id,
                    &output,
                    &structured,
                    &structured_status,
                    step_count,
                    retry_count,
                    tool_calls_count,
                )
                .await
            });
        }

        // 2. Try user fallback callback (async — calls Python)
        if let Some(cb) = self.fallback.lock().unwrap().clone() {
            let structured = ctx.last_result.structured.clone();
            let structured_status = ctx.last_result.structured_status.clone();
            let output = ctx.last_result.output.clone();
            let step_count = ctx.step_history.len();
            let retry_count = count_consecutive_retries(ctx.step_history, ctx.current_step.id());
            let tool_calls_count = ctx.last_result.tool_calls_count;
            return Box::pin(async move {
                call_python_judge(
                    &cb,
                    &step_id,
                    &output,
                    &structured,
                    &structured_status,
                    step_count,
                    retry_count,
                    tool_calls_count,
                )
                .await
            });
        }

        // 3. Try edge condition fallback (sync)
        if let Some(edge_judge) = self.edge_fallback.lock().unwrap().clone() {
            let transition = edge_judge.decide_sync(ctx);
            return Box::pin(async move { transition });
        }

        // 4. No handler
        let reason = format!("no handler registered for step '{}'", step_id);
        Box::pin(async move { Transition::Abort { reason } })
    }

    fn is_noop(&self) -> bool {
        false
    }
}

/// Python wrapper for composite judge. Exposes `.on()` and `.fallback()`.
#[pyclass(name = "CompositeJudge")]
pub struct PyCompositeJudge {
    inner: Arc<PyCompositeJudgeInner>,
}

impl Default for PyCompositeJudge {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyCompositeJudge {
    /// Register a per-step routing handler.
    ///
    /// callback signature: `callback(ctx: dict) -> str`
    /// Return value encoding: "to:<step_id>", "retry", "fail:<reason>", "abort:<reason>"
    fn on(&self, step: &str, callback: Py<PyAny>) -> PyResult<()> {
        self.inner
            .handlers
            .lock()
            .unwrap()
            .insert(step.to_string(), Arc::new(callback));
        Ok(())
    }

    /// Set a fallback handler for steps without a registered `.on()` handler.
    fn fallback(&self, callback: Py<PyAny>) -> PyResult<()> {
        *self.inner.fallback.lock().unwrap() = Some(Arc::new(callback));
        Ok(())
    }

    fn __repr__(&self) -> String {
        let count = self.inner.handlers.lock().unwrap().len();
        format!("CompositeJudge(handlers={})", count)
    }
}

impl PyCompositeJudge {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(PyCompositeJudgeInner::new()),
        }
    }

    pub fn inner(&self) -> Arc<PyCompositeJudgeInner> {
        self.inner.clone()
    }
}
/// Shared helper: call a Python judge callback and parse the transition.
#[allow(clippy::too_many_arguments)]
async fn call_python_judge(
    callback: &Arc<Py<PyAny>>,
    step_id: &str,
    output: &str,
    structured: &Option<Value>,
    structured_status: &llm_harness_workflow::workflow::model::StructuredStatus,
    step_count: usize,
    retry_count: usize,
    tool_calls_count: u32,
) -> Transition {
    let cb = Arc::clone(callback);
    let structured = structured.clone();
    let structured_status = structured_status.clone();
    let output = output.to_string();
    let step_id = step_id.to_string();

    let result = tokio::task::spawn_blocking(move || {
        Python::attach(|py| {
            let cb = cb.bind(py);
            let dict = PyDict::new(py);
            dict.set_item("step_id", &step_id)?;
            dict.set_item("output", &output)?;
            dict.set_item("step_count", step_count)?;
            dict.set_item("retry_count", retry_count)?;
            dict.set_item("tool_calls_count", tool_calls_count)?;
            if let Some(s) = &structured {
                dict.set_item("structured", value_to_pyobject(py, s)?)?;
            } else {
                dict.set_item("structured", py.None())?;
            }
            dict.set_item(
                "structured_status",
                structured_status_str(&structured_status),
            )?;
            let raw = cb.call1((dict,))?;
            let transition_str: String = raw.extract()?;
            Ok::<_, PyErr>(transition_str)
        })
    })
    .await;

    match result {
        Ok(Ok(s)) => parse_transition(&s),
        Ok(Err(e)) => Transition::Abort {
            reason: format!("judge callback error: {e}"),
        },
        Err(e) => Transition::Abort {
            reason: format!("judge join failed: {e}"),
        },
    }
}

/// 持有 `StepExecutor` 的不透明 Python 包装，供 Python 侧引用已注册的 executor。
///
/// 内部以 `Arc<dyn StepExecutor>` 存储，因此可以包装 `PyExecutor`（Python callback）、
/// `ShellExecutor`、`HttpCallExecutor` 等任何实现了 `StepExecutor` 的类型。
#[pyclass(name = "Executor")]
pub struct PyExecutorWrapper {
    pub executor: Arc<dyn StepExecutor>,
}

/// Parse optional `policy` key from a step dict into `StepExecutionPolicy`.
///
/// Expected format:
/// ```python
/// {"policy": {"max_attempts": 3, "retry_backoff_ms": 2000, "timeout_ms": 120000}}
/// ```
fn parse_step_policy(step_dict: &Bound<'_, PyDict>) -> PyResult<Option<StepExecutionPolicy>> {
    let policy_val = match step_dict.get_item("policy")? {
        Some(v) if !v.is_none() => v,
        _ => return Ok(None),
    };
    let policy_dict = policy_val.cast::<PyDict>()?;
    let mut policy = StepExecutionPolicy::default();
    if let Some(v) = policy_dict
        .get_item("max_attempts")?
        .filter(|v| !v.is_none())
    {
        policy.max_attempts = Some(v.extract::<u32>()?);
    }
    if let Some(v) = policy_dict
        .get_item("retry_backoff_ms")?
        .filter(|v| !v.is_none())
    {
        policy.retry_backoff_ms = Some(v.extract::<u64>()?);
    }
    if let Some(v) = policy_dict.get_item("timeout_ms")?.filter(|v| !v.is_none()) {
        policy.timeout_ms = Some(v.extract::<u64>()?);
    }
    Ok(Some(policy))
}

// ── dict_to_workflow ────────────────────────────────────────────────────────

/// 将 Python dict 解析为 `Workflow` 结构。
///
/// 期望格式：
/// ```python
/// {
///     "entry_step": "step1",
///     "steps": [
///         {"id": "step1", "name": "Step 1", "prompt": "...", "allowed_tools": [...]},
///         # 或 executor step:
///         {"id": "step2", "name": "Exec", "executor": "exec_name", "executor_config": {...}},
///     ],
///     "edges": [{"from": "step1", "to": "step2"}],
/// }
/// ```
fn dict_to_workflow(dict: &Bound<'_, PyDict>) -> PyResult<Workflow> {
    // Support two input formats:
    //   1. "steps" + "edges" + "entry_step" (native Workflow dict)
    //   2. "stages" (declarative pipeline YAML, each stage has next_on_* routes)
    // Format 2 is converted to steps + edges internally, eliminating the need
    // for Python-side PipelineConfig + pipeline_to_workflow_dict converter.
    if let Some(stages_val) = dict.get_item("stages")?
        && !stages_val.is_none()
    {
        return stages_to_workflow(dict);
    }

    let entry_step: String = dict
        .get_item("entry_step")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'entry_step'"))?
        .extract()?;

    let steps_list = dict
        .get_item("steps")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'steps'"))?;
    let steps_seq = steps_list.cast::<pyo3::types::PyList>()?;

    let mut steps = Vec::with_capacity(steps_seq.len());
    for item in steps_seq.iter() {
        let step_dict = item.cast::<PyDict>()?;
        let id: String = step_dict
            .get_item("id")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing step 'id'"))?
            .extract()?;
        let name: String = step_dict
            .get_item("name")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing step 'name'"))?
            .extract()?;

        // 判断是 LLM step 还是 executor step
        if let Some(executor_name_val) = step_dict.get_item("executor")? {
            // Executor step
            let executor_name: String = executor_name_val.extract()?;
            let config = step_dict
                .get_item("executor_config")?
                .filter(|v| !v.is_none())
                .map(|v| pyobject_to_value(&v))
                .transpose()?;
            let mut step = Step::executor(id, name, executor_name, config);
            if let Some(policy) = parse_step_policy(step_dict)? {
                step = step.with_policy(policy);
            }
            steps.push(step);
        } else {
            // LLM step
            let prompt: String = step_dict
                .get_item("prompt")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing step 'prompt'"))?
                .extract()?;
            let allowed_tools: Vec<String> = step_dict
                .get_item("allowed_tools")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_default();
            let structured: Option<bool> = step_dict
                .get_item("structured")?
                .filter(|v| !v.is_none())
                .and_then(|v| v.extract::<bool>().ok());
            let mut step = Step::llm(id, name, prompt, allowed_tools).with_structured(structured);
            if let Some(policy) = parse_step_policy(step_dict)? {
                step = step.with_policy(policy);
            }
            steps.push(step);
        }
    }

    let edges_list = dict
        .get_item("edges")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'edges'"))?;
    let edges_seq = edges_list.cast::<pyo3::types::PyList>()?;

    let mut edges = Vec::with_capacity(edges_seq.len());
    for item in edges_seq.iter() {
        let edge_dict = item.cast::<PyDict>()?;
        let from: String = edge_dict
            .get_item("from")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing edge 'from'"))?
            .extract()?;
        let to: String = edge_dict
            .get_item("to")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing edge 'to'"))?
            .extract()?;
        // Parse optional "condition" key. A string is treated as a legacy
        // label for custom judges (EdgeCondition::Label). A dict is treated
        // as a declarative ConditionExpr (EdgeCondition::Expr), deserialized
        // via serde_json for parity with the Rust model.
        let condition = if let Some(cond_val) = edge_dict.get_item("condition")? {
            if cond_val.is_none() {
                None
            } else if let Ok(s) = cond_val.extract::<String>() {
                Some(EdgeCondition::Label(s))
            } else {
                let v = pyobject_to_value(&cond_val)?;
                Some(serde_json::from_value::<EdgeCondition>(v).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("invalid edge condition: {e}"))
                })?)
            }
        } else {
            None
        };
        edges.push(Edge {
            from,
            to,
            condition,
        });
    }

    Ok(Workflow {
        entry_step,
        steps,
        edges,
    })
}

/// Convert a declarative "stages" pipeline dict to a Workflow.
///
/// Each stage is a dict with:
///   - "name": step id (also used as name)
///   - "type": "tool" | "agent" | "checker" | "terminal"
///   - "tool": executor name (for tool/checker types)
///   - "prompt_template": prompt text (for agent type)
///   - "next_on_*": route keys → target stage name
///   - "loop": { max_iterations, target_stage } (optional)
///   - "exit_code": int (for terminal type)
///   - "message": string (for terminal type)
///
/// Non-terminal stages become executor steps (sharing a single executor name
/// dispatched by step_id). Terminal stages become `Step::Terminal` entries
/// (exit_code 0 → Abort/success, non-zero → Fail). Edges to terminal stages
/// are included so the `EdgeConditionJudge` can route to them; `run_step`
/// then short-circuits the terminal step without invoking any executor or LLM.
fn stages_to_workflow(dict: &Bound<'_, PyDict>) -> PyResult<Workflow> {
    let stages_val = dict
        .get_item("stages")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'stages'"))?;
    let stages_seq = stages_val.cast::<pyo3::types::PyList>()?;

    let mut steps = Vec::new();
    let mut edges = Vec::new();
    let mut entry_step: Option<String> = None;

    for item in stages_seq.iter() {
        let stage_dict = item.cast::<PyDict>()?;
        let name: String = stage_dict
            .get_item("name")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing stage 'name'"))?
            .extract()?;
        let stage_type: String = stage_dict
            .get_item("type")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing stage 'type'"))?
            .extract()?;

        if entry_step.is_none() {
            entry_step = Some(name.clone());
        }

        if stage_type == "terminal" {
            let exit_code: i32 = stage_dict
                .get_item("exit_code")?
                .filter(|v| !v.is_none())
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0);
            let message: Option<String> = stage_dict
                .get_item("message")?
                .filter(|v| !v.is_none())
                .map(|v| v.extract())
                .transpose()?;
            steps.push(Step::terminal(
                name.clone(),
                name.clone(),
                exit_code,
                message,
            ));
            continue;
        }

        // All non-terminal stages become executor steps.
        // Use "executor" field if present (per-stage executor override);
        // otherwise default to "eda_executor" (single shared executor that
        // dispatches by step_id). The "tool" field is domain-specific (used
        // by the executor callback to select a tool), NOT the executor name.
        let executor_name: String = stage_dict
            .get_item("executor")?
            .filter(|v| !v.is_none())
            .map(|v| v.extract::<String>())
            .transpose()?
            .unwrap_or_else(|| "eda_executor".to_string());

        let mut step = Step::executor(name.clone(), name.clone(), executor_name, None);

        // Attach loop config if present.
        if let Some(loop_val) = stage_dict.get_item("loop")?
            && !loop_val.is_none()
        {
            let loop_dict = loop_val.cast::<PyDict>()?;
            let max_iterations: u32 = loop_dict
                .get_item("max_iterations")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'max_iterations'"))?
                .extract()?;
            let target_stage: Option<String> = loop_dict
                .get_item("target_stage")?
                .filter(|v| !v.is_none())
                .map(|v| v.extract())
                .transpose()?;

            let mut policy = step.policy().cloned().unwrap_or_default();
            policy.r#loop = Some(LoopConfig {
                max_iterations,
                target_stage,
                exit_route: None, // resolved from next_on_* edges below
            });
            step = step.with_policy(policy);
        }

        steps.push(step);

        // Collect next_on_* routes as edges. Matched by explicit prefix, not
        // by excluding a hand-maintained list of "known non-route" keys —
        // Studio adds per-step metadata fields over time (ui, component, ...)
        // and an exclusion list silently breaks (or, worse, misroutes) every
        // time a new one is added and forgotten here. "ui" being a dict
        // previously crashed this loop with a dict-is-not-a-str TypeError
        // the moment any step had UI config set, precisely because it wasn't
        // in the old exclusion list.
        for key in stage_dict.keys().iter() {
            let key_str: String = key.extract()?;
            let Some(label) = key_str.strip_prefix("next_on_") else {
                continue;
            };
            let target: String = stage_dict
                .get_item(&key_str)?
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "missing route value for '{key_str}'"
                    ))
                })?
                .extract()?;

            edges.push(Edge {
                from: name.clone(),
                to: target,
                condition: Some(EdgeCondition::Label(label.to_string())),
            });
        }
    }

    let entry_step =
        entry_step.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no stages defined"))?;

    Ok(Workflow {
        entry_step,
        steps,
        edges,
    })
}

// ── workflow_event_to_dict ──────────────────────────────────────────────────

/// 将 `WorkflowEvent` 转换为 Python dict。
///
/// 每个 dict 包含 `"type"` 字段标识事件类型。
/// 镜像 `blender-scene-generator/src/server/events.rs` 的序列化逻辑。
fn workflow_event_to_dict(py: Python<'_>, event: &WorkflowEvent) -> PyResult<Py<PyAny>> {
    use llm_harness_workflow::workflow::engine::StepProgress;

    match event {
        WorkflowEvent::StepStarted { step_id, step_name } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "step_started")?;
            dict.set_item("step_id", step_id.clone())?;
            dict.set_item("step_name", step_name.clone())?;
            Ok(dict.into_any().unbind())
        }
        WorkflowEvent::StepFinished { step_id, result } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "step_finished")?;
            dict.set_item("step_id", step_id.clone())?;
            dict.set_item("output", result.output.clone())?;
            match &result.structured {
                Some(v) => dict.set_item("structured", value_to_pyobject(py, v)?)?,
                None => dict.set_item("structured", py.None())?,
            }
            dict.set_item("tool_calls_count", result.tool_calls_count)?;
            dict.set_item("cost", cost_aggregate_to_dict(py, &result.cost)?)?;
            Ok(dict.into_any().unbind())
        }
        WorkflowEvent::Paused { reason } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "paused")?;
            dict.set_item("reason", reason.clone())?;
            Ok(dict.into_any().unbind())
        }
        WorkflowEvent::Resumed => {
            let dict = PyDict::new(py);
            dict.set_item("type", "resumed")?;
            Ok(dict.into_any().unbind())
        }
        WorkflowEvent::Cancelled { reason } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "cancelled")?;
            dict.set_item("reason", reason.clone())?;
            Ok(dict.into_any().unbind())
        }
        WorkflowEvent::Failed { error } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "failed")?;
            dict.set_item("error", error.clone())?;
            Ok(dict.into_any().unbind())
        }
        WorkflowEvent::StepProgress { step_id, progress } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "step_progress")?;
            dict.set_item("step_id", step_id.clone())?;

            let prog_dict = PyDict::new(py);
            match progress {
                StepProgress::ToolCallStart { tool_use_id, name } => {
                    prog_dict.set_item("type", "tool_call_start")?;
                    prog_dict.set_item("tool_use_id", tool_use_id.clone())?;
                    prog_dict.set_item("name", name.clone())?;
                }
                StepProgress::ToolCallEnd { tool_use_id, args } => {
                    prog_dict.set_item("type", "tool_call_end")?;
                    prog_dict.set_item("tool_use_id", tool_use_id.clone())?;
                    prog_dict.set_item("args", args.to_string())?;
                }
                StepProgress::ToolExecutionStart {
                    tool_use_id,
                    tool_name,
                } => {
                    prog_dict.set_item("type", "tool_execution_start")?;
                    prog_dict.set_item("tool_use_id", tool_use_id.clone())?;
                    prog_dict.set_item("tool_name", tool_name.clone())?;
                }
                StepProgress::ToolExecutionEnd {
                    tool_use_id,
                    ok,
                    error,
                } => {
                    prog_dict.set_item("type", "tool_execution_end")?;
                    prog_dict.set_item("tool_use_id", tool_use_id.clone())?;
                    prog_dict.set_item("ok", *ok)?;
                    match error {
                        Some(e) => prog_dict.set_item("error", e.clone())?,
                        None => prog_dict.set_item("error", py.None())?,
                    }
                }
                StepProgress::TurnEnd { index } => {
                    prog_dict.set_item("type", "turn_end")?;
                    prog_dict.set_item("index", *index)?;
                }
                StepProgress::MessageEnd { message_id, kind } => {
                    prog_dict.set_item("type", "message_end")?;
                    prog_dict.set_item("message_id", message_id.clone())?;
                    prog_dict.set_item("kind", format!("{kind:?}"))?;
                }
            }
            dict.set_item("progress", prog_dict)?;
            Ok(dict.into_any().unbind())
        }
        WorkflowEvent::TextDelta { step_id, text } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "text_delta")?;
            dict.set_item("step_id", step_id.clone())?;
            dict.set_item("text", text.clone())?;
            Ok(dict.into_any().unbind())
        }
        WorkflowEvent::ThinkingDelta { step_id, thinking } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "thinking_delta")?;
            dict.set_item("step_id", step_id.clone())?;
            dict.set_item("thinking", thinking.clone())?;
            Ok(dict.into_any().unbind())
        }
    }
}

// ── UnsupportedEnvFactory ───────────────────────────────────────────────────

/// `EnvFactory` 返回 `UnsupportedEnv`（无文件系统/shell 能力）。
///
/// Tools needing filesystem access should use the bridge pattern
/// (their own `ExecutionEnv`) rather than the execution env provided
/// by the engine.
struct UnsupportedEnvFactory;

impl EnvFactory for UnsupportedEnvFactory {
    fn create(&self, cwd: &std::path::Path) -> Result<Arc<dyn ExecutionEnv>, AgentError> {
        Ok(Arc::new(UnsupportedEnv::with_working_dir(
            cwd.to_path_buf(),
        )))
    }
}

// ── ExecutionEnv 暴露 ──────────────────────────────────────────────────────

/// Python 侧不透明的 `ExecutionEnv` 包装。
///
/// 通过 `create_os_env(working_dir)` 创建，承载真实 OS 文件系统与 shell
/// 执行能力。传入 `WorkflowEngine(workflow, provider, model, judge, env=...)`
/// 后，引擎内 `ShellExecutor` / `HttpCallExecutor` 等执行器即可调用真实命令。
#[pyclass(name = "ExecutionEnv")]
pub struct PyEnvWrapper {
    pub env: Arc<dyn ExecutionEnv>,
}

#[pymethods]
impl PyEnvWrapper {
    fn __repr__(&self) -> String {
        format!("ExecutionEnv(working_dir={:?})", self.env.working_dir())
    }
}

impl PyEnvWrapper {
    pub fn new(env: Arc<dyn ExecutionEnv>) -> Self {
        Self { env }
    }
}

/// 将用户提供的 `Arc<dyn ExecutionEnv>` 包装为 `EnvFactory`。
///
/// `create()` 忽略传入的 cwd（env 在构造时已绑定 working_dir）。
/// 这让 `WorkflowEngine.__new__(env=...)` 能把同一个 env 注入引擎。
struct PyEnvFactory {
    env: Arc<dyn ExecutionEnv>,
}

impl EnvFactory for PyEnvFactory {
    fn create(&self, _cwd: &std::path::Path) -> Result<Arc<dyn ExecutionEnv>, AgentError> {
        Ok(self.env.clone())
    }
}

/// `Arc<PyPlugin>` 的 `Plugin` 适配器。
///
/// `PyPlugin` 实现了 `Plugin`，但 `Arc<PyPlugin>` 没有。
/// `with_step_plugin` 的工厂闭包需要返回 `Box<dyn Plugin>`，
/// 此适配器让 `Arc<dyn Plugin>` 可作为 `Plugin` 使用（`with_step_plugin` 闭包工厂）。
struct PyPluginAdapter(Arc<dyn llm_harness_agent::Plugin>);

impl Plugin for PyPluginAdapter {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn register_tools(&self, tools: &mut Vec<Arc<dyn Tool>>) {
        self.0.register_tools(tools);
    }

    fn register_hooks(&self, hooks: &mut HarnessHooks) {
        self.0.register_hooks(hooks);
    }
}

/// Extract `Arc<dyn StepTransitionJudge>` from a Python judge object.
///
/// Accepts both `Judge` (from `create_judge()`) and `CompositeJudge`
/// (from `create_composite_judge()`). For CompositeJudge, if the workflow
/// has declarative Expr edges and no user fallback is set, auto-injects
/// `EdgeConditionJudge` as the edge fallback.
fn extract_judge(
    _py: Python<'_>,
    judge: &Bound<'_, PyAny>,
    workflow: &Workflow,
) -> PyResult<Arc<dyn StepTransitionJudge>> {
    if let Ok(wrapper) = judge.extract::<PyRef<PyJudgeWrapper>>() {
        return Ok(wrapper.judge.clone());
    }
    if let Ok(composite) = judge.extract::<PyRef<PyCompositeJudge>>() {
        let inner = composite.inner();
        // Auto-inject EdgeConditionJudge as fallback for unregistered steps
        // if the workflow has declarative edges (Expr or Label). Label edges
        // are produced by stages_to_workflow() for declarative pipelines.
        let has_declarative_edges = workflow.edges.iter().any(|e| {
            matches!(
                e.condition,
                Some(EdgeCondition::Expr(_)) | Some(EdgeCondition::Label(_))
            )
        });
        if has_declarative_edges {
            inner.set_edge_fallback(EdgeConditionJudge::from_workflow(workflow));
        }
        return Ok(inner as Arc<dyn StepTransitionJudge>);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "judge must be created by create_judge() or create_composite_judge()",
    ))
}

// ── Type conversion helpers ─────────────────────────────────────────────────

/// 将 `WorkflowStatus` 转为小写字符串。
fn workflow_status_to_str(status: &WorkflowStatus) -> &'static str {
    match status {
        WorkflowStatus::Idle => "idle",
        WorkflowStatus::Running => "running",
        WorkflowStatus::Paused => "paused",
        WorkflowStatus::Succeeded => "succeeded",
        WorkflowStatus::Failed => "failed",
        WorkflowStatus::Cancelled => "cancelled",
    }
}

/// 将 `CostAggregate` 转换为 Python dict。
pub(crate) fn cost_aggregate_to_dict(py: Python<'_>, cost: &CostAggregate) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("total_input_tokens", cost.total_input_tokens)?;
    dict.set_item("total_output_tokens", cost.total_output_tokens)?;
    dict.set_item("total_cache_read_tokens", cost.total_cache_read_tokens)?;
    dict.set_item("total_cache_write_tokens", cost.total_cache_write_tokens)?;
    dict.set_item("total_reasoning_tokens", cost.total_reasoning_tokens)?;
    dict.set_item("total_cost", cost.total_cost)?;

    let by_model = PyDict::new(py);
    for (model, mc) in &cost.by_model {
        let mc_dict = PyDict::new(py);
        mc_dict.set_item("input_tokens", mc.input_tokens)?;
        mc_dict.set_item("output_tokens", mc.output_tokens)?;
        mc_dict.set_item("cache_read_tokens", mc.cache_read_tokens)?;
        mc_dict.set_item("cache_write_tokens", mc.cache_write_tokens)?;
        mc_dict.set_item("reasoning_tokens", mc.reasoning_tokens)?;
        mc_dict.set_item("cost", mc.cost)?;
        mc_dict.set_item("call_count", mc.call_count)?;
        by_model.set_item(model, mc_dict)?;
    }
    dict.set_item("by_model", by_model)?;
    Ok(dict.into_any().unbind())
}

/// 将 `Transition` 转换为 Python dict。
fn transition_to_dict(py: Python<'_>, t: &Transition) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    match t {
        Transition::To(step_id) => {
            dict.set_item("type", "to")?;
            dict.set_item("step_id", step_id.clone())?;
        }
        Transition::Retry => {
            dict.set_item("type", "retry")?;
        }
        Transition::Fail { reason } => {
            dict.set_item("type", "fail")?;
            dict.set_item("reason", reason.clone())?;
        }
        Transition::Abort { reason } => {
            dict.set_item("type", "abort")?;
            dict.set_item("reason", reason.clone())?;
        }
        Transition::Pause { reason } => {
            dict.set_item("type", "pause")?;
            dict.set_item("reason", reason.clone())?;
        }
    }
    Ok(dict.into_any().unbind())
}

/// 将 `StepResult` 转换为 Python dict。
fn step_result_to_dict(py: Python<'_>, r: &StepResult) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("output", r.output.clone())?;
    match &r.structured {
        Some(v) => dict.set_item("structured", value_to_pyobject(py, v)?)?,
        None => dict.set_item("structured", py.None())?,
    }
    dict.set_item("tool_calls_count", r.tool_calls_count)?;
    dict.set_item(
        "structured_status",
        structured_status_str(&r.structured_status),
    )?;
    dict.set_item("session_id", r.session_id.clone())?;
    dict.set_item("cost", cost_aggregate_to_dict(py, &r.cost)?)?;
    Ok(dict.into_any().unbind())
}

/// 将 `StructuredStatus` 转为面向 Python 的稳定字符串值。
///
/// 不依赖 `serde` 对枚举的序列化形状，避免上游 variant 命名变化时
/// 泄漏到 Python 侧。与 README 文档一致：
/// `"not_required"` / `"ok"` / `"failed"`。
fn structured_status_str(
    s: &llm_harness_workflow::workflow::model::StructuredStatus,
) -> &'static str {
    use llm_harness_workflow::workflow::model::StructuredStatus;
    match s {
        StructuredStatus::NotRequired => "not_required",
        StructuredStatus::Ok => "ok",
        StructuredStatus::Failed => "failed",
    }
}

/// 将 `StepRecord` 转换为 Python dict。
fn step_record_to_dict(py: Python<'_>, record: &StepRecord) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("step_id", record.step_id.clone())?;
    dict.set_item("started_at", record.started_at.to_rfc3339())?;
    dict.set_item("ended_at", record.ended_at.to_rfc3339())?;
    match &record.result {
        Some(r) => dict.set_item("result", step_result_to_dict(py, r)?)?,
        None => dict.set_item("result", py.None())?,
    }
    dict.set_item("transition", transition_to_dict(py, &record.transition)?)?;
    Ok(dict.into_any().unbind())
}

/// 将 `WorkflowError` 映射为类型化的 Python 异常。
///
/// - `Validation` → `ValueError`（workflow 定义不合法）
/// - `WorkflowNotFound` / `ExecutorNotFound` → `KeyError`（查找缺失）
/// - `Paused` → `RuntimeError`（暂停是控制流，非错误，但仍需向上传播）
/// - 其余 → `RuntimeError`
fn workflow_error_to_pyerr(e: WorkflowError) -> PyErr {
    crate::shared::pyerror::workflow_error_to_pyerr(e)
}
// ── PyWorkflowEngine ────────────────────────────────────────────────────────

/// Python 侧的 `WorkflowEngine` 包装类。
///
/// 暴露 fluent API：`with_tool()` / `with_step_plugin()` /
/// `with_executor()` / `run()` / `subscribe()` / `task_id()`。
///
/// 内部持有 `Option<Arc<WorkflowEngine>>`。`with_*` 方法通过 `Arc::try_unwrap`
/// 取得所有权（builder 阶段只有一个引用，必定成功）。`run()` 克隆 `Arc` 后
/// 在 tokio runtime 上执行，原始 `Arc` 保留在 `Option` 中，使得 `pause()` /
/// `resume()` / `cancel()` / `state()` 等 `&self` 方法可在 `run()` 期间从
/// 另一个 Python 线程调用。
#[pyclass(name = "WorkflowEngine")]
pub struct PyWorkflowEngine {
    pub(crate) engine: Option<Arc<WorkflowEngine>>,
}

#[pymethods]
impl PyWorkflowEngine {
    /// 从 workflow dict 构造 `WorkflowEngine`。
    ///
    /// `workflow_dict` 包含 `entry_step`/`steps`/`edges`。
    /// `provider` 提供底层 LLM client；`model` 为模型标识；
    /// `judge` 决定步骤间跳转。
    ///
    /// `env` 可选：由 `create_os_env(working_dir)` 创建的 `ExecutionEnv`。
    /// 提供后，`ShellExecutor` 等执行器可调用真实命令；不提供时引擎使用
    /// `UnsupportedEnv`，`execute_shell` 永远返回错误。
    #[new]
    #[pyo3(signature = (workflow_dict, provider, model, judge, session_base_dir="sessions", env=None), text_signature = "(workflow_dict, provider, model, judge, session_base_dir=\"sessions\", env=None)")]
    fn new(
        py: Python<'_>,
        workflow_dict: &Bound<'_, PyDict>,
        provider: &Bound<'_, PyProvider>,
        model: &str,
        judge: &Bound<'_, PyAny>,
        session_base_dir: &str,
        env: Option<Bound<'_, PyEnvWrapper>>,
    ) -> PyResult<Self> {
        let workflow = dict_to_workflow(workflow_dict)?;
        let client = provider.borrow().client.clone();
        let judge_arc: Arc<dyn StepTransitionJudge> = extract_judge(py, judge, &workflow)?;

        let env_factory: Arc<dyn EnvFactory> = match env {
            Some(wrapper) => {
                let env: Arc<dyn ExecutionEnv> = wrapper.borrow().env.clone();
                Arc::new(PyEnvFactory { env })
            }
            None => Arc::new(UnsupportedEnvFactory),
        };

        let config = WorkflowEngineConfig {
            client,
            model: model.to_string(),
            env_factory,
            session_factory: Arc::new(JsonlSessionFactory),
            session_base_dir: std::path::PathBuf::from(session_base_dir),
            customize_builder: None,
        };

        let engine = py.detach(|| {
            WorkflowEngine::new(workflow, config, judge_arc).map_err(workflow_error_to_pyerr)
        })?;

        Ok(Self {
            engine: Some(Arc::new(engine)),
        })
    }

    /// 从 TaskStore 恢复引擎。classmethod。
    ///
    /// `task_store_dir` 是 `JsonlTaskStore` 的根目录（之前 run() 使用的
    /// `with_task_store(dir)` 指定的路径）。
    /// `task_id` 是要恢复的 task ID（`task-<uuid>` 格式）。
    /// `provider`/`model`/`judge` 与 `new()` 相同。
    ///
    /// `env` 可选：与 `new()` 的 `env` 参数语义一致。若原 workflow 使用了
    /// `ShellExecutor` 等 shell 执行器，恢复时必须传入同一个 env，
    /// 否则 shell 步骤会因 `UnsupportedEnv` 而失败。
    #[allow(clippy::too_many_arguments)]
    #[classmethod]
    #[pyo3(signature = (task_store_dir, task_id, provider, model, judge, session_base_dir="sessions", env=None))]
    fn restore(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        task_store_dir: &str,
        task_id: &str,
        provider: &Bound<'_, PyProvider>,
        model: &str,
        judge: &Bound<'_, PyAny>,
        session_base_dir: &str,
        env: Option<Bound<'_, PyEnvWrapper>>,
    ) -> PyResult<Self> {
        let store = Arc::new(JsonlTaskStore::new(PathBuf::from(task_store_dir)));
        let task_id = TaskId(task_id.to_string());
        let client = provider.borrow().client.clone();
        // For restore, we don't have the workflow dict to auto-inject edge fallback.
        // The user should re-register handlers on a new CompositeJudge if needed.
        let empty_workflow = Workflow {
            entry_step: String::new(),
            steps: vec![],
            edges: vec![],
        };
        let judge_arc: Arc<dyn StepTransitionJudge> = extract_judge(py, judge, &empty_workflow)?;

        let env_factory: Arc<dyn EnvFactory> = match env {
            Some(wrapper) => {
                let env: Arc<dyn ExecutionEnv> = wrapper.borrow().env.clone();
                Arc::new(PyEnvFactory { env })
            }
            None => Arc::new(UnsupportedEnvFactory),
        };

        let config = WorkflowEngineConfig {
            client,
            model: model.to_string(),
            env_factory,
            session_factory: Arc::new(JsonlSessionFactory),
            session_base_dir: std::path::PathBuf::from(session_base_dir),
            customize_builder: None,
        };

        let rt = runtime(py);
        let engine = py
            .detach(move || {
                rt.block_on(async move {
                    WorkflowEngine::restore(store, task_id, config, judge_arc).await
                })
            })
            .map_err(workflow_error_to_pyerr)?;

        Ok(Self {
            engine: Some(Arc::new(engine)),
        })
    }

    /// 从指定 step 恢复并重跑（`--restart-from` 场景）。classmethod。
    ///
    /// 加载 task store 中 task_id 的完整上下文，截断 step_history 中目标 step
    /// 及其下游所有 record，将 current_step 指向 step，status 置 Paused。
    /// 调用方随后调 `run()` 从 step 重新执行。
    ///
    /// `step` 必须在 step_history 中（该步曾执行过），否则返回错误。
    /// context 黑板不回滚，保留当前累积值。
    ///
    /// `env` 可选：与 `new()` 的 `env` 参数语义一致。若原 workflow 使用了
    /// `ShellExecutor` 等 shell 执行器，恢复时必须传入同一个 env。
    #[allow(clippy::too_many_arguments)]
    #[classmethod]
    #[pyo3(signature = (task_store_dir, task_id, step, provider, model, judge, session_base_dir="sessions", env=None))]
    fn restore_from_step(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        task_store_dir: &str,
        task_id: &str,
        step: &str,
        provider: &Bound<'_, PyProvider>,
        model: &str,
        judge: &Bound<'_, PyAny>,
        session_base_dir: &str,
        env: Option<Bound<'_, PyEnvWrapper>>,
    ) -> PyResult<Self> {
        let store = Arc::new(JsonlTaskStore::new(PathBuf::from(task_store_dir)));
        let task_id = TaskId(task_id.to_string());
        let client = provider.borrow().client.clone();
        let empty_workflow = Workflow {
            entry_step: String::new(),
            steps: vec![],
            edges: vec![],
        };
        let judge_arc: Arc<dyn StepTransitionJudge> = extract_judge(py, judge, &empty_workflow)?;

        let env_factory: Arc<dyn EnvFactory> = match env {
            Some(wrapper) => {
                let env: Arc<dyn ExecutionEnv> = wrapper.borrow().env.clone();
                Arc::new(PyEnvFactory { env })
            }
            None => Arc::new(UnsupportedEnvFactory),
        };

        let config = WorkflowEngineConfig {
            client,
            model: model.to_string(),
            env_factory,
            session_factory: Arc::new(JsonlSessionFactory),
            session_base_dir: std::path::PathBuf::from(session_base_dir),
            customize_builder: None,
        };

        let rt = runtime(py);
        let engine = py
            .detach(move || {
                rt.block_on(async move {
                    WorkflowEngine::restore_from_step(
                        store,
                        task_id,
                        step.to_string(),
                        config,
                        judge_arc,
                    )
                    .await
                })
            })
            .map_err(workflow_error_to_pyerr)?;

        Ok(Self {
            engine: Some(Arc::new(engine)),
        })
    }

    /// 注册一个额外 `Tool`。返回 self 以支持链式调用。
    fn with_tool<'a>(
        mut slf: PyRefMut<'a, Self>,
        tool: &Bound<'_, PyToolWrapper>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let arc = slf
            .engine
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine already consumed"))?;
        let engine = Arc::try_unwrap(arc).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "engine is shared (running?); cannot add tool",
            )
        })?;
        let t: Arc<dyn Tool> = tool.borrow().tool.clone();
        slf.engine = Some(Arc::new(engine.with_tool(t)));
        Ok(slf)
    }

    /// 注册一个外部交互 tool（wait / human approval / human input）。返回 self 以支持链式调用。
    fn with_external_tool<'a>(
        mut slf: PyRefMut<'a, Self>,
        tool: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let arc = slf
            .engine
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine already consumed"))?;
        let engine = Arc::try_unwrap(arc).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "engine is shared (running?); cannot add tool",
            )
        })?;
        let t: Arc<dyn Tool> = if let Ok(w) = tool.cast::<PyWaitForExternalEventTool>() {
            w.borrow().tool.clone()
        } else if let Ok(a) = tool.cast::<crate::core::pyeventstream::PyHumanApprovalTool>() {
            a.borrow().tool.clone()
        } else if let Ok(i) = tool.cast::<crate::core::pyeventstream::PyHumanInputTool>() {
            i.borrow().tool.clone()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "expected a wait/approval/input tool",
            ));
        };
        slf.engine = Some(Arc::new(engine.with_tool(t)));
        Ok(slf)
    }

    /// 注入额外 hooks。返回 self 以支持链式调用。
    ///
    /// 通过 `customize_builder` 配置：每步构造 harness 时应用。
    /// 多次调用累加（后续 hooks 追加到已有 customize 闭包之后）。
    fn with_hooks<'a>(
        mut slf: PyRefMut<'a, Self>,
        hooks_list: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let arc = slf
            .engine
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine already consumed"))?;

        let mut engine = Arc::try_unwrap(arc).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "engine is shared (running?); cannot add hooks",
            )
        })?;

        let mut harness_hooks = HarnessHooks::none();
        for item in hooks_list.iter() {
            let wrapper = item.cast::<crate::core::pyhooks::PyHookWrapper>()?;
            let kind = &wrapper.borrow().kind;
            use crate::core::pyhooks::HookKind;
            match kind {
                HookKind::BeforeTurn(h) => harness_hooks.before_turn.push(h.clone()),
                HookKind::AfterTurn(h) => harness_hooks.after_turn.push(h.clone()),
                HookKind::BeforeRun(h) => harness_hooks.before_run.push(h.clone()),
                HookKind::AfterProviderResponse(h) => {
                    harness_hooks.after_provider_response.push(h.clone())
                }
                HookKind::BeforeProviderRequest(h) => {
                    harness_hooks.before_provider_request.push(h.clone())
                }
                HookKind::BeforeToolCall(h) => harness_hooks.before_tool_call.push(h.clone()),
                HookKind::AfterToolCall(h) => harness_hooks.after_tool_call.push(h.clone()),
                HookKind::ShouldStop(h) => harness_hooks.should_stop.push(h.clone()),
                HookKind::BeforeCompact(h) => harness_hooks.before_compact.push(h.clone()),
                HookKind::TransformContext(h) => harness_hooks.transform_context.push(h.clone()),
                HookKind::PrepareNextTurn(h) => harness_hooks.prepare_next_turn.push(h.clone()),
                HookKind::FinalAnswerValidator(h) => {
                    harness_hooks.final_answer_validator.push(h.clone())
                }
                HookKind::AfterRun(h) => harness_hooks.after_run.push(h.clone()),
                HookKind::OnAbort(h) => harness_hooks.on_abort.push(h.clone()),
                HookKind::ProviderError(h) => harness_hooks.provider_error.push(h.clone()),
            }
        }

        let prev = engine.config_customize_builder().clone();
        engine.set_customize_builder(Arc::new(move |b| {
            let b = if let Some(p) = &prev { p(b) } else { b };
            b.hooks(harness_hooks.clone())
        }));
        slf.engine = Some(Arc::new(engine));
        Ok(slf)
    }

    /// 设置每步 LLM 调用的最大输出 token 数。
    ///
    /// 通过 `customize_builder` 配置：每步构造 harness 时应用。
    /// 多次调用累加（后调覆盖先调，语义同 builder last-write-wins）。
    fn with_max_tokens<'a>(
        mut slf: PyRefMut<'a, Self>,
        max_tokens: u32,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let arc = slf
            .engine
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine already consumed"))?;
        let mut engine = Arc::try_unwrap(arc).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "engine is shared (running?); cannot set max_tokens",
            )
        })?;
        let prev = engine.config_customize_builder().clone();
        engine.set_customize_builder(Arc::new(move |b| {
            let b = if let Some(p) = &prev { p(b) } else { b };
            b.max_tokens(max_tokens)
        }));
        slf.engine = Some(Arc::new(engine));
        Ok(slf)
    }
    /// 为 workflow 级所有 LLM step 注入 PricingProvider。
    ///
    /// 通过 `customize_builder` 配置：每个 LLM step 构造 harness 时，
    /// 在共享链上调用 `builder.pricing(provider)`。executor step 不构造
    /// harness，故不受影响。多次调用累加（与 `with_hooks`/`with_max_tokens`
    /// 同链，后调覆盖先调，语义同 builder last-write-wins）。
    #[pyo3(text_signature = "($self, provider)")]
    fn with_pricing<'a>(
        mut slf: PyRefMut<'a, Self>,
        provider: &Bound<'_, PyPricingProvider>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let arc = slf
            .engine
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine already consumed"))?;
        let mut engine = Arc::try_unwrap(arc).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "engine is shared (running?); cannot set pricing",
            )
        })?;
        let p = provider.borrow().provider.clone();
        let prev = engine.config_customize_builder().clone();
        engine.set_customize_builder(Arc::new(move |b| {
            let b = if let Some(p) = &prev { p(b) } else { b };
            b.pricing(p.clone())
        }));
        slf.engine = Some(Arc::new(engine));
        Ok(slf)
    }

    /// 设置每步 LLM 调用的 thinking level。
    ///
    /// 通过 `customize_builder` 配置：每步构造 harness 时应用。
    /// 多次调用累加（后调覆盖先调，语义同 builder last-write-wins）。
    fn with_thinking_level<'a>(
        mut slf: PyRefMut<'a, Self>,
        level: &str,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let arc = slf
            .engine
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine already consumed"))?;
        let mut engine = Arc::try_unwrap(arc).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "engine is shared (running?); cannot set thinking_level",
            )
        })?;
        let parsed = match level.to_ascii_lowercase().as_str() {
            "off" => llm_harness_types::ThinkingLevel::Off,
            "minimal" => llm_harness_types::ThinkingLevel::Minimal,
            "low" => llm_harness_types::ThinkingLevel::Low,
            "medium" => llm_harness_types::ThinkingLevel::Medium,
            "high" => llm_harness_types::ThinkingLevel::High,
            "xhigh" => llm_harness_types::ThinkingLevel::XHigh,
            _ => llm_harness_types::ThinkingLevel::High,
        };
        let prev = engine.config_customize_builder().clone();
        engine.set_customize_builder(Arc::new(move |b| {
            let b = if let Some(p) = &prev { p(b) } else { b };
            b.thinking_level(parsed)
        }));
        slf.engine = Some(Arc::new(engine));
        Ok(slf)
    }

    /// 为指定 step 安装 plugin。返回 self 以支持链式调用。
    fn with_step_plugin<'a>(
        mut slf: PyRefMut<'a, Self>,
        step_id: &str,
        plugin: &Bound<'_, PyPluginWrapper>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let arc = slf
            .engine
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine already consumed"))?;
        let engine = Arc::try_unwrap(arc).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "engine is shared (running?); cannot add step plugin",
            )
        })?;
        let p = plugin.borrow().plugin.clone();
        slf.engine =
            Some(Arc::new(engine.with_step_plugin(step_id, move || {
                Box::new(PyPluginAdapter(p.clone()))
            })));
        Ok(slf)
    }

    /// 为指定 step 安装 per-step builder 定制闭包。返回 self 以支持链式调用。
    ///
    /// `customize` 签名：`customize(builder: HarnessBuilder) -> HarnessBuilder`。
    /// 在共享 `customize_builder`（`with_hooks`/`with_max_tokens`/`with_thinking_level`
    /// 等设置）之后、step plugin 注册之前应用，可覆盖共享设置。
    ///
    /// thinking_level。对未注册的 step 无影响。
    ///
    /// TODO(error-propagation): callback 内 Python 异常当前由 `.expect()`
    /// 转成 panic，沿 tokio → `py.detach` 传回 Python 成为 `PanicException`，
    /// 会丢失原始异常类型。要做结构化错误传播需改 runtime 的
    /// `BuilderCustomize` 签名（`Fn(HarnessBuilder) -> HarnessBuilder` →
    /// 返回 `Result`），跨仓库改动，暂未实施。
    fn with_step_builder<'a>(
        mut slf: PyRefMut<'a, Self>,
        step_id: &str,
        customize: Py<PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let arc = slf
            .engine
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine already consumed"))?;
        let engine = Arc::try_unwrap(arc).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "engine is shared (running?); cannot add step builder",
            )
        })?;
        let customize = Arc::new(customize);
        slf.engine = Some(Arc::new(engine.with_step_builder(
            step_id,
            move |b: HarnessBuilder| {
                // Python callback 在 GIL 下同步调用。customize 闭包在 builder 构造
                // 阶段执行（非热路径），且 callback 通常只调几个 builder 方法，
                // 阻塞时间极短。
                Python::attach(|py| {
                    let py_builder = PyHarnessBuilder::from_builder(b);
                    let py_obj = Py::new(py, py_builder)?;
                    let ret = customize.call1(py, (py_obj,))?;
                    let bound = ret.bind(py);
                    let mut borrowed = bound
                        .cast::<PyHarnessBuilder>()
                        .map_err(|_| {
                            pyo3::exceptions::PyTypeError::new_err(
                                "with_step_builder callback must return a HarnessBuilder",
                            )
                        })?
                        .borrow_mut();
                    borrowed.take_builder().ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(
                            "with_step_builder callback returned a consumed HarnessBuilder",
                        )
                    })
                })
                .expect("with_step_builder: Python callback panicked")
            },
        )));
        Ok(slf)
    }

    /// 注册一个命名 executor。返回 self 以支持链式调用。
    fn with_executor<'a>(
        mut slf: PyRefMut<'a, Self>,
        name: &str,
        executor: &Bound<'_, PyExecutorWrapper>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let arc = slf
            .engine
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine already consumed"))?;
        let engine = Arc::try_unwrap(arc).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "engine is shared (running?); cannot add executor",
            )
        })?;
        let e = executor.borrow().executor.clone();
        slf.engine = Some(Arc::new(engine.with_executor(name, e)));
        Ok(slf)
    }

    /// 设置自定义 TaskStore（JSONL 文件存储）。
    /// `dir` 是所有 task 数据的根目录。
    fn with_task_store<'a>(mut slf: PyRefMut<'a, Self>, dir: &str) -> PyResult<PyRefMut<'a, Self>> {
        let arc = slf
            .engine
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine already consumed"))?;
        let engine = Arc::try_unwrap(arc).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "engine is shared (running?); cannot set task store",
            )
        })?;
        let store = Arc::new(JsonlTaskStore::new(PathBuf::from(dir)));
        slf.engine = Some(Arc::new(engine.with_task_store(store)));
        Ok(slf)
    }

    /// 枚举 TaskStore 中的所有 task，返回摘要列表。classmethod。
    ///
    /// `task_store_dir` 是 `JsonlTaskStore` 的根目录（之前 `with_task_store(dir)`
    /// 或 `restore()` 使用的路径）。
    ///
    /// 返回 `list[dict]`，每个 dict 包含：
    /// - `task_id` (str): task 标识
    /// - `status` (str): 生命周期状态（idle/running/paused/succeeded/failed/cancelled）
    /// - `current_step` (str): 当前步骤
    /// - `step_count` (int): 已完成步骤数
    /// - `started_at` (str | None): 启动时间（ISO 8601）
    /// - `ended_at` (str | None): 结束时间
    /// - `reason` (str | None): 暂停/失败/取消原因
    /// - `planned_by` (str | None): 规划此 workflow 的 task ID
    ///
    /// 结果按 `started_at` 降序排列（最新的在前）。
    #[classmethod]
    #[pyo3(text_signature = "(task_store_dir)")]
    fn list_tasks(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        task_store_dir: &str,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let store = JsonlTaskStore::new(PathBuf::from(task_store_dir));
        let rt = runtime(py);
        let summaries: Vec<TaskSummary> =
            crate::shared::pyerror::detach_catch_panic_result(py, move || {
                rt.block_on(async move { store.list_tasks().await })
            })?;
        let mut result = Vec::with_capacity(summaries.len());
        for s in &summaries {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("task_id", &s.task_id.0)?;
            dict.set_item("status", workflow_status_to_str(&s.status))?;
            dict.set_item("current_step", &s.current_step)?;
            dict.set_item("step_count", s.step_count)?;
            let started_at: Py<PyAny> = s
                .started_at
                .map(|dt| {
                    dt.to_rfc3339()
                        .into_pyobject(py)
                        .unwrap()
                        .into_any()
                        .unbind()
                })
                .unwrap_or_else(|| py.None());
            dict.set_item("started_at", started_at)?;
            let ended_at: Py<PyAny> = s
                .ended_at
                .map(|dt| {
                    dt.to_rfc3339()
                        .into_pyobject(py)
                        .unwrap()
                        .into_any()
                        .unbind()
                })
                .unwrap_or_else(|| py.None());
            dict.set_item("ended_at", ended_at)?;
            let reason: Py<PyAny> = s
                .reason
                .as_ref()
                .map(|r| r.as_str().into_pyobject(py).unwrap().into_any().unbind())
                .unwrap_or_else(|| py.None());
            dict.set_item("reason", reason)?;
            let planned_by: Py<PyAny> = s
                .planned_by
                .as_ref()
                .map(|t| t.0.as_str().into_pyobject(py).unwrap().into_any().unbind())
                .unwrap_or_else(|| py.None());
            dict.set_item("planned_by", planned_by)?;
            result.push(dict.into_any().unbind());
        }
        Ok(result)
    }

    /// 设置步骤数上限。超过 → Failed。
    ///
    /// 语义：`step_history.len()` 超过 `max` 时整个 workflow 置为 Failed。
    /// 此值是 workflow 级总护栏——包含所有 step（含 Retry 重跑）。
    /// 默认 100。与 `with_max_retries` 独立：retry 受两者共同约束。
    fn with_max_steps<'a>(mut slf: PyRefMut<'a, Self>, max: usize) -> PyResult<PyRefMut<'a, Self>> {
        let arc = slf
            .engine
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine already consumed"))?;
        let engine = Arc::try_unwrap(arc).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "engine is shared (running?); cannot set max_steps",
            )
        })?;
        slf.engine = Some(Arc::new(engine.with_max_steps(max)));
        Ok(slf)
    }

    /// 设置连续 Retry 上限。超过 → Failed。
    ///
    /// 语义（per-step，非 workflow 级）：当 judge 对当前 step 连续返回
    /// `Retry` 的次数超过 `max` 时，workflow 置为 Failed。`max_retries=N`
    /// 允许 N 次 Retry，第 N+1 次触发 Failed（不含原始执行）。
    ///
    /// judge 仍会在每次 Retry 后被调用——engine 不自动吞重试，judge
    /// 需自行决定是否继续 Retry。可在 judge 中读 `ctx["retry_count"]`
    /// 获取当前 step 的连续 Retry 次数（0 = 首次执行后）。
    ///
    /// 与 `StepExecutionPolicy.max_attempts` 独立：最坏情况单步执行次数
    /// = `max_retries × max_attempts`。`max_steps` 是最终兜底。
    /// 默认 5。
    fn with_max_retries<'a>(
        mut slf: PyRefMut<'a, Self>,
        max: usize,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let arc = slf
            .engine
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine already consumed"))?;
        let engine = Arc::try_unwrap(arc).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "engine is shared (running?); cannot set max_retries",
            )
        })?;
        slf.engine = Some(Arc::new(engine.with_max_retries(max)));
        Ok(slf)
    }

    /// 设置共享 context 变量（KV 黑板）。executor 可通过 `ExecutorCtx` 读取。
    fn set_context_variable(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine not available"))?;
        let json_val = pyobject_to_value(value)?;
        let ctx = engine.context();
        let mut guard = ctx.try_lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to lock workflow context: {e}"
            ))
        })?;
        guard.variables.insert(key.to_string(), json_val);
        Ok(())
    }

    /// 读取共享 context 变量。返回 None 如果 key 不存在。
    fn get_context_variable(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine not available"))?;
        let ctx = engine.context();
        let guard = ctx.try_lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to lock workflow context: {e}"
            ))
        })?;
        match guard.variables.get(key) {
            Some(val) => value_to_pyobject(py, val),
            None => Ok(py.None()),
        }
    }

    /// 启动/恢复执行。阻塞直到 workflow 完成。
    ///
    /// 释放 GIL 后在 tokio runtime 上运行 `engine.run()`。
    /// `run()` 期间可从另一个 Python 线程调用 `pause()` / `cancel()` 等。
    fn run(&self, py: Python<'_>) -> PyResult<()> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine not available"))?;
        let engine_clone = engine.clone();
        let rt = runtime(py);
        crate::shared::pyerror::block_on_with_signal_check(
            py,
            rt,
            async move { engine_clone.run().await.map_err(workflow_error_to_pyerr) },
            200,
        )?;
        Ok(())
    }

    /// 返回当前 task ID（`task-<uuid>` 格式）。
    fn task_id(&self) -> PyResult<String> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine not available"))?;
        Ok(engine.task_id().0.clone())
    }

    /// 查询当前 workflow 状态。
    /// 返回字符串："idle" / "running" / "paused" / "succeeded" / "failed" / "cancelled"。
    fn state(&self, py: Python<'_>) -> PyResult<String> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine not available"))?;
        let rt = runtime(py);
        let status = crate::shared::pyerror::detach_catch_panic(py, move || {
            rt.block_on(async move { engine.state().await })
        })?;
        Ok(workflow_status_to_str(&status).to_string())
    }

    /// 返回当前执行步骤 ID。
    fn current_step(&self, py: Python<'_>) -> PyResult<String> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine not available"))?;
        let rt = runtime(py);
        let step = crate::shared::pyerror::detach_catch_panic(py, move || {
            rt.block_on(async move { engine.current_step().await })
        })?;
        Ok(step)
    }

    /// 返回步骤历史（审计链），list[dict]。
    fn step_history(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine not available"))?;
        let rt = runtime(py);
        let history = crate::shared::pyerror::detach_catch_panic(py, move || {
            rt.block_on(async move { engine.step_history().await })
        })?;
        let mut result = Vec::with_capacity(history.len());
        for record in &history {
            result.push(step_record_to_dict(py, record)?);
        }
        Ok(result)
    }

    /// 非阻塞：请求暂停。`run()` 在步边界检查并消费。
    /// `reason` 是暂停原因（人类可读）。
    fn pause(&self, reason: &str) -> PyResult<()> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine not available"))?;
        engine.pause(reason).map_err(workflow_error_to_pyerr)
    }

    /// 恢复已暂停或失败的 task。将状态重置为 Paused，随后 `run()` 可继续。
    fn resume(&self, py: Python<'_>) -> PyResult<()> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine not available"))?;
        let rt = runtime(py);
        let result = crate::shared::pyerror::detach_catch_panic(py, move || {
            rt.block_on(async move { engine.resume().await })
        })?;
        result.map_err(workflow_error_to_pyerr)
    }

    /// 取消正在运行的 workflow。当前步记为 Abort，状态置为 Cancelled。
    fn cancel(&self, py: Python<'_>, reason: &str) -> PyResult<()> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine not available"))?;
        let reason = reason.to_string();
        let rt = runtime(py);
        let result = crate::shared::pyerror::detach_catch_panic(py, move || {
            rt.block_on(async move { engine.cancel(&reason).await })
        })?;
        result.map_err(workflow_error_to_pyerr)
    }

    /// 创建检查点（append-only）。`description` 人类可读，`payload` 任意 JSON 可序列化对象。
    fn checkpoint(
        &self,
        py: Python<'_>,
        description: &str,
        payload: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine not available"))?;
        let payload_val = pyobject_to_value(payload)?;
        let description = description.to_string();
        let rt = runtime(py);
        let result = crate::shared::pyerror::detach_catch_panic(py, move || {
            rt.block_on(async move { engine.checkpoint(&description, payload_val).await })
        })?;
        result.map_err(workflow_error_to_pyerr)
    }

    /// 返回累计 token/成本统计，dict。
    fn total_cost(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine not available"))?;
        let rt = runtime(py);
        let cost = crate::shared::pyerror::detach_catch_panic(py, move || {
            rt.block_on(async move { engine.total_cost().await })
        })?;
        cost_aggregate_to_dict(py, &cost)
    }

    /// 订阅引擎事件流，返回 `WorkflowEventIterator`。
    #[pyo3(signature = (timeout_ms=5000, max_consecutive_timeouts=1))]
    fn subscribe(
        &self,
        py: Python<'_>,
        timeout_ms: u64,
        max_consecutive_timeouts: u32,
    ) -> PyResult<Py<PyWorkflowEventIterator>> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine not available"))?;
        let rx = engine.subscribe();
        let handle = runtime(py).handle().clone();
        let iter = PyWorkflowEventIterator::new(rx, timeout_ms, max_consecutive_timeouts, handle);
        Py::new(py, iter)
    }

    /// Context manager entry.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Context manager exit: cancels any running workflow.
    fn __exit__(
        &mut self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if let Some(engine) = self.engine.as_ref() {
            let engine_clone = engine.clone();
            let py = _exc_type.py();
            let rt = runtime(py);
            let _ = crate::shared::pyerror::detach_catch_panic(py, move || {
                rt.block_on(async { engine_clone.cancel("context manager exit").await })
            })?;
        }
        Ok(false)
    }

    /// 返回引擎状态摘要。
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine not available"))?;
        let task_id = engine.task_id().0.clone();
        let engine_clone = engine.clone();
        let rt = runtime(py);
        let state = crate::shared::pyerror::detach_catch_panic(py, move || {
            rt.block_on(async { workflow_status_to_str(&engine_clone.state().await).to_string() })
        })?;
        Ok(format!(
            "WorkflowEngine(task_id={}, state={})",
            task_id, state
        ))
    }
}

// ── PyWorkflowEventIterator ─────────────────────────────────────────────────

/// Python 迭代器，包装 `broadcast::Receiver<WorkflowEvent>`。
///
/// 与 `PyHarnessEventIterator` 对称，但处理 `WorkflowEvent` 类型。
/// 释放 GIL 后阻塞等待事件，超时或 channel 关闭时返回 `None`。
#[pyclass(name = "WorkflowEventIterator")]
pub struct PyWorkflowEventIterator {
    rx: Option<tokio::sync::broadcast::Receiver<WorkflowEvent>>,
    timeout_ms: u64,
    max_consecutive_timeouts: u32,
    consecutive_timeouts: u32,
    handle: tokio::runtime::Handle,
}

impl PyWorkflowEventIterator {
    pub fn new(
        rx: tokio::sync::broadcast::Receiver<WorkflowEvent>,
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
impl PyWorkflowEventIterator {
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
                let dict = workflow_event_to_dict(py, &event)?;
                Ok(Some(dict))
            }
            Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(n))) => {
                let warning = PyDict::new(py);
                warning.set_item("type", "lagged")?;
                warning.set_item("skipped", n)?;
                Ok(Some(warning.into_any().unbind()))
            }
            Ok(Err(tokio::sync::broadcast::error::RecvError::Closed)) => Ok(None),
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

#[cfg(test)]
mod tests {
    use super::{dict_to_workflow, parse_step_policy};
    use llm_harness_workflow::workflow::model::{
        EdgeCondition, Step, StepExecutionPolicy, Workflow,
    };
    use pyo3::types::PyDictMethods;
    use pyo3::{Py, Python};

    /// Build an owned Python dict via inline source.
    /// The source must bind a variable named `d` to the desired dict.
    fn make_dict(source: &str) -> Py<pyo3::types::PyDict> {
        Python::attach(|py| {
            let locals = pyo3::types::PyDict::new(py);
            py.run(
                std::ffi::CString::new(source).unwrap().as_c_str(),
                None,
                Some(&locals),
            )
            .unwrap();
            let obj = locals
                .get_item("d")
                .unwrap()
                .expect("missing 'd' in source");
            obj.cast::<pyo3::types::PyDict>().unwrap().clone().unbind()
        })
    }

    fn policy_of(step: &Step) -> Option<&StepExecutionPolicy> {
        step.policy()
    }

    /// Parse a step-level dict's policy via the real code path.
    fn parse_policy(source: &str) -> Option<StepExecutionPolicy> {
        let dict = make_dict(source);
        Python::attach(|py| parse_step_policy(dict.bind(py))).unwrap()
    }

    /// Parse a full workflow dict via the real code path.
    fn parse_workflow(source: &str) -> Workflow {
        let dict = make_dict(source);
        Python::attach(|py| dict_to_workflow(dict.bind(py))).unwrap()
    }

    /// Parse a step-level dict's policy, expecting an error.
    fn parse_policy_err(source: &str) -> String {
        let dict = make_dict(source);
        Python::attach(|py| match parse_step_policy(dict.bind(py)) {
            Ok(v) => format!("expected error, got Ok({:?})", v),
            Err(e) => e.to_string(),
        })
    }

    // ── parse_step_policy unit tests ─────────────────────────────────────

    #[test]
    fn parse_step_policy_full() {
        let policy = parse_policy(
            r#"d = {"policy": {"max_attempts": 3, "retry_backoff_ms": 2000, "timeout_ms": 120000}}"#,
        ).unwrap();
        assert_eq!(policy.max_attempts, Some(3));
        assert_eq!(policy.retry_backoff_ms, Some(2000));
        assert_eq!(policy.timeout_ms, Some(120000));
    }

    #[test]
    fn parse_step_policy_partial_only_sets_provided_fields() {
        let policy = parse_policy(r#"d = {"policy": {"timeout_ms": 5000}}"#).unwrap();
        assert_eq!(policy.timeout_ms, Some(5000));
        assert_eq!(policy.max_attempts, None);
        assert_eq!(policy.retry_backoff_ms, None);
    }

    #[test]
    fn parse_step_policy_explicit_none_is_absent() {
        let opt = parse_policy(r#"d = {"policy": None}"#);
        assert!(opt.is_none(), "explicit None should be treated as absent");
    }

    #[test]
    fn parse_step_policy_missing_key_is_none() {
        let opt = parse_policy(r#"d = {"id": "s1"}"#);
        assert!(opt.is_none(), "missing 'policy' key should yield None");
    }

    #[test]
    fn parse_step_policy_partial_with_none_values() {
        let policy =
            parse_policy(r#"d = {"policy": {"max_attempts": None, "timeout_ms": 5000}}"#).unwrap();
        assert_eq!(policy.max_attempts, None);
        assert_eq!(policy.timeout_ms, Some(5000));
    }

    #[test]
    fn parse_step_policy_non_dict_value_errors() {
        let msg = parse_policy_err(r#"d = {"policy": 3}"#);
        assert!(
            !msg.starts_with("expected error"),
            "non-dict policy must raise: {msg}"
        );
    }

    #[test]
    fn parse_step_policy_wrong_field_type_errors() {
        let msg = parse_policy_err(r#"d = {"policy": {"max_attempts": "lots"}}"#);
        assert!(
            !msg.starts_with("expected error"),
            "wrong field type must raise: {msg}"
        );
    }

    // ── dict_to_workflow integration tests ───────────────────────────────

    #[test]
    fn dict_to_workflow_applies_policy_to_llm_step() {
        let wf = parse_workflow(
            r#"d = {
                "entry_step": "s1",
                "steps": [
                    {"id": "s1", "name": "S1", "prompt": "hi", "policy": {"max_attempts": 3, "timeout_ms": 10000}},
                ],
                "edges": [],
            }"#,
        );
        let step = wf.steps.iter().find(|s| s.id() == "s1").expect("step s1");
        let policy = policy_of(step).expect("policy should be set");
        assert_eq!(policy.max_attempts, Some(3));
        assert_eq!(policy.timeout_ms, Some(10000));
        assert_eq!(policy.retry_backoff_ms, None);
    }

    #[test]
    fn dict_to_workflow_applies_policy_to_executor_step() {
        let wf = parse_workflow(
            r#"d = {
                "entry_step": "s1",
                "steps": [
                    {"id": "s1", "name": "S1", "executor": "sh", "executor_config": {"cmd": "echo"}, "policy": {"max_attempts": 5, "retry_backoff_ms": 500}},
                ],
                "edges": [],
            }"#,
        );
        let step = wf.steps.iter().find(|s| s.id() == "s1").expect("step s1");
        let policy = policy_of(step).expect("policy should be set");
        assert_eq!(policy.max_attempts, Some(5));
        assert_eq!(policy.retry_backoff_ms, Some(500));
        assert_eq!(policy.timeout_ms, None);
    }

    #[test]
    fn dict_to_workflow_no_policy_preserves_default() {
        let wf = parse_workflow(
            r#"d = {
                "entry_step": "s1",
                "steps": [
                    {"id": "s1", "name": "S1", "prompt": "hi"},
                ],
                "edges": [],
            }"#,
        );
        let step = wf.steps.iter().find(|s| s.id() == "s1").expect("step s1");
        assert!(
            policy_of(step).is_none(),
            "absent policy must leave step.policy() as None"
        );
    }

    #[test]
    fn dict_to_workflow_partial_policy_on_llm_step() {
        let wf = parse_workflow(
            r#"d = {
                "entry_step": "s1",
                "steps": [
                    {"id": "s1", "name": "S1", "prompt": "hi", "policy": {"max_attempts": 2}},
                ],
                "edges": [],
            }"#,
        );
        let step = wf.steps.iter().find(|s| s.id() == "s1").expect("step s1");
        let policy = policy_of(step).expect("policy should be set");
        assert_eq!(policy.max_attempts, Some(2));
        assert_eq!(policy.timeout_ms, None);
        assert_eq!(policy.retry_backoff_ms, None);
    }

    // ── stages_to_workflow: non-route step metadata ──────────────────────
    //
    // Regression coverage for a bug where the route-collection loop treated
    // ANY key not on a hand-maintained exclusion list as a next_on_* route,
    // rather than checking the "next_on_" prefix directly. A step with a
    // "ui" field (a dict, not a route-target string — set by senza-studio's
    // Inspector/set_ui_config) crashed with a dict-is-not-a-str TypeError.

    #[test]
    fn stages_to_workflow_ignores_ui_field() {
        let wf = parse_workflow(
            r#"d = {
                "stages": [
                    {"name": "step1", "type": "agent", "prompt_template": "hi",
                     "ui": {"display": "chat", "fields": ["a", "b"]},
                     "next_on_success": "step2"},
                    {"name": "step2", "type": "terminal", "message": "done"},
                ]
            }"#,
        );
        assert_eq!(wf.entry_step, "step1");
        assert_eq!(wf.edges.len(), 1);
        assert_eq!(wf.edges[0].from, "step1");
        assert_eq!(wf.edges[0].to, "step2");
    }

    #[test]
    fn stages_to_workflow_ignores_component_field() {
        let wf = parse_workflow(
            r#"d = {
                "stages": [
                    {"name": "step1", "type": "agent", "component": "approval_flow",
                     "next_on_success": "step2"},
                    {"name": "step2", "type": "terminal"},
                ]
            }"#,
        );
        assert_eq!(wf.edges.len(), 1);
        assert_eq!(wf.edges[0].to, "step2");
    }

    #[test]
    fn stages_to_workflow_still_routes_next_on_fields() {
        let wf = parse_workflow(
            r#"d = {
                "stages": [
                    {"name": "check", "type": "checker",
                     "next_on_pass": "ok", "next_on_fail": "bad"},
                    {"name": "ok", "type": "terminal"},
                    {"name": "bad", "type": "terminal"},
                ]
            }"#,
        );
        assert_eq!(wf.edges.len(), 2);
        let labels: std::collections::HashSet<_> = wf
            .edges
            .iter()
            .map(|e| match &e.condition {
                Some(EdgeCondition::Label(l)) => l.as_str(),
                _ => "",
            })
            .collect();
        assert!(labels.contains("pass"));
        assert!(labels.contains("fail"));
    }
}
