//! Python callable 包装为 `Plugin` trait。
//!
//! Plugin 在 harness 构建时将 Python 侧的 tools 和 hooks 注册到 Rust vec 中。

use std::sync::Arc;

use llm_harness_agent::{HarnessHooks, Plugin};
use llm_harness_types::Tool;
use pyo3::prelude::*;

use crate::core::pyhooks::HookKind;

/// Python 侧的 Plugin 配置。
pub struct PyPlugin {
    name: String,
    tools: Vec<Arc<dyn Tool>>,
    // hooks 按 kind 分类存储，register_hooks 时分发到对应 vec
    before_turn: Vec<Arc<dyn llm_harness_types::BeforeTurnHook>>,
    after_turn: Vec<Arc<dyn llm_harness_types::AfterTurnHook>>,
    before_run: Vec<Arc<dyn llm_harness_types::BeforeRunHook>>,
    should_stop: Vec<Arc<dyn llm_harness_types::ShouldStopHook>>,
    before_tool_call: Vec<Arc<dyn llm_harness_types::BeforeToolCallHook>>,
    after_tool_call: Vec<Arc<dyn llm_harness_types::AfterToolCallHook>>,
    before_compact: Vec<Arc<dyn llm_harness_types::BeforeCompactHook>>,
    transform_context: Vec<Arc<dyn llm_harness_types::TransformContextHook>>,
    prepare_next_turn: Vec<Arc<dyn llm_harness_types::PrepareNextTurnHook>>,
    before_provider_request: Vec<Arc<dyn llm_harness_types::BeforeProviderRequestHook>>,
    after_provider_response: Vec<Arc<dyn llm_harness_types::AfterProviderResponseHook>>,
    final_answer_validator: Vec<Arc<dyn llm_harness_types::FinalAnswerValidator>>,
    after_run: Vec<Arc<dyn llm_harness_types::AfterRunHook>>,
    on_abort: Vec<Arc<dyn llm_harness_types::OnAbortHook>>,
    provider_error: Vec<Arc<dyn llm_harness_types::ProviderErrorHook>>,
}

impl PyPlugin {
    pub fn new(name: String, tools: Vec<Arc<dyn Tool>>, hooks: Vec<HookKind>) -> Self {
        let mut p = Self {
            name,
            tools,
            before_turn: vec![],
            after_turn: vec![],
            before_run: vec![],
            should_stop: vec![],
            before_tool_call: vec![],
            after_tool_call: vec![],
            before_compact: vec![],
            transform_context: vec![],
            prepare_next_turn: vec![],
            before_provider_request: vec![],
            after_provider_response: vec![],
            final_answer_validator: vec![],
            after_run: vec![],
            on_abort: vec![],
            provider_error: vec![],
        };
        for kind in hooks {
            match kind {
                HookKind::BeforeTurn(h) => p.before_turn.push(h),
                HookKind::AfterTurn(h) => p.after_turn.push(h),
                HookKind::BeforeRun(h) => p.before_run.push(h),
                HookKind::ShouldStop(h) => p.should_stop.push(h),
                HookKind::BeforeToolCall(h) => p.before_tool_call.push(h),
                HookKind::AfterToolCall(h) => p.after_tool_call.push(h),
                HookKind::BeforeCompact(h) => p.before_compact.push(h),
                HookKind::TransformContext(h) => p.transform_context.push(h),
                HookKind::PrepareNextTurn(h) => p.prepare_next_turn.push(h),
                HookKind::BeforeProviderRequest(h) => p.before_provider_request.push(h),
                HookKind::AfterProviderResponse(h) => p.after_provider_response.push(h),
                HookKind::FinalAnswerValidator(h) => p.final_answer_validator.push(h),
                HookKind::AfterRun(h) => p.after_run.push(h),
                HookKind::OnAbort(h) => p.on_abort.push(h),
                HookKind::ProviderError(h) => p.provider_error.push(h),
            }
        }
        p
    }
}

impl Plugin for PyPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn register_tools(&self, tools: &mut Vec<Arc<dyn Tool>>) {
        tools.extend(self.tools.iter().cloned());
    }

    fn register_hooks(&self, hooks: &mut HarnessHooks) {
        hooks.before_turn.extend(self.before_turn.iter().cloned());
        hooks.after_turn.extend(self.after_turn.iter().cloned());
        hooks.before_run.extend(self.before_run.iter().cloned());
        hooks.should_stop.extend(self.should_stop.iter().cloned());
        hooks
            .before_tool_call
            .extend(self.before_tool_call.iter().cloned());
        hooks
            .after_tool_call
            .extend(self.after_tool_call.iter().cloned());
        hooks
            .before_compact
            .extend(self.before_compact.iter().cloned());
        hooks
            .transform_context
            .extend(self.transform_context.iter().cloned());
        hooks
            .prepare_next_turn
            .extend(self.prepare_next_turn.iter().cloned());
        hooks
            .before_provider_request
            .extend(self.before_provider_request.iter().cloned());
        hooks
            .after_provider_response
            .extend(self.after_provider_response.iter().cloned());
        hooks
            .final_answer_validator
            .extend(self.final_answer_validator.iter().cloned());
        hooks.after_run.extend(self.after_run.iter().cloned());
        hooks.on_abort.extend(self.on_abort.iter().cloned());
        hooks
            .provider_error
            .extend(self.provider_error.iter().cloned());
    }
}

/// 持有 `Plugin` trait 对象的不透明 Python 包装。
///
/// 可包装 `PyPlugin`（Python 侧 `create_plugin`）或任意 Rust 侧
/// `Plugin` 实现（如 `FsToolsPlugin`）。
#[pyclass(name = "Plugin")]
pub struct PyPluginWrapper {
    pub plugin: Arc<dyn Plugin>,
}

impl PyPluginWrapper {
    /// 从任意 `Plugin` trait 对象构造包装。
    pub fn new(plugin: Arc<dyn Plugin>) -> Self {
        Self { plugin }
    }
}

#[pymethods]
impl PyPluginWrapper {
    /// 返回 plugin 名称。
    #[getter]
    fn name(&self) -> &str {
        self.plugin.name()
    }
}
