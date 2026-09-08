//! `HarnessBuilder` 的 Python 包装。
//!
//! 提供 fluent API 镜像 Rust `HarnessBuilder`：`.system_prompt()`、
//! `.max_tokens()`、`.temperature()`、`.tool()`、`.plugin()`、
//! `.provider()`、`.build()`。
//!
//! `build()` 释放 GIL 后用全局 tokio runtime 执行 async `HarnessBuilder::build`，
//! 返回 `PyAgentHarness`（包装真实 `AgentHarness`）。

use std::path::PathBuf;
use std::sync::Arc;

use llm_harness_agent::HarnessBuilder;
use llm_harness_agent::{CompactionPromptSpec, ModelInfo};
use llm_harness_agent::{Plugin, Skill};
use llm_harness_knowledge::{KnowledgeAccessContext, KnowledgeScope, PrincipalRef};
use llm_harness_loop::config::RetryConfig;
use llm_harness_loop::final_answer::FinalAnswerMode;
use llm_harness_mcp::builder::HarnessBuilderMcpExt;
use llm_harness_types::{ExecutionEnv, StreamOptions, Tool, UnsupportedEnv};
use pyo3::prelude::*;

use crate::core::pyagent::runtime;
use crate::core::pyharness::PyAgentHarness;
use crate::core::pyharness::parse_thinking_level;
use crate::core::pyhooks::PyHookWrapper;
use crate::core::pyplugin::PyPluginWrapper;
use crate::core::pyprovider::PyProvider;
use crate::core::pyresponseformat::PyResponseFormat;
use crate::core::pytool::PyToolWrapper;
use crate::runtime::pybudget::PyBudgetExceededHook;
use crate::runtime::pypricing::PyPricingProvider;
use crate::runtime::pyskills::PySkill;

use crate::runtime::pymcp::{PyMcpManager, PyMcpServerConfig};
use crate::runtime::pyspawn::wire_spawn;
use crate::runtime::pyworkflow::PyEnvWrapper;
/// Python 侧的 `HarnessBuilder`。
///
/// 镜像 Rust `HarnessBuilder` 的 fluent API。fluent 方法以 `PyRefMut`
/// 接收 `self`，修改内部 builder 后返回自身，支持链式调用。
#[pyclass(name = "HarnessBuilder")]
pub struct PyHarnessBuilder {
    builder: Option<HarnessBuilder>,
    /// 可选执行环境；`build()` 时注入。`None` → `UnsupportedEnv`（默认）。
    env: Option<Arc<dyn ExecutionEnv>>,
    /// MCP server 配置列表（name, config），在 build() 时提升为 McpHarnessBuilder。
    mcp_servers: Vec<(String, llm_harness_mcp::config::McpServerConfig)>,
    /// MCP 配置文件路径列表，在 build() 时异步读取。
    mcp_config_files: Vec<PathBuf>,
    /// 外部 McpManager（高级 API），在 build() 时注入。
    mcp_manager: Option<Arc<llm_harness_mcp::manager::McpManager>>,
    /// Spawn 配置（model, client, session_dir），在 build() 时完成 spawn 基础设施粘合。
    spawn_config: Option<SpawnConfig>,
    /// Optional session repo for persistent sessions.
    session_repo: Option<Arc<dyn llm_harness_agent::SessionRepo>>,
    /// Optional session ID to restore an existing session.
    session_id: Option<String>,
    /// Optional per-harness knowledge access context; `None` uses the SDK default.
    knowledge_access: Option<KnowledgeAccessContext>,
}

/// Spawn 配置：`enable_spawn()` 存储的参数，`build()` 时消费。
pub(crate) struct SpawnConfig {
    pub(crate) model: String,
    pub(crate) client: Arc<dyn llm_harness_loop::LlmClient>,
    pub(crate) session_dir: PathBuf,
    /// 并发 sub-agent 上限；`None` = 不限（runtime 默认）。
    pub(crate) max_concurrent: Option<usize>,
}
#[pymethods]
impl PyHarnessBuilder {
    #[new]
    #[pyo3(text_signature = "(model)")]
    fn new(model: &str) -> Self {
        Self {
            builder: Some(HarnessBuilder::new(model)),
            env: None,
            mcp_servers: Vec::new(),
            mcp_config_files: Vec::new(),
            mcp_manager: None,
            spawn_config: None,
            session_repo: None,
            session_id: None,
            knowledge_access: None,
        }
    }

    /// 设置系统提示。重复调用后写覆盖前写。
    fn system_prompt<'a>(mut slf: PyRefMut<'a, Self>, prompt: &str) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            slf.builder = Some(b.system_prompt(Some(prompt.to_string())));
        }
        slf
    }

    /// 设置每次 provider 调用的最大输出 token 数。
    fn max_tokens<'a>(mut slf: PyRefMut<'a, Self>, tokens: u32) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            slf.builder = Some(b.max_tokens(tokens));
        }
        slf
    }

    /// 设置采样温度。`None` 重置为 provider 默认值。
    fn temperature<'a>(mut slf: PyRefMut<'a, Self>, temp: Option<f32>) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            slf.builder = Some(b.temperature(temp));
        }
        slf
    }

    /// 注册一个 `Tool`（来自 `create_tool` 或 `create_web_search_tool` 等）。
    fn tool<'a>(
        mut slf: PyRefMut<'a, Self>,
        tool: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        if let Some(b) = slf.builder.take() {
            let t: Arc<dyn Tool> = if let Ok(w) = tool.extract::<PyRef<'_, PyToolWrapper>>() {
                w.tool.clone()
            } else if let Ok(n) = tool.extract::<PyRef<'_, crate::core::pywebtools::PyNativeTool>>()
            {
                n.tool.clone()
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "expected a Tool (from create_tool) or NativeTool (from create_web_search_tool, etc.)",
                ));
            };
            slf.builder = Some(b.tool(t));
        }
        Ok(slf)
    }

    /// 批量注册多个 `Tool`（来自 `create_tool` 或 `create_web_search_tool` 等）。
    #[pyo3(text_signature = "($self, tools)")]
    fn tools<'a>(
        mut slf: PyRefMut<'a, Self>,
        tools: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        if let Some(mut b) = slf.builder.take() {
            for item in &tools {
                let t: Arc<dyn Tool> = if let Ok(w) = item.extract::<PyRef<'_, PyToolWrapper>>() {
                    w.tool.clone()
                } else if let Ok(n) =
                    item.extract::<PyRef<'_, crate::core::pywebtools::PyNativeTool>>()
                {
                    n.tool.clone()
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "each item must be a Tool or NativeTool",
                    ));
                };
                b = b.tool(t);
            }
            slf.builder = Some(b);
        }
        Ok(slf)
    }

    /// 安装一个 `Plugin`（来自 `create_plugin`），累积其 tools/hooks/skills。
    fn plugin<'a>(
        mut slf: PyRefMut<'a, Self>,
        plugin: &Bound<'_, PyPluginWrapper>,
    ) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            let p = &plugin.borrow().plugin;
            slf.builder = Some(b.install(p.as_ref()));
        }
        slf
    }

    /// 注册一个 LLM provider，匹配 `pattern` 的 model 会路由到此 provider。
    fn provider<'a>(
        mut slf: PyRefMut<'a, Self>,
        pattern: &str,
        provider: &Bound<'_, PyProvider>,
    ) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            let client = provider.borrow().client.clone();
            slf.builder = Some(b.provider(pattern, client));
        }
        slf
    }

    /// 设置执行环境，供 `bash`/`read`/`write`/`edit` 等需要文件系统或
    /// shell 能力的工具使用。传入 `create_os_env(working_dir)` 创建的 env。
    ///
    /// 未调用时使用 `UnsupportedEnv`——上述工具会返回错误。
    #[pyo3(text_signature = "($self, env)")]
    fn env<'a>(mut slf: PyRefMut<'a, Self>, env: &Bound<'_, PyEnvWrapper>) -> PyRefMut<'a, Self> {
        slf.env = Some(env.borrow().env.clone());
        slf
    }

    /// 设置知识访问上下文（KnowledgeSource/Memory/Recall 工具的 run 授权）。
    ///
    /// 知识工具是 fail-closed：每个 run 必须携带 `KnowledgeAccessContext` 扩展，否则
    /// `knowledge_search` 等返回 "unauthorized"。Senza 默认注入单用户可信上下文
    /// （scope="senza", principal="python-sdk"），与本 SDK 的 `AllowAllAuthorizer`
    /// 姿态一致。调用本方法可覆盖为真实身份/租户（例如多租户场景按应用提供 principal）。
    ///
    /// Args:
    ///     scope: knowledge scope 命名空间（默认 "senza"）。
    ///     principal: 主体标识（默认 "python-sdk"）。
    ///     kind: 主体类型，如 "user"/"service"（默认 "user"）。
    #[pyo3(
        signature = (scope="senza", principal="python-sdk", kind="user"),
        text_signature = "($self, scope='senza', principal='python-sdk', kind='user')"
    )]
    fn knowledge_access<'a>(
        mut slf: PyRefMut<'a, Self>,
        scope: &str,
        principal: &str,
        kind: &str,
    ) -> PyRefMut<'a, Self> {
        slf.knowledge_access = Some(KnowledgeAccessContext::new(
            KnowledgeScope::new(scope),
            PrincipalRef::new(principal, kind),
        ));
        slf
    }

    /// 设置 thinking level（构建时）。
    ///
    /// 接受: "off", "minimal", "low", "medium", "high", "xhigh", 或 "budget:<tokens>"。
    fn thinking_level<'a>(
        mut slf: PyRefMut<'a, Self>,
        level: &str,
    ) -> PyResult<PyRefMut<'a, Self>> {
        if let Some(b) = slf.builder.take() {
            let tl = parse_thinking_level(level)?;
            slf.builder = Some(b.thinking_level(tl));
        }
        Ok(slf)
    }

    /// Enable or disable auto-compaction (enabled by default).
    fn auto_compact<'a>(mut slf: PyRefMut<'a, Self>, enabled: bool) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            slf.builder = Some(b.auto_compact(enabled));
        }
        slf
    }

    /// Set the token budget reserved for system prompt + new response during compaction.
    fn compaction_reserve_tokens<'a>(
        mut slf: PyRefMut<'a, Self>,
        tokens: Option<u32>,
    ) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            slf.builder = Some(b.compaction_reserve_tokens(tokens));
        }
        slf
    }

    /// Set how many recent tokens to keep unsummarized during compaction.
    fn compaction_keep_recent_tokens<'a>(
        mut slf: PyRefMut<'a, Self>,
        tokens: Option<u32>,
    ) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            slf.builder = Some(b.compaction_keep_recent_tokens(tokens));
        }
        slf
    }

    /// 注册一个 `ShouldStopHook`（无需包装在 Plugin 中）。
    ///
    /// 多次调用累积多个 hook——`CompositeShouldStopHook` 为全执行语义，
    /// 注册顺序不影响正确性（每个 hook 都会运行）。
    #[pyo3(text_signature = "($self, hook)")]
    fn should_stop_hook<'a>(
        mut slf: PyRefMut<'a, Self>,
        hook: &Bound<'_, PyHookWrapper>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        if let Some(b) = slf.builder.take() {
            let h = hook.borrow().as_should_stop_hook()?;
            slf.builder = Some(b.should_stop_hook(h));
        }
        Ok(slf)
    }

    /// 注册一个 `AfterTurnHook`（无需包装在 Plugin 中）。
    ///
    /// 多次调用累积多个 hook——`CompositeAfterTurnHook` 为全执行语义，
    /// 按注册顺序依次执行每个 hook（顺序保证）。
    #[pyo3(text_signature = "($self, hook)")]
    fn after_turn_hook<'a>(
        mut slf: PyRefMut<'a, Self>,
        hook: &Bound<'_, PyHookWrapper>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        if let Some(b) = slf.builder.take() {
            let h = hook.borrow().as_after_turn_hook()?;
            slf.builder = Some(b.after_turn_hook(h));
        }
        Ok(slf)
    }

    /// 注册一个 `ProviderErrorHook`（无需包装在 Plugin 中）。
    ///
    /// provider 非瞬态错误（重试耗尽后仍失败）上抛前调用；hook 返回
    /// `"retry"` 则同轮重试，返回 `"surface"` / `None` 则原样上抛。
    /// 多次调用累积多个 hook——按注册顺序执行，首个 retry 生效。
    #[pyo3(text_signature = "($self, hook)")]
    fn provider_error_hook<'a>(
        mut slf: PyRefMut<'a, Self>,
        hook: &Bound<'_, PyHookWrapper>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        if let Some(b) = slf.builder.take() {
            let h = hook.borrow().as_provider_error_hook()?;
            slf.builder = Some(b.provider_error_hook(h));
        }
        Ok(slf)
    }

    /// 设置 response format，用于要求模型输出结构化 JSON。
    ///
    /// 传入 `create_json_object_format()` 或 `create_json_schema_format(...)` 创建的 format。
    /// 传 `None` 重置为默认值（不强制格式）。
    #[pyo3(text_signature = "($self, fmt)")]
    fn response_format<'a>(
        mut slf: PyRefMut<'a, Self>,
        fmt: Option<&Bound<'_, PyResponseFormat>>,
    ) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            let fmt = fmt.map(|f| f.borrow().fmt.clone());
            slf.builder = Some(b.response_format(fmt));
        }
        slf
    }

    /// 直接设置 hook 集合。push 语义：hooks 追加到 builder 现有 hooks。
    ///
    /// 列表中每个 `Hook` 按其 kind 分发到对应的 hook 向量。多次调用可
    /// 组合来自不同来源的 hooks。
    #[pyo3(text_signature = "($self, hooks_list)")]
    fn hooks<'a>(
        mut slf: PyRefMut<'a, Self>,
        hooks_list: Vec<Bound<'_, PyHookWrapper>>,
    ) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            let mut harness_hooks = llm_harness_agent::HarnessHooks::none();
            for h in &hooks_list {
                h.borrow().push_into(&mut harness_hooks);
            }
            slf.builder = Some(b.hooks(harness_hooks));
        }
        slf
    }

    /// 设置 transient provider 错误的重试配置。
    #[pyo3(text_signature = "($self, max_retries, base_delay_ms)")]
    fn retry<'a>(
        mut slf: PyRefMut<'a, Self>,
        max_retries: u32,
        base_delay_ms: u64,
    ) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            slf.builder = Some(b.retry(Some(RetryConfig::new(max_retries, base_delay_ms))));
        }
        slf
    }

    /// 设置模型元数据（context_window, max_tokens）。
    #[pyo3(text_signature = "($self, context_window, max_tokens)")]
    fn model_info<'a>(
        mut slf: PyRefMut<'a, Self>,
        context_window: u32,
        max_tokens: u32,
    ) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            slf.builder = Some(b.model_info(Some(ModelInfo {
                context_window,
                max_tokens,
            })));
        }
        slf
    }

    /// 设置 final-answer 分类模式。
    ///
    /// 接受: `"heuristic"`（默认，非工具终止消息视为最终答案）或
    /// `"tool"`（要求模型调用 `final_answer` 工具）。
    #[pyo3(text_signature = "($self, mode)")]
    fn final_answer_mode<'a>(
        mut slf: PyRefMut<'a, Self>,
        mode: &str,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let m = match mode {
            "heuristic" => FinalAnswerMode::Heuristic,
            "tool" => FinalAnswerMode::required_tool(),
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "final_answer_mode must be 'heuristic' or 'tool', got '{other}'"
                )));
            }
        };
        if let Some(b) = slf.builder.take() {
            slf.builder = Some(b.final_answer_mode(m));
        }
        Ok(slf)
    }

    /// Register a `FinalAnswerValidator` (without wrapping in a Plugin).
    ///
    /// Multiple calls accumulate validators. A rejected candidate is never
    /// committed; the loop retries (letting the model generate a new answer).
    #[pyo3(text_signature = "($self, validator)")]
    fn final_answer_validator<'a>(
        mut slf: PyRefMut<'a, Self>,
        validator: &Bound<'_, PyHookWrapper>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        if let Some(b) = slf.builder.take() {
            let mut harness_hooks = llm_harness_agent::HarnessHooks::none();
            validator.borrow().push_into(&mut harness_hooks);
            slf.builder = Some(b.hooks(harness_hooks));
        }
        Ok(slf)
    }

    /// 设置 LLM 请求的 stream options。
    #[pyo3(text_signature = "($self, timeout_ms=None, max_retries=None)")]
    #[pyo3(signature = (timeout_ms=None, max_retries=None))]
    fn stream_options<'a>(
        mut slf: PyRefMut<'a, Self>,
        timeout_ms: Option<u64>,
        max_retries: Option<u32>,
    ) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            slf.builder = Some(b.stream_options(Some(StreamOptions {
                timeout_ms,
                max_retries,
                ..Default::default()
            })));
        }
        slf
    }

    /// 设置 steer/follow-up 队列容量。`None` 重置为默认值（32）。
    #[pyo3(text_signature = "($self, capacity=None)")]
    #[pyo3(signature = (capacity=None))]
    fn queue_capacity<'a>(
        mut slf: PyRefMut<'a, Self>,
        capacity: Option<usize>,
    ) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            slf.builder = Some(b.queue_capacity(capacity));
        }
        slf
    }

    /// 禁用 `SkillReadTool` 的自动注册。
    ///
    /// 默认情况下，当 skills 存在时 `build()` 会自动注册 `SkillReadTool`。
    /// 调用此方法可选择退出。
    #[pyo3(text_signature = "($self)")]
    fn disable_skill_read_tool<'a>(mut slf: PyRefMut<'a, Self>) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            slf.builder = Some(b.disable_skill_read_tool());
        }
        slf
    }

    /// 追加单个 skill。
    ///
    /// skill 须由 `load_skills()` 创建。多次调用累积多个 skill。
    #[pyo3(text_signature = "($self, skill)")]
    fn skill<'a>(mut slf: PyRefMut<'a, Self>, skill: &Bound<'_, PySkill>) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            let plugin = SingleSkillPlugin {
                skill: skill.borrow().skill.clone(),
            };
            slf.builder = Some(b.install(&plugin));
        }
        slf
    }

    /// 追加多个 skill。
    ///
    /// `skills` 须由 `load_skills()` 创建。多次调用累积。
    #[pyo3(text_signature = "($self, skills)")]
    fn skills<'a>(
        mut slf: PyRefMut<'a, Self>,
        skills: Vec<Bound<'_, PySkill>>,
    ) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            let collected: Vec<Skill> = skills.iter().map(|s| s.borrow().skill.clone()).collect();
            let plugin = MultiSkillPlugin { skills: collected };
            slf.builder = Some(b.install(&plugin));
        }
        slf
    }
    /// 配置独立的 compaction 模型。
    ///
    /// 设置后，compaction 摘要使用独立 provider/model，
    /// 而非主对话 client。`context_window` 和 `max_tokens`
    /// 应反映 compaction 模型的真实参数。
    #[pyo3(text_signature = "($self, model, context_window, max_tokens)")]
    fn compaction_model<'a>(
        mut slf: PyRefMut<'a, Self>,
        model: &str,
        context_window: u32,
        max_tokens: u32,
    ) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            slf.builder = Some(b.compaction_model(
                model,
                ModelInfo {
                    context_window,
                    max_tokens,
                },
            ));
        }
        slf
    }

    /// 设置自定义 compaction prompt 模板。
    ///
    /// 两个参数都提供时构造 `CompactionPromptSpec`；`user_template`
    /// 必须包含 `{conversation}` 占位符，支持 `{previous_summary}`、
    /// `{file_operations}`、`{query}`。传 `None` 清除（使用默认）。
    #[pyo3(text_signature = "($self, system_prompt=None, user_template=None)")]
    #[pyo3(signature = (system_prompt=None, user_template=None))]
    fn compaction_prompt<'a>(
        mut slf: PyRefMut<'a, Self>,
        system_prompt: Option<&str>,
        user_template: Option<&str>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        if let Some(b) = slf.builder.take() {
            let spec = match (system_prompt, user_template) {
                (None, None) => None,
                (Some(sp), Some(ut)) => Some(
                    CompactionPromptSpec::new(sp, ut)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                _ => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "compaction_prompt: provide both system_prompt and user_template, or None for both",
                    ));
                }
            };
            slf.builder = Some(b.compaction_prompt(spec));
        }
        Ok(slf)
    }

    /// 设置 compaction 查询意图，用于上下文感知的摘要。
    #[pyo3(signature = (query=None))]
    #[pyo3(text_signature = "($self, query=None)")]
    fn compaction_query<'a>(
        mut slf: PyRefMut<'a, Self>,
        query: Option<String>,
    ) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            slf.builder = Some(b.compaction_query(query));
        }
        slf
    }

    /// 设置 pricing provider，用于成本计算。
    ///
    /// 设置后 builder 自动注入 `CostAccumulatorHook`，
    /// `harness.usage()["total_cost"]` 才有 USD 值。
    #[pyo3(text_signature = "($self, provider)")]
    fn pricing<'a>(
        mut slf: PyRefMut<'a, Self>,
        provider: &Bound<'_, PyPricingProvider>,
    ) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            let p = provider.borrow().provider.clone();
            slf.builder = Some(b.pricing(p));
        }
        slf
    }

    /// Attach a caller-owned UsageLedger for shared cost accounting across harnesses.
    ///
    /// The ledger is cloned into the builder; the Python-side object remains
    /// usable (snapshot reflects state shared with the built harness).
    #[pyo3(text_signature = "($self, ledger)")]
    fn usage_ledger<'a>(
        mut slf: PyRefMut<'a, Self>,
        ledger: &Bound<'_, PyUsageLedger>,
    ) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            slf.builder = Some(b.usage_ledger(ledger.borrow().ledger.clone()));
        }
        slf
    }
    /// 配置预算上限和可选的超限 hook。
    ///
    /// - `limit` — 预算上限（USD）。
    /// - `exceeded_hook=None` → surveillance 模式：只统计成本，不停。
    /// - `exceeded_hook=Some(h)` → 超限时由 `h` 决定继续/停止。
    #[pyo3(text_signature = "($self, limit, exceeded_hook=None)")]
    #[pyo3(signature = (limit, exceeded_hook=None))]
    fn budget<'a>(
        mut slf: PyRefMut<'a, Self>,
        limit: f64,
        exceeded_hook: Option<&Bound<'_, PyBudgetExceededHook>>,
    ) -> PyRefMut<'a, Self> {
        if let Some(b) = slf.builder.take() {
            let ledger = llm_harness_agent::UsageLedger::default();
            let cost_state = ledger.shared_state();
            let hook = exceeded_hook.map(|h| h.borrow().hook.clone());
            let adapter = llm_harness_strategy::BudgetControlAdapter::new(cost_state, limit, hook);
            slf.builder = Some(b.usage_ledger(ledger).should_stop_hook(Arc::new(adapter)));
        }
        slf
    }

    /// 添加一个 MCP server。
    ///
    /// 配置在 `build()` 时生效。可多次调用添加多个 server。
    /// ```python
    /// builder = HarnessBuilder("model").mcp_server("fs", McpServerConfig.stdio(...))
    /// ```
    #[pyo3(text_signature = "($self, name, config)")]
    fn mcp_server<'a>(
        mut slf: PyRefMut<'a, Self>,
        name: String,
        config: &PyMcpServerConfig,
    ) -> PyRefMut<'a, Self> {
        slf.mcp_servers.push((name, config.inner.clone()));
        slf
    }

    /// 指定 mcp.json 配置文件路径。
    /// 文件在 `build()` 时异步读取。可多次调用指定多个文件。
    #[pyo3(text_signature = "($self, path)")]
    fn mcp_config_file<'a>(mut slf: PyRefMut<'a, Self>, path: &str) -> PyRefMut<'a, Self> {
        slf.mcp_config_files.push(PathBuf::from(path));
        slf
    }

    /// 传入已创建的 McpManager（高级 API）。
    /// 用外部 manager 手动管理 MCP server 生命周期。
    #[pyo3(text_signature = "($self, manager)")]
    fn with_mcp_manager<'a>(
        mut slf: PyRefMut<'a, Self>,
        manager: &PyMcpManager,
    ) -> PyRefMut<'a, Self> {
        slf.mcp_manager = Some(manager.inner.clone());
        slf
    }

    /// Enable sub-agent spawn infrastructure.
    ///
    /// Wires `MessageBus`, `HarnessSubAgentSpawner`, `SpawnPlugin`,
    /// and the five main-agent spawn tools (`spawn_agent`,
    /// `message_subagent`, `await_subagent_reply`, `query_subagent`,
    /// `abort_subagent`) into the harness at build time.
    ///
    /// Args:
    ///     model: Default model name for sub-agents.
    ///     provider: LLM provider for sub-agents (same as main agent's provider).
    ///     session_dir: Directory for sub-agent session JSONL files.
    #[pyo3(signature = (model, provider, session_dir, max_concurrent=None))]
    fn enable_spawn<'a>(
        mut slf: PyRefMut<'a, Self>,
        model: &str,
        provider: &Bound<'_, PyProvider>,
        session_dir: &str,
        max_concurrent: Option<usize>,
    ) -> PyRefMut<'a, Self> {
        slf.spawn_config = Some(SpawnConfig {
            model: model.to_string(),
            client: provider.borrow().client.clone(),
            session_dir: PathBuf::from(session_dir),
            max_concurrent,
        });
        slf
    }

    /// 返回 builder 状态摘要。
    fn __repr__(&self) -> String {
        match &self.builder {
            Some(_) => {
                let mcp_flag = if !self.mcp_servers.is_empty()
                    || !self.mcp_config_files.is_empty()
                    || self.mcp_manager.is_some()
                {
                    ", mcp"
                } else {
                    ""
                };
                format!("HarnessBuilder(pending{mcp_flag})")
            }
            None => "HarnessBuilder(consumed)".to_string(),
        }
    }

    /// Set a session repo for persistent (JSONL-backed) sessions.
    ///
    /// If `session_id` is given, opens an existing session; otherwise creates
    /// a new one. When set, `build()` uses `build_with_session()` instead of
    /// the default in-memory session.
    #[pyo3(text_signature = "($self, repo, session_id=None)")]
    #[pyo3(signature = (repo, session_id=None))]
    fn session_repo<'a>(
        mut slf: PyRefMut<'a, Self>,
        repo: &Bound<'_, crate::knowledge::pysessionrecall::PySessionRepo>,
        session_id: Option<String>,
    ) -> PyRefMut<'a, Self> {
        slf.session_repo = Some(repo.borrow().repo.clone());
        slf.session_id = session_id;
        slf
    }

    /// 构建 harness 并返回 `AgentHarness`。
    ///
    /// 执行环境为 `.env()` 设置的 env；未设置时使用 `UnsupportedEnv`
    /// （无文件系统 / shell 能力）。释放 GIL 后用全局 tokio runtime
    /// 执行 async build。若未注册任何 provider，返回 `RuntimeError`
    /// （`HarnessBuildError::NoProvider`）。
    fn build(&mut self, py: Python<'_>) -> PyResult<Py<PyAgentHarness>> {
        let builder = self.builder.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("build() already consumed this builder")
        })?;

        let env: Arc<dyn ExecutionEnv> = self
            .env
            .take()
            .unwrap_or_else(|| Arc::new(UnsupportedEnv::new()));
        let rt = runtime(py);

        let has_mcp = !self.mcp_servers.is_empty()
            || !self.mcp_config_files.is_empty()
            || self.mcp_manager.is_some();

        let spawn_config = self.spawn_config.take();
        let session_repo = self.session_repo.take();
        let session_id = self.session_id.take();
        let knowledge_access = self.knowledge_access.take();

        // If a session repo is set, load/create a session and use
        // build_with_session() instead of the default in-memory build.
        if let Some(repo) = session_repo {
            let storage = if let Some(id) = session_id {
                crate::shared::pyerror::detach_catch_panic_result(py, move || {
                    rt.block_on(async move { repo.open(&id).await })
                })?
            } else {
                crate::shared::pyerror::detach_catch_panic_result(py, move || {
                    rt.block_on(async move {
                        repo.create(llm_harness_agent::session::CreateSessionOptions::default())
                            .await
                    })
                })?
            };
            let session = llm_harness_agent::Session::new(storage);

            let (builder, spawn_wiring) = match spawn_config {
                Some(cfg) => wire_spawn(builder, cfg),
                None => (builder, None),
            };

            let harness = crate::shared::pyerror::detach_catch_panic_result(py, move || {
                builder.build_with_session(env, session)
            })?;
            let harness = Arc::new(harness);

            if let Some(wiring) = spawn_wiring {
                wiring.post_build(&harness);
            }

            return Py::new(
                py,
                match knowledge_access.clone() {
                    Some(acc) => PyAgentHarness::new_base_with_access(harness, acc),
                    None => PyAgentHarness::new_base(harness),
                },
            );
        }

        if has_mcp {
            // 提升为 McpHarnessBuilder 并追加 MCP 配置。
            let mut mcp_builder = builder.with_mcp();
            for (name, config) in std::mem::take(&mut self.mcp_servers) {
                mcp_builder = mcp_builder.mcp_server(name, config);
            }
            for path in std::mem::take(&mut self.mcp_config_files) {
                mcp_builder = mcp_builder.mcp_config_file(path);
            }
            if let Some(manager) = self.mcp_manager.take() {
                mcp_builder = mcp_builder.with_mcp_manager(manager);
            }

            let mcp_harness = crate::shared::pyerror::detach_catch_panic_result(py, move || {
                rt.block_on(async move { mcp_builder.build(env).await })
            })?;
            Py::new(
                py,
                match knowledge_access.clone() {
                    Some(acc) => PyAgentHarness::new_mcp_with_access(Arc::new(mcp_harness), acc),
                    None => PyAgentHarness::new_mcp(Arc::new(mcp_harness)),
                },
            )
        } else {
            // If spawn is enabled, wire spawn infrastructure into the builder
            // before build, and set post-build hooks after.
            let (builder, spawn_wiring) = match spawn_config {
                Some(cfg) => wire_spawn(builder, cfg),
                None => (builder, None),
            };

            let harness = crate::shared::pyerror::detach_catch_panic_result(py, move || {
                rt.block_on(async move { builder.build(env).await })
            })?;
            let harness = Arc::new(harness);

            if let Some(wiring) = spawn_wiring {
                wiring.post_build(&harness);
            }

            Py::new(
                py,
                match knowledge_access.clone() {
                    Some(acc) => PyAgentHarness::new_base_with_access(harness, acc),
                    None => PyAgentHarness::new_base(harness),
                },
            )
        }
    }
}

// ── pub(crate) helpers（非 #[pymethods]：返回 Rust 类型） ─────────────────────

impl PyHarnessBuilder {
    /// 取出内部 `HarnessBuilder`（供 `with_step_builder` 适配器使用）。
    pub(crate) fn take_builder(&mut self) -> Option<HarnessBuilder> {
        self.builder.take()
    }

    /// 从已有 `HarnessBuilder` 构造包装（供 `with_step_builder` 适配器使用）。
    pub(crate) fn from_builder(b: HarnessBuilder) -> Self {
        Self {
            builder: Some(b),
            env: None,
            mcp_servers: Vec::new(),
            mcp_config_files: Vec::new(),
            mcp_manager: None,
            spawn_config: None,
            session_repo: None,
            session_id: None,
            knowledge_access: None,
        }
    }
}

// ── Skill plugin helpers ────────────────────────────────────────────────────

/// 单 skill 插件——通过 `Plugin::register_skills` 注入一个 skill。
struct SingleSkillPlugin {
    skill: Skill,
}

impl Plugin for SingleSkillPlugin {
    fn name(&self) -> &str {
        "senza-single-skill"
    }

    fn register_skills(&self, skills: &mut Vec<Skill>) {
        skills.push(self.skill.clone());
    }
}

/// 多 skill 插件——通过 `Plugin::register_skills` 注入一组 skill。
struct MultiSkillPlugin {
    skills: Vec<Skill>,
}

impl Plugin for MultiSkillPlugin {
    fn name(&self) -> &str {
        "senza-multi-skill"
    }

    fn register_skills(&self, skills: &mut Vec<Skill>) {
        skills.extend(self.skills.iter().cloned());
    }
}

// ── UsageLedger ─────────────────────────────────────────────────────────────

/// Caller-owned usage accounting state, shareable across multiple harnesses.
///
/// Wraps `llm_harness_agent::UsageLedger`, which holds an
/// `Arc<Mutex<CostAggregate>>`. Because the inner state is `Arc`-shared,
/// cloning the ledger (as `usage_ledger()` does) shares the same accumulator —
/// cost recorded by any harness is visible via `snapshot()` on this object.
#[pyclass(name = "UsageLedger", skip_from_py_object)]
#[derive(Clone)]
pub struct PyUsageLedger {
    pub(crate) ledger: llm_harness_agent::UsageLedger,
}

#[pymethods]
impl PyUsageLedger {
    #[new]
    fn new() -> Self {
        Self {
            ledger: llm_harness_agent::UsageLedger::default(),
        }
    }

    /// Return the current completed-call accounting snapshot as a dict.
    #[pyo3(text_signature = "($self)")]
    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cost = self.ledger.snapshot();
        crate::runtime::pyworkflow::cost_aggregate_to_dict(py, &cost)
    }

    fn __repr__(&self) -> &'static str {
        "UsageLedger"
    }
}

impl Default for PyUsageLedger {
    fn default() -> Self {
        Self::new()
    }
}
