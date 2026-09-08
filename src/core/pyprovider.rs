//! LLM provider 工厂：从 Python 侧创建内置 provider 实例。
//!
//! v1 只支持内置 provider（OpenAI/Anthropic），不暴露 `LlmClient` trait。
//! Python 侧通过 `create_openai_provider()` / `create_anthropic_provider()`
//! 创建不透明的 `PyProvider`，传给 `HarnessBuilder.provider(pattern, provider)`。

use std::sync::Arc;

use llm_harness_loop::{AnthropicProvider, LlmClient, OpenAIProvider, ThinkingScheme};
use pyo3::prelude::*;

/// 不透明的 LLM provider 包装。持有 `Arc<dyn LlmClient>`。
#[pyclass(name = "Provider")]
pub struct PyProvider {
    pub(crate) client: Arc<dyn LlmClient>,
}

/// 创建 OpenAI 兼容 provider。
///
/// `base_url` 为空时使用默认 `https://api.openai.com`。
/// 当 `chat_path` 未指定时，自动去除 `base_url` 尾部的 `/v1` 以避免双重路径。
/// `chat_path` 指定 chat completions 路径（默认 `/v1/chat/completions`）。
/// `thinking_scheme` 指定 thinking 翻译策略：`"none"` / `"reasoning_effort"` / `"thinking_toggle"`。
/// `parse_reasoning_content` 解析 DeepSeek 风格 reasoning_content；
/// `tolerant_keepalive` 容忍流中 keepalive 消息（DeepSeek 兼容）。
#[pyfunction]
#[pyo3(signature = (api_key, base_url=None, chat_path=None, thinking_scheme=None, parse_reasoning_content=true, tolerant_keepalive=true, documents=false, documents_inline=false))]
#[allow(clippy::too_many_arguments)]
pub fn create_openai_provider(
    py: Python<'_>,
    api_key: &str,
    base_url: Option<&str>,
    chat_path: Option<&str>,
    thinking_scheme: Option<&str>,
    parse_reasoning_content: bool,
    tolerant_keepalive: bool,
    documents: bool,
    documents_inline: bool,
) -> PyResult<Py<PyProvider>> {
    let mut builder = OpenAIProvider::builder(api_key)
        .parse_reasoning_content(parse_reasoning_content)
        .tolerant_keepalive(tolerant_keepalive)
        .documents(documents || documents_inline);
    if documents_inline {
        builder = builder.documents_inline(true);
    }
    if let Some(url) = base_url
        && !url.is_empty()
    {
        // When chat_path is not customised, auto-strip a trailing /v1
        // from base_url to prevent the double-/v1 404 that affects
        // most OpenAI-compatible deployments (e.g. vLLM, DeepSeek).
        let url = if chat_path.is_none() {
            let trimmed = url.trim_end_matches('/');
            trimmed.strip_suffix("/v1").unwrap_or(trimmed)
        } else {
            url
        };
        builder = builder.base_url(url);
    }
    if let Some(path) = chat_path
        && !path.is_empty()
    {
        builder = builder.chat_path(path);
    }
    if let Some(scheme) = thinking_scheme
        && !scheme.is_empty()
    {
        let scheme = match scheme.to_ascii_lowercase().as_str() {
            "none" => ThinkingScheme::None,
            "reasoning_effort" => ThinkingScheme::ReasoningEffort,
            "thinking_toggle" => ThinkingScheme::ThinkingToggle,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown thinking_scheme '{other}': expected 'none', 'reasoning_effort', or 'thinking_toggle'"
                )));
            }
        };
        builder = builder.thinking_scheme(scheme);
    }
    let client: Arc<dyn LlmClient> = Arc::new(builder.build());
    Py::new(py, PyProvider { client })
}

/// 创建 Anthropic provider。
///
/// `base_url` 为空时使用默认 `https://api.anthropic.com`。
/// `messages_path` 指定 messages API 路径（默认 `/v1/messages`），
/// 用于 Anthropic 兼容代理（Azure、AWS Bedrock、自建网关等）。
#[pyfunction]
#[pyo3(signature = (api_key, base_url=None, messages_path=None))]
pub fn create_anthropic_provider(
    py: Python<'_>,
    api_key: &str,
    base_url: Option<&str>,
    messages_path: Option<&str>,
) -> PyResult<Py<PyProvider>> {
    let mut builder = AnthropicProvider::builder(api_key);
    if let Some(url) = base_url
        && !url.is_empty()
    {
        builder = builder.base_url(url);
    }
    if let Some(path) = messages_path
        && !path.is_empty()
    {
        builder = builder.messages_path(path);
    }
    let client: Arc<dyn LlmClient> = Arc::new(builder.build());
    Py::new(py, PyProvider { client })
}
