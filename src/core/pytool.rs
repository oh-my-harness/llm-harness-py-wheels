//! Python callable 包装为 `Tool` trait 实现。
//!
//! 验证风险点：`Py<PyAny>` 持有 Python callable、`spawn_blocking` +
//! `Python::attach` + `call1` 调用 Python 函数、`ToolResult` 从 Python
//! dict 解析的完整路径。

use std::sync::Arc;

use futures::future::BoxFuture;
use llm_harness_types::{
    DataBlock, Tool, ToolContext, ToolExecutionMode, ToolFailure, ToolProgress, ToolResult,
};
#[cfg(feature = "test-utils")]
use llm_harness_types::{RunContext, RunRequest};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::shared::value_conv::{pyobject_to_value, value_to_pyobject};
use llm_harness_types::{BinarySource, ContentBlock, ImageSource};

/// Python callable 包装为 `Tool` trait。
pub struct PyTool {
    name: String,
    description: String,
    schema: Value,
    callback: Arc<Py<PyAny>>,
    is_async: bool,
    report_duration: bool,
}

impl PyTool {
    pub fn new(
        name: String,
        description: String,
        schema: Value,
        callback: Py<PyAny>,
        report_duration: bool,
    ) -> Self {
        let is_async = Python::attach(|py| {
            let inspect = pyo3::types::PyModule::import(py, "inspect")?;
            let is_coro: bool = inspect
                .call_method1("iscoroutinefunction", (callback.bind(py),))?
                .extract()?;
            Ok::<_, PyErr>(is_coro)
        })
        .unwrap_or_else(|e| {
            tracing::debug!("failed to detect async callback, assuming sync: {e}");
            false
        });
        Self {
            name,
            description,
            schema,
            callback: Arc::new(callback),
            is_async,
            report_duration,
        }
    }
}

impl Tool for PyTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn report_duration(&self) -> bool {
        self.report_duration
    }

    fn execute<'a>(
        &'a self,
        args: Value,
        ctx: &'a ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult, ToolFailure>> {
        let callback = Arc::clone(&self.callback);
        let is_async = self.is_async;
        let abort = ctx.abort.clone();
        let update_tx = ctx.update_tx.clone();
        Box::pin(async move {
            let result = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let cb = callback.bind(py);
                    let py_args = value_to_pyobject(py, &args)?;
                    let py_ctx = PyToolContext::new(abort.clone(), update_tx.clone());

                    if is_async {
                        // async: schedule the coroutine on the user's main
                        // event loop when possible (issue #13), falling
                        // back to asyncio.run().
                        let coro = cb.call1((py_args, py_ctx))?;
                        let raw = crate::core::pyloop::run_coro(py, &coro)?;
                        parse_tool_result(&raw)
                    } else {
                        // sync: 直接调用
                        let raw = cb.call1((py_args, py_ctx))?;
                        parse_tool_result(&raw)
                    }
                })
            })
            .await
            .map_err(|e| ToolFailure::new("execution_error", format!("callback join failed: {e}")))?
            .map_err(|e: PyErr| ToolFailure::new("execution_error", e.to_string()))?;
            Ok(result)
        })
    }

    fn parameters_schema(&self) -> &Value {
        &self.schema
    }

    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Parallel
    }
}

/// 用户附件：包装 `ContentBlock::Image` 或 `ContentBlock::Document`。
///
/// 由 Python 侧构造函数（`image_url` / `image_base64` / `document_url` /
/// `document_file`）创建，对用户不透明。`prompt(attachments=[...])` 与
/// 工具回调返回值中消费。
#[pyclass(name = "Attachment")]
pub struct PyAttachment {
    block: ContentBlock,
}

impl PyAttachment {
    pub(crate) fn block(&self) -> &ContentBlock {
        &self.block
    }
}

/// 从 Python 构造函数参数构造 `ContentBlock`（纯转换，可单测）。
pub(crate) fn attachment_to_content_block(
    kind: &str,
    data: &str,
    name: Option<String>,
    media_type: Option<String>,
) -> Result<ContentBlock, String> {
    match kind {
        "image_url" => Ok(ContentBlock::Image {
            source: ImageSource::Url {
                url: data.to_string(),
            },
        }),
        "image_base64" => Ok(ContentBlock::Image {
            source: ImageSource::Base64 {
                media_type: media_type.unwrap_or_else(|| "image/png".to_string()),
                data: data.to_string(),
            },
        }),
        "document_url" => Ok(ContentBlock::Document {
            name,
            media_type: media_type.ok_or("document requires media_type")?,
            data: BinarySource::Url {
                url: data.to_string(),
            },
        }),
        "document_base64" => Ok(ContentBlock::Document {
            name,
            media_type: media_type.ok_or("document requires media_type")?,
            data: BinarySource::Base64 {
                data: data.to_string(),
            },
        }),
        other => Err(format!("unknown attachment kind: {other}")),
    }
}

/// `ContentBlock` → `DataBlock`（工具结果桥接用）。
pub(crate) fn content_block_to_data_block(cb: &ContentBlock) -> Option<DataBlock> {
    match cb {
        ContentBlock::Image { source } => Some(DataBlock::Image {
            source: source.clone(),
        }),
        ContentBlock::Document {
            name,
            media_type,
            data,
        } => Some(DataBlock::Document {
            name: name.clone(),
            media_type: media_type.clone(),
            data: data.clone(),
        }),
        ContentBlock::Text { text } => Some(DataBlock::Text {
            text: text.clone(),
            mime_type: None,
        }),
        _ => None,
    }
}

/// Parse one dict-shaped content block into a `ContentBlock`.
///
/// Accepts the Qevos/legacy shapes so existing tools migrate without change:
/// - `{"type": "text", "text": ...}`
/// - `{"type": "image", "media_type": ..., "data": <base64>}` (Qevos style)
/// - `{"type": "image", "url": ...}` / `{"type": "image_url", "url": ...}`
/// - `{"type": "image_base64", "data": ..., "media_type"?}` — same as the
///   `Attachment` constructor kinds
/// - `{"type": "document", "name": ..., "media_type": ..., "data"/"url": ...}`
pub(crate) fn dict_content_block(
    item_dict: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<Option<ContentBlock>> {
    let block_type: String = item_dict
        .get_item("type")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("content block missing 'type'"))?
        .extract()?;
    match block_type.as_str() {
        "text" => {
            let text: String = item_dict
                .get_item("text")?
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("text content block missing 'text'")
                })?
                .extract()?;
            Ok(Some(ContentBlock::Text { text }))
        }
        "image" => {
            if let Some(url) = item_dict.get_item("url")?.filter(|v| !v.is_none()) {
                let url: String = url.extract()?;
                return Ok(Some(ContentBlock::Image {
                    source: ImageSource::Url { url },
                }));
            }
            let data: String = item_dict
                .get_item("data")?
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "image content block requires 'data' (base64) or 'url'",
                    )
                })?
                .extract()?;
            let media_type: String = item_dict
                .get_item("media_type")?
                .and_then(|v| v.extract().ok())
                .unwrap_or_else(|| "image/png".to_string());
            Ok(Some(ContentBlock::Image {
                source: ImageSource::Base64 { media_type, data },
            }))
        }
        // Constructor-style kinds: delegate to the shared converter so the
        // accepted shapes stay identical to `image_url()` / `document_*()`.
        "image_url" | "image_base64" | "document_url" | "document_base64" => {
            let data: String = item_dict
                .get_item("data")?
                .or_else(|| item_dict.get_item("url").ok().flatten())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "attachment content block missing 'data'/'url'",
                    )
                })?
                .extract()?;
            let name: Option<String> = item_dict
                .get_item("name")?
                .filter(|v| !v.is_none())
                .and_then(|v| v.extract().ok());
            let media_type: Option<String> = item_dict
                .get_item("media_type")?
                .filter(|v| !v.is_none())
                .and_then(|v| v.extract().ok());
            attachment_to_content_block(&block_type, &data, name, media_type)
                .map(Some)
                .map_err(pyo3::exceptions::PyValueError::new_err)
        }
        _other => Ok(None),
    }
}

#[pymethods]
impl PyAttachment {
    #[new]
    #[pyo3(signature = (kind, data, name, media_type))]
    fn new(
        kind: &str,
        data: &str,
        name: Option<String>,
        media_type: Option<String>,
    ) -> PyResult<Self> {
        attachment_to_content_block(kind, data, name, media_type)
            .map(|block| Self { block })
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        match &self.block {
            ContentBlock::Image { source } => match source {
                ImageSource::Url { url } => format!("Attachment(image_url={url})"),
                ImageSource::Base64 { media_type, .. } => {
                    format!("Attachment(image_base64, {media_type})")
                }
            },
            ContentBlock::Document { name, .. } => {
                format!(
                    "Attachment(document, {})",
                    name.as_deref().unwrap_or("<unnamed>")
                )
            }
            _ => "Attachment(?)".to_string(),
        }
    }
}

#[cfg(test)]
mod attachment_tests {
    use super::{attachment_to_content_block, dict_content_block};
    use crate::core::pytool::{ContentBlock, ImageSource};
    use pyo3::Python;
    use pyo3::types::PyDict;

    fn block_from_dict(py: Python<'_>, json: &str) -> Option<ContentBlock> {
        let value = py
            .eval(std::ffi::CString::new(json).unwrap().as_c_str(), None, None)
            .unwrap();
        let dict = value.cast::<PyDict>().unwrap();
        dict_content_block(dict).unwrap()
    }

    #[test]
    fn dict_qevos_image_block() {
        Python::attach(|py| {
            let b = block_from_dict(
                py,
                r#"{"type": "image", "media_type": "image/jpeg", "data": "AAAA"}"#,
            )
            .unwrap();
            match b {
                ContentBlock::Image {
                    source: ImageSource::Base64 { media_type, data },
                } => {
                    assert_eq!(media_type, "image/jpeg");
                    assert_eq!(data, "AAAA");
                }
                _ => panic!("expected base64 image"),
            }
        });
    }

    #[test]
    fn dict_image_url_block() {
        Python::attach(|py| {
            for shape in [
                r#"{"type": "image", "url": "https://x/y.png"}"#,
                r#"{"type": "image_url", "url": "https://x/y.png"}"#,
            ] {
                let b = block_from_dict(py, shape).unwrap();
                assert!(
                    matches!(
                        b,
                        ContentBlock::Image {
                            source: ImageSource::Url { .. }
                        }
                    ),
                    "shape: {shape}"
                );
            }
        });
    }

    #[test]
    fn dict_document_block_delegates() {
        Python::attach(|py| {
            let b = block_from_dict(
                py,
                r#"{"type": "document_base64", "name": "d.pdf", "media_type": "application/pdf", "data": "AAAA"}"#,
            )
            .unwrap();
            assert!(matches!(b, ContentBlock::Document { .. }));
        });
    }

    #[test]
    fn dict_unknown_type_returns_none() {
        Python::attach(|py| {
            let b = block_from_dict(py, r#"{"type": "audio", "data": "x"}"#);
            assert!(b.is_none());
        });
    }

    #[test]
    fn dict_image_missing_data_and_url_errors() {
        Python::attach(|py| {
            let value = py
                .eval(
                    std::ffi::CString::new(r#"{"type": "image"}"#)
                        .unwrap()
                        .as_c_str(),
                    None,
                    None,
                )
                .unwrap();
            let dict = value.cast::<PyDict>().unwrap();
            assert!(dict_content_block(dict).is_err());
        });
    }

    #[test]
    fn image_url_block() {
        let b = attachment_to_content_block("image_url", "https://x/y.png", None, None).unwrap();
        assert!(matches!(b, ContentBlock::Image { .. }));
    }

    #[test]
    fn image_base64_defaults_png() {
        let b = attachment_to_content_block("image_base64", "AAAA", None, None).unwrap();
        match b {
            ContentBlock::Image {
                source: ImageSource::Base64 { media_type, .. },
            } => assert_eq!(media_type, "image/png"),
            _ => panic!("expected base64 image"),
        }
    }

    #[test]
    fn document_requires_media_type() {
        assert!(
            attachment_to_content_block("document_url", "https://x/d.pdf", None, None).is_err()
        );
    }

    #[test]
    fn document_base64_ok() {
        let b = attachment_to_content_block(
            "document_base64",
            "AAAA",
            Some("d.pdf".into()),
            Some("application/pdf".into()),
        )
        .unwrap();
        assert!(matches!(b, ContentBlock::Document { .. }));
    }

    #[test]
    fn unknown_kind_errors() {
        assert!(attachment_to_content_block("audio", "x", None, None).is_err());
    }
}

/// 解析 Python 返回值为 `ToolResult`。
///
/// 期望 dict 形如：
/// ```python
/// {"content": [{"type": "text", "text": "..."}], "details": ..., "terminate": False}
/// ```
///
/// 另接受 `Attachment`（裸值 / list 元素 / content 列表元素）。
fn parse_tool_result(obj: &Bound<'_, PyAny>) -> PyResult<ToolResult> {
    // Bare Attachment return → single image/document block.
    if let Ok(att) = obj.extract::<Bound<'_, PyAttachment>>() {
        return match content_block_to_data_block(att.borrow().block()) {
            Some(block) => Ok(ToolResult::full(vec![block], Value::Null, false)),
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "attachment is not image/document",
            )),
        };
    }
    // Accept plain string as shorthand for {"content": [{"type": "text", "text": <str>}]}
    if let Ok(s) = obj.extract::<String>() {
        return Ok(ToolResult::full(
            vec![DataBlock::text(s)],
            Value::Null,
            false,
        ));
    }
    // Sequence return (list or tuple): elements may be Attachment or str (mixed).
    let as_list: Option<Bound<'_, PyList>> = match obj.cast::<PyList>() {
        Ok(l) => Some(l.clone()),
        Err(_) => obj
            .cast::<pyo3::types::PyTuple>()
            .ok()
            .map(|t| PyList::new(obj.py(), t.iter()).expect("tuple to list conversion")),
    };
    if let Some(list) = as_list {
        let mut blocks = Vec::with_capacity(list.len());
        for item in list {
            if let Ok(att) = item.extract::<Bound<'_, PyAttachment>>() {
                match content_block_to_data_block(att.borrow().block()) {
                    Some(block) => blocks.push(block),
                    None => {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "attachment is not image/document",
                        ));
                    }
                }
            } else if let Ok(s) = item.extract::<String>() {
                blocks.push(DataBlock::text(s));
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "list return items must be Attachment or str",
                ));
            }
        }
        return Ok(ToolResult::full(blocks, Value::Null, false));
    }
    let dict = obj.cast::<PyDict>()?;

    // If dict has no "content" key, treat the entire dict as a text content
    // block (JSON-serialized). Extract terminate if present.
    if dict.get_item("content")?.is_none() {
        let json_val = pyobject_to_value(obj)?;
        let json_str = serde_json::to_string(&json_val)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let terminate = dict
            .get_item("terminate")?
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false);
        return Ok(ToolResult::full(
            vec![DataBlock::text(json_str)],
            Value::Null,
            terminate,
        ));
    }

    // content: 可选，缺省或 None → 空列表
    let content_vec = match dict.get_item("content")? {
        Some(v) if !v.is_none() => {
            let content_list = v.cast::<PyList>()?;
            let mut blocks = Vec::with_capacity(content_list.len());
            for item in content_list {
                // Attachment items are allowed alongside dicts.
                if let Ok(att) = item.extract::<Bound<'_, PyAttachment>>() {
                    match content_block_to_data_block(att.borrow().block()) {
                        Some(block) => blocks.push(block),
                        None => {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "attachment is not image/document",
                            ));
                        }
                    }
                    continue;
                }
                let item_dict = item.cast::<PyDict>()?;
                match dict_content_block(item_dict)? {
                    Some(cb) => {
                        if let Some(block) = content_block_to_data_block(&cb) {
                            blocks.push(block);
                        } else {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "content block is not a valid image/document/text block",
                            ));
                        }
                    }
                    None => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "unsupported content block type: {} (hint: return a bare Attachment or list of Attachments for images/documents)",
                            item_dict
                                .get_item("type")?
                                .and_then(|v| v.extract::<String>().ok())
                                .unwrap_or_default()
                        )));
                    }
                }
            }
            blocks
        }
        _ => vec![],
    };

    // details: 可选，缺省或 None → Null
    let details = match dict.get_item("details")? {
        Some(v) if !v.is_none() => pyobject_to_value(&v)?,
        _ => Value::Null,
    };

    // terminate: 可选，缺省或 None → false
    let terminate = dict
        .get_item("terminate")?
        .and_then(|v| v.extract::<bool>().ok())
        .unwrap_or(false);

    Ok(ToolResult::full(content_vec, details, terminate))
}

/// Python 侧的 tool context，暴露 `is_cancelled` 和 `send_update`。
#[pyclass(name = "ToolContext")]
pub struct PyToolContext {
    abort: CancellationToken,
    update_tx: tokio::sync::mpsc::Sender<ToolProgress>,
}

impl PyToolContext {
    pub fn new(
        abort: CancellationToken,
        update_tx: tokio::sync::mpsc::Sender<ToolProgress>,
    ) -> Self {
        Self { abort, update_tx }
    }
}

#[pymethods]
impl PyToolContext {
    /// 返回当前是否已收到取消信号。
    fn is_cancelled(&self) -> bool {
        self.abort.is_cancelled()
    }

    /// 推送一个部分结果（Python dict），解析后发送到 update channel。
    fn send_update(&self, result: &Bound<'_, PyAny>) -> PyResult<()> {
        let parsed = parse_tool_result(result)?;
        let progress = ToolProgress {
            content: parsed.model_content,
            details: parsed.details,
        };
        self.update_tx
            .try_send(progress)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

/// 持有 `PyTool` 的不透明 Python 包装，供 Python 侧引用已注册的 tool。
#[pyclass(name = "Tool")]
pub struct PyToolWrapper {
    pub tool: Arc<PyTool>,
}

#[pymethods]
impl PyToolWrapper {
    /// 返回 tool 的名称。
    #[getter]
    fn name(&self) -> &str {
        self.tool.name()
    }

    /// 返回 tool 的描述。
    #[getter]
    fn description(&self) -> &str {
        self.tool.description()
    }

    /// 同步驱动 tool.execute：在独立 tokio runtime 上运行 async future。
    ///
    /// 仅供测试/验证使用；真实场景由 agent loop 调用 `execute`。
    #[cfg(feature = "test-utils")]
    fn drive(&self, args: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = args.py();
        let args_val = crate::shared::value_conv::pyobject_to_value(args)?;
        let tool = self.tool.clone();
        // 在 Python 释放 GIL 后运行 tokio runtime，避免 GIL 与 runtime 死锁。
        // panic 隔离：Rust panic 转为 RustPanicError。
        crate::shared::pyerror::detach_catch_panic(py, move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "failed to create runtime: {e}"
                    ))
                })?;
            rt.block_on(async move {
                let ctx = build_test_ctx();
                let result = tool
                    .execute(args_val, &ctx)
                    .await
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Python::attach(|py| toolresult_to_pyobject(py, &result))
            })
        })?
    }
}

/// 构造测试用 `ToolContext`（不依赖完整 agent loop）。
#[cfg(feature = "test-utils")]
fn build_test_ctx() -> ToolContext {
    use llm_harness_loop::test_utils::{NoOpEnv, test_assistant_message};
    ToolContext {
        run: Arc::new(RunContext::new(RunRequest::default())),
        env: Arc::new(NoOpEnv),
        abort: CancellationToken::new(),
        tool_use_id: "test".into(),
        turn_index: 0,
        assistant_message: Arc::new(test_assistant_message(vec![])),
        update_tx: tokio::sync::mpsc::channel(1).0,
    }
}

/// 将 `ToolResult` 转换为 Python dict。
#[cfg(feature = "test-utils")]
fn toolresult_to_pyobject(py: Python<'_>, result: &ToolResult) -> PyResult<Py<PyAny>> {
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
