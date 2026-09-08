//! Vision degrade / observation shielding 的 Python 包装。
//!
//! - `vision_degrade()`：text-only provider 拒绝含视觉块的请求
//!   （`InvalidRequest`，HTTP 400）时自动剥离 image/document 块并同轮重试
//!   （每 run 最多修复一次，防循环）。
//! - `observation_shielding(config)`：将旧 assistant turn 的 tool
//!   observation 替换为占位文本，只保留最近 N 个 turn 的观测。

use pyo3::prelude::*;

use crate::core::pyhooks::PyHookWrapper;

/// 创建 `VisionDegradeHook`（provider_error hook 的 preset）。
///
/// 注册方式：`builder.provider_error_hook(senza.strategy.vision_degrade())`。
/// 无参数——修复策略固定：仅处理 `InvalidRequest`，剥离 image/document 块，
/// 每 run 一次。
#[pyfunction]
pub fn create_vision_degrade_hook<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyHookWrapper>> {
    let hook: std::sync::Arc<dyn llm_harness_types::ProviderErrorHook> =
        std::sync::Arc::new(llm_harness_strategy::VisionDegradeHook::new());
    Py::new(
        py,
        PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::ProviderError(hook),
        },
    )
    .map(|p| p.into_bound(py))
}

/// 创建 `ObservationShieldingHook`（transform_context hook 的 preset）。
///
/// config 键（均可选）：
/// - `retained_turns: int` —— 保留最近 N 个 assistant turn 的观测，默认 5
/// - `placeholder: str` —— 旧观测的替换文本
///
/// 注册方式：`builder.hooks([senza.strategy.observation_shielding(...)])`
/// 或包进 plugin。
#[pyfunction]
#[pyo3(signature = (config=None))]
pub fn create_observation_shielding_hook<'py>(
    py: Python<'py>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyHookWrapper>> {
    let mut cfg = llm_harness_strategy::ObservationShieldingConfig::default();
    if let Some(c) = config {
        let v = c.getattr("get")?.call1(("retained_turns", py.None()))?;
        if !v.is_none() {
            let n: i64 = v.extract()?;
            if n < 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "retained_turns must be >= 0",
                ));
            }
            cfg.retained_turns = n as usize;
        }
        let v = c.getattr("get")?.call1(("placeholder", py.None()))?;
        if !v.is_none() {
            cfg.placeholder = v.extract::<String>()?;
        }
    }
    let hook: std::sync::Arc<dyn llm_harness_types::TransformContextHook> =
        std::sync::Arc::new(llm_harness_strategy::ObservationShieldingHook::new(cfg));
    Py::new(
        py,
        PyHookWrapper {
            kind: crate::core::pyhooks::HookKind::TransformContext(hook),
        },
    )
    .map(|p| p.into_bound(py))
}
