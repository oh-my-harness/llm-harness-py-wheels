//! Sandbox PyO3 binding — wraps `SeatbeltSandbox` (macOS) and `BwrapSandbox` (Linux).
//!
//! The `Sandbox` trait has `shutdown(self: Box<Self>)` which prevents calling
//! it through `Arc<dyn Sandbox>`. However, `Arc<dyn Sandbox>` still compiles
//! because `shutdown` simply can't be invoked through `Arc` — and we don't
//! expose it. We store the concrete backend type behind `Arc<dyn Sandbox>`
//! for uniform `is_running()` / `start()` / `config()` access.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use llm_harness_platform::sandbox::{ResourceLimits, Sandbox, SandboxConfig};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::core::pyagent::runtime;
use crate::shared::pyerror::detach_catch_panic_result;

/// Parse a Python dict into a `SandboxConfig`.
///
/// Recognized keys (all optional):
/// - `fs_allowlist`: list[str]
/// - `fs_denylist`: list[str]
/// - `work_dir`: str | None
/// - `max_memory_mb`: int | None
/// - `max_cpus`: int | None
/// - `max_processes`: int | None (enforced on Linux bwrap only; seatbelt ignores it)
/// - `max_disk_mb`: int | None
/// - `timeout_seconds`: float | None
#[cfg_attr(not(any(target_os = "linux", target_os = "macos")), allow(dead_code))]
pub(crate) fn dict_to_sandbox_config(
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<SandboxConfig> {
    let mut fs_allowlist = Vec::new();
    let mut fs_denylist = Vec::new();
    let net_allowlist = Vec::new();
    let mut max_cpus = None;
    let mut max_processes = None;
    let mut max_memory_mb = None;
    let mut max_disk_mb = None;
    let mut timeout = None;
    let mut work_dir = None;

    if let Some(cfg) = config {
        if let Some(v) = cfg.get_item("fs_allowlist")? {
            let iter = v.try_iter()?;
            for item in iter {
                let item = item?;
                fs_allowlist.push(PathBuf::from(item.extract::<String>()?));
            }
        }
        if let Some(v) = cfg.get_item("fs_denylist")? {
            let iter = v.try_iter()?;
            for item in iter {
                let item = item?;
                fs_denylist.push(PathBuf::from(item.extract::<String>()?));
            }
        }
        if let Some(v) = cfg.get_item("work_dir")?
            && !v.is_none()
        {
            work_dir = Some(PathBuf::from(v.extract::<String>()?));
        }
        if let Some(v) = cfg.get_item("max_memory_mb")?
            && !v.is_none()
        {
            max_memory_mb = Some(v.extract::<usize>()?);
        }
        if let Some(v) = cfg.get_item("max_cpus")?
            && !v.is_none()
        {
            max_cpus = Some(v.extract::<usize>()?);
        }
        if let Some(v) = cfg.get_item("max_processes")?
            && !v.is_none()
        {
            max_processes = Some(v.extract::<usize>()?);
        }
        if let Some(v) = cfg.get_item("max_disk_mb")?
            && !v.is_none()
        {
            max_disk_mb = Some(v.extract::<usize>()?);
        }
        if let Some(v) = cfg.get_item("timeout_seconds")?
            && !v.is_none()
        {
            let secs = v.extract::<f64>()?;
            timeout = Some(Duration::from_secs_f64(secs));
        }
    }

    Ok(SandboxConfig {
        fs_allowlist,
        fs_denylist,
        net_allowlist,
        resource_limits: ResourceLimits {
            max_cpus,
            max_processes,
            max_memory_mb,
            max_disk_mb,
            timeout,
        },
        work_dir,
    })
}

/// Python-side wrapper for a sandbox backend.
///
/// Wraps a platform-specific sandbox (`SeatbeltSandbox` on macOS,
/// `BwrapSandbox` on Linux) behind the `Sandbox` trait.
#[pyclass(name = "Sandbox")]
pub struct PySandbox {
    inner: Arc<dyn Sandbox>,
}

#[pymethods]
impl PySandbox {
    /// Whether the sandbox is currently running.
    fn is_running(&self) -> bool {
        self.inner.is_running()
    }

    /// Start or verify the sandbox is ready.
    ///
    /// Blocks on the tokio runtime with panic isolation. Raises
    /// `RuntimeError` if the sandbox fails to start.
    fn start(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        let rt = runtime(py);
        detach_catch_panic_result(py, move || rt.block_on(async move { inner.start().await }))?;
        Ok(())
    }
}

// ── Factory functions ─────────────────────────────────────────────────────

/// Create a `SeatbeltSandbox` (macOS only).
#[cfg(target_os = "macos")]
#[pyfunction]
#[pyo3(signature = (config=None))]
pub fn create_seatbelt_sandbox<'py>(
    py: Python<'py>,
    config: Option<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PySandbox>> {
    let cfg = dict_to_sandbox_config(config.as_ref())?;
    let sandbox = llm_harness_sandbox::SeatbeltSandbox::new(cfg)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Py::new(
        py,
        PySandbox {
            inner: Arc::new(sandbox),
        },
    )
    .map(|p| p.into_bound(py))
}

/// Create a `BwrapSandbox` (Linux only).
#[cfg(target_os = "linux")]
#[pyfunction]
#[pyo3(signature = (config=None))]
pub fn create_bwrap_sandbox<'py>(
    py: Python<'py>,
    config: Option<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PySandbox>> {
    let cfg = dict_to_sandbox_config(config.as_ref())?;
    let sandbox = llm_harness_sandbox::BwrapSandbox::new(cfg)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Py::new(
        py,
        PySandbox {
            inner: Arc::new(sandbox),
        },
    )
    .map(|p| p.into_bound(py))
}
