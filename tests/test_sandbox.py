"""Tests for Sandbox binding."""

from __future__ import annotations

import platform

import pytest
import senza


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="SeatbeltSandbox is only available on macOS",
)
def test_seatbelt_sandbox_creates():
    sandbox = senza.infra.seatbelt_sandbox()
    assert sandbox is not None
    assert sandbox.is_running() is False


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="SeatbeltSandbox is only available on macOS",
)
def test_seatbelt_sandbox_creates_with_config():
    """Supported config keys are accepted; unsupported resource limits are rejected."""
    # Supported: fs_allowlist, work_dir, timeout_seconds.
    sandbox = senza.infra.seatbelt_sandbox(
        {
            "fs_allowlist": ["/tmp"],
            "work_dir": "/tmp/sandbox",
            "timeout_seconds": 30.0,
        }
    )
    assert sandbox is not None
    assert sandbox.is_running() is False

    # Unsupported resource limits are rejected at construction (fail-closed).
    for key, val in [("max_cpus", 2), ("max_memory_mb", 512), ("max_disk_mb", 256)]:
        with pytest.raises(RuntimeError, match=key):
            senza.infra.seatbelt_sandbox({key: val})


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="SeatbeltSandbox is only available on macOS",
)
def test_seatbelt_sandbox_start_succeeds():
    """SeatbeltSandbox::start() probes sandbox-exec and succeeds on macOS."""
    sandbox = senza.infra.seatbelt_sandbox()
    sandbox.start()
    assert sandbox.is_running() is True


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="SeatbeltSandbox is only available on macOS",
)
def test_seatbelt_sandbox_accepts_max_processes():
    """max_processes is parsed and accepted (enforced on Linux bwrap only)."""
    sandbox = senza.infra.seatbelt_sandbox({"max_processes": 100})
    assert sandbox is not None
    assert sandbox.is_running() is False


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="BwrapSandbox is only available on Linux",
)
def test_bwrap_sandbox_accepts_max_processes():
    """max_processes is parsed and accepted on Linux (cgroup v2 pids limit)."""
    sandbox = senza.infra.bwrap_sandbox({"max_processes": 100})
    assert sandbox is not None
