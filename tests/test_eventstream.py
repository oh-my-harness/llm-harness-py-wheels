"""Tests for human approval/input channels."""

import pytest
import senza

# ── factory surface ─────────────────────────────────────────────────────────


def test_create_human_approval_channel():
    handle, tool = senza.create_human_approval_channel("deploy-gate")
    assert type(handle).__name__ == "HumanResponseHandle"
    assert type(tool).__name__ == "HumanApprovalTool"
    assert tool.name() == "request_human_approval"
    assert isinstance(tool.description(), str)


def test_create_human_input_channel():
    handle, tool = senza.create_human_input_channel("clarify-1")
    assert type(handle).__name__ == "HumanResponseHandle"
    assert type(tool).__name__ == "HumanInputTool"
    assert tool.name() == "request_human_input"
    assert isinstance(tool.description(), str)


def test_human_channels_accept_options():
    senza.create_human_approval_channel("g1", timeout_seconds=1.5, default="approve")
    senza.create_human_input_channel("g2", timeout_seconds=2.0, default={"a": 1})


def test_human_approval_bad_default_rejected():
    with pytest.raises(ValueError):
        senza.create_human_approval_channel("g3", default="maybe")


def test_human_handle_submit_before_request_raises():
    """submit() before the tool issued a request is a clear error, not a hang."""
    handle, _tool = senza.create_human_approval_channel("g4")
    with pytest.raises(RuntimeError, match="request"):
        handle.submit("approved", {"decision": "approve"})
