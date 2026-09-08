from __future__ import annotations

import json
from types import SimpleNamespace

from academy.scenarios import runner
from academy.scenarios.catalog import load_catalog


def test_catalog_covers_all_legacy_scripts_and_expected_quarantine():
    catalog = load_catalog()

    assert len(catalog) == 43
    assert len({scenario.id for scenario in catalog}) == 43
    assert {
        scenario.id for scenario in catalog if scenario.tier == "quarantined"
    } == {
        "plugin.notify",
        "safety.rules_approval",
        "workflow.data_analysis",
        "workflow.pause_cancel",
        "workflow.shell_executor",
    }
    assert catalog.resolve("02_tool_calling").id == "agent.tool_calling"
    assert catalog.resolve("live-tests/examples/07_hooks.py").id == "agent.hooks"


def test_catalog_supplies_longer_budgets_for_expensive_scenarios():
    catalog = load_catalog()

    assert catalog.resolve("context.compaction").timeout_seconds == 600
    assert catalog.resolve("multi_agent.spawn").timeout_seconds == 420
    assert catalog.resolve("agent.tool_calling").timeout_seconds == 120
    assert set(catalog.resolve("provider.multi_provider").requirements["env"]) == {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    }


def test_missing_provider_skips_without_starting_a_subprocess(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    invoked = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: invoked.append(args))

    status = runner.run_scenario(
        load_catalog().resolve("agent.tool_calling"), json_output=True
    )

    assert status == 0
    assert invoked == []
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "skipped"
    assert payload["reasons"][0]["code"] == "missing-provider"


def test_missing_senza_module_is_reported_before_starting_subprocess(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(runner, "_module_available", lambda name: False)
    invoked = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: invoked.append(args))

    status = runner.run_scenario(
        load_catalog().resolve("agent.tool_calling"), json_output=True
    )

    assert status == 0
    assert invoked == []
    payload = json.loads(capsys.readouterr().out)
    assert {reason["code"] for reason in payload["reasons"]} == {
        "missing-python-module"
    }


def test_quarantined_scenario_requires_explicit_opt_in(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    invoked = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: invoked.append(args))

    status = runner.run_scenario(
        load_catalog().resolve("workflow.pause_cancel"), json_output=True
    )

    assert status == 2
    assert invoked == []
    assert json.loads(capsys.readouterr().out)["status"] == "refused"


def test_json_subprocess_output_redacts_secret_values(monkeypatch, capsys):
    secret = "sk-test-secret-value-123456"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setattr(runner, "_module_available", lambda name: True)

    def fake_run(*args, **kwargs):
        assert kwargs["timeout"] == 120.0
        return SimpleNamespace(
            returncode=0,
            stdout="provider echoed {}".format(secret),
            stderr="",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    status = runner.run_scenario(
        load_catalog().resolve("agent.tool_calling"), json_output=True
    )

    output = capsys.readouterr().out
    assert status == 0
    assert secret not in output
    assert "REDACTED" in output


def test_non_finite_timeout_is_a_structured_cli_error(capsys):
    status = runner.main(
        ["run", "agent.tool_calling", "--timeout", "inf", "--json"]
    )

    assert status == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "finite" in payload["error"]


def test_course_entrypoint_runs_recorded_and_resolves_live_primary(monkeypatch, capsys):
    catalog = load_catalog()
    recorded_calls = []

    def fake_recorded(lab, *, timeout, json_output):
        recorded_calls.append((lab["id"], timeout, json_output))
        return 0

    monkeypatch.setattr(runner, "_run_recorded_lab", fake_recorded)
    assert runner.main(["course", "1", "--mode", "recorded"]) == 0
    assert recorded_calls == [("01", 120.0, False)]

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert runner.main(["course", "01", "--mode", "live", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenario_id"] == catalog.resolve("agent.tool_calling").id
    assert payload["status"] == "skipped"
