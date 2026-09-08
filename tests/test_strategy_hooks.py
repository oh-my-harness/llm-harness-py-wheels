"""Tests for strategy preset hooks: vision_degrade, observation_shielding."""

import pytest
import senza

# ── vision_degrade ──────────────────────────────────────────────────────────


def test_strategy_namespace_has_vision_degrade():
    assert hasattr(senza.strategy, "vision_degrade")


def test_vision_degrade_returns_hook():
    hook = senza.strategy.vision_degrade()
    assert isinstance(hook, senza.Hook)


# ── observation_shielding ───────────────────────────────────────────────────


def test_strategy_namespace_has_observation_shielding():
    assert hasattr(senza.strategy, "observation_shielding")


def test_observation_shielding_returns_hook_default():
    hook = senza.strategy.observation_shielding()
    assert isinstance(hook, senza.Hook)


def test_observation_shielding_accepts_config():
    hook = senza.strategy.observation_shielding({"retained_turns": 3, "placeholder": "[hidden]"})
    assert isinstance(hook, senza.Hook)


def test_observation_shielding_accepts_partial_config():
    hook = senza.strategy.observation_shielding({"retained_turns": 2})
    assert isinstance(hook, senza.Hook)


def test_observation_shielding_rejects_negative_turns():
    with pytest.raises(ValueError):
        senza.strategy.observation_shielding({"retained_turns": -1})
