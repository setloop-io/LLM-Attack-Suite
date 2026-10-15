"""Attack registry: contract validation, entry-point discovery, register_attack."""

import argparse
import importlib.metadata
import types

import pytest

from attacker import attacks
from attacker.attacks import (ATTACKS, ENTRY_POINT_GROUP, discover_plugins,
                              for_mode, load, register_attack,
                              validate_contract)


def make_fake_attack(experiment_id="pytest_fake"):
    return types.SimpleNamespace(
        EXPERIMENT_ID=experiment_id, MODES=("training",),
        REQUIRES_LABELS=False, DESCRIPTION="pytest fake attack",
        build_parser=lambda: argparse.ArgumentParser(),
        run=lambda args: 0)


class FakeEP:
    def __init__(self, name, obj):
        self.name, self._obj = name, obj

    def load(self):
        return self._obj


@pytest.fixture
def cleanup_registry():
    before = set(ATTACKS)
    yield
    for name in set(ATTACKS) - before:
        ATTACKS.pop(name, None)


def test_all_builtins_satisfy_contract():
    for name in list(ATTACKS):
        assert validate_contract(load(name), name) is load(name)


def test_validate_contract_error_messages():
    bad = types.SimpleNamespace(EXPERIMENT_ID="bad", MODES=("training",),
                                REQUIRES_LABELS=True, DESCRIPTION="x")
    with pytest.raises(ValueError) as exc:
        validate_contract(bad, "bad-fake")
    msg = str(exc.value)
    assert "bad-fake" in msg and "build_parser" in msg and "run" in msg

    good_shape = make_fake_attack()
    validate_contract(good_shape)

    for field, value in (("EXPERIMENT_ID", ""), ("MODES", ("sideways",)),
                         ("MODES", ()), ("REQUIRES_LABELS", None),
                         ("DESCRIPTION", None)):
        broken = make_fake_attack()
        setattr(broken, field, value)
        with pytest.raises(ValueError, match=field):
            validate_contract(broken, "broken")


def test_register_attack_programmatic(cleanup_registry):
    fake = make_fake_attack()
    register_attack("pytest-plugin", fake)
    assert ATTACKS["pytest-plugin"] is fake
    assert load("pytest-plugin") is fake
    assert "pytest-plugin" in for_mode("training")
    assert "pytest-plugin" not in for_mode("inference")

    with pytest.raises(ValueError, match="pytest-plugin"):
        register_attack("pytest-plugin", fake)
    with pytest.raises(ValueError, match="accumulation"):
        register_attack("accumulation", fake)
    with pytest.raises(ValueError, match="contract"):
        register_attack("pytest-broken", types.SimpleNamespace())


def test_discover_plugins_fake_entry_points(cleanup_registry):
    fake = make_fake_attack()
    added = discover_plugins([FakeEP("pytest-ep", fake),
                              FakeEP("pytest-ep-factory", lambda: fake)])
    assert sorted(added) == ["pytest-ep", "pytest-ep-factory"]
    assert load("pytest-ep") is fake
    assert load("pytest-ep-factory") is fake


def test_discover_plugins_collision_is_descriptive(cleanup_registry):
    with pytest.raises(RuntimeError) as exc:
        discover_plugins([FakeEP("wire-eval", make_fake_attack())])
    assert "wire-eval" in str(exc.value) and ENTRY_POINT_GROUP in str(exc.value)


def test_discover_plugins_load_failure_is_descriptive(cleanup_registry):
    class BrokenEP(FakeEP):
        def load(self):
            raise ImportError("no such plugin module")

    with pytest.raises(RuntimeError, match="pytest-broken"):
        discover_plugins([BrokenEP("pytest-broken", None)])


def test_discover_plugins_contract_violation_is_descriptive(cleanup_registry):
    with pytest.raises(ValueError, match="pytest-invalid"):
        discover_plugins([FakeEP("pytest-invalid", object())])


def test_discovery_degrades_gracefully_without_entry_points():
    assert discover_plugins([]) == []


def test_discovery_via_importlib_metadata(monkeypatch, cleanup_registry):
    fake = make_fake_attack()
    monkeypatch.setattr(importlib.metadata, "entry_points",
                        lambda group=None: [FakeEP("pytest-meta", fake)]
                        if group == ENTRY_POINT_GROUP else [])
    assert attacks.discover_plugins() == ["pytest-meta"]
    assert load("pytest-meta") is fake

    monkeypatch.setattr(importlib.metadata, "entry_points",
                        lambda group=None: (_ for _ in ()).throw(
                            RuntimeError("broken metadata")))
    assert attacks.discover_plugins() == []


def test_load_and_for_mode_unchanged_for_builtins():
    mod = load("accumulation")
    assert mod.EXPERIMENT_ID == "er1a_accumulation"
    assert "accumulation" in for_mode("training")
