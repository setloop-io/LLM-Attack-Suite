"""ica (attacker/ica.py): the FastICA module is torch-only by design (the
repo venvs ship no sklearn, so the framework carries its own torch FastICA).
What is pinned torch-free — the repo rule this file must keep: the module
imports, --self-test runs (reporting SKIPPED for the torch checks), and the
numerical entry points refuse non-torch input instead of answering silently.
The convergence/recovery known-answer checks live in ica.self_test()'s torch
branch and run on torch hosts."""

import runpy
import warnings

import pytest

from attacker import ica


def test_module_imports_torch_less():
    if ica.torch is not None:
        pytest.skip("torch installed — torch-absent behavior not applicable")
    assert callable(ica.whiten) and callable(ica.fastica)
    assert callable(ica.match_components)


def test_self_test_reports_torch_checks_skipped_and_passes(capsys):
    if ica.torch is not None:
        pytest.skip("torch installed — the torch checks run instead of SKIPPED")
    assert ica.self_test() == 0
    out = capsys.readouterr().out
    assert "SKIPPED" in out and "SELF-TEST PASSED" in out


def test_entry_points_refuse_non_torch_input():
    with pytest.raises(AttributeError):
        ica.whiten([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(AttributeError):
        ica.fastica([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(AttributeError):
        ica.match_components([[1.0], [2.0]], [[1.0], [2.0]])


def test_module_runs_standalone_torchless():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("attacker.ica", run_name="__main__")
    assert exc.value.code == 0
