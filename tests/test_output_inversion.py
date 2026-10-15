"""output-inversion attack module (attacker/attacks/output_inversion.py):
the torch-free delegation path (split-training/output_inversion.py) end to
end — command construction, artifact wrapping, return-code propagation —
plus the fail-closed guards. The --toy decode loop (lines 95-130) is
torch-gated and stays uncovered here.

The delegated script and subprocess are faked at the module's own
namespace boundary (the sibling split-training/ tree is absent from this
checkout), so the module's real delegation wiring is what gets exercised.
"""

import json
import os
import types

import pytest

from attacker import attacks
from attacker.attacks import output_inversion as oinv


def _parse(*argv):
    args = oinv.build_parser().parse_args(list(argv))
    args.mode = "inference"
    return args


class _FakeOsPath:
    """os.path stand-in: the (absent) delegation script 'exists';
    everything else delegates to the real os.path."""
    dirname = staticmethod(os.path.dirname)
    abspath = staticmethod(os.path.abspath)
    join = staticmethod(os.path.join)

    @staticmethod
    def exists(path):
        return path.endswith("output_inversion.py") or os.path.exists(path)


def _install_delegation_fakes(monkeypatch, rc=0, payload=None):
    """Fake the script's existence and the subprocess run; the fake
    'script' writes its result JSON to the --output it is given."""
    calls = []

    def fake_run(cmd):
        calls.append(list(cmd))
        if rc == 0:
            target = cmd[cmd.index("--output") + 1]
            with open(target, "w") as fh:
                json.dump(payload or {"decode_top1": 97.5, "engine":
                                      "delegated-script"}, fh)
        return types.SimpleNamespace(returncode=rc)

    monkeypatch.setattr(oinv, "os", types.SimpleNamespace(path=_FakeOsPath))
    monkeypatch.setattr(oinv, "subprocess",
                        types.SimpleNamespace(run=fake_run))
    return calls


def test_contract_and_registry():
    assert oinv.EXPERIMENT_ID == "output_inversion"
    assert oinv.MODES == ("inference",)
    assert oinv.REQUIRES_LABELS is False
    attacks.validate_contract(oinv, "output-inversion")
    assert attacks.ATTACKS.get("output-inversion") == \
        "attacker.attacks.output_inversion"
    assert attacks.load("output-inversion") is oinv


def test_parser_defaults():
    args = _parse()
    assert args.output_inversion_script is None
    assert args.toy is False and args.seed == 1234


def test_missing_delegation_script_is_descriptive(tmp_path):
    with pytest.raises(SystemExit, match="output_inversion.py not found"):
        oinv.run(_parse("--output-inversion-script", "--model M",
                        "--output", str(tmp_path / "o.json")))


def test_delegation_requires_an_output_path(tmp_path, monkeypatch):
    _install_delegation_fakes(monkeypatch)
    with pytest.raises(SystemExit, match="pass --output"):
        oinv.run(_parse("--output-inversion-script", "--model M"))


def test_toy_path_requires_torch(tmp_path):
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch present: the guard does not fire")
    with pytest.raises(SystemExit, match="requires torch"):
        oinv.run(_parse("--toy", "--output", str(tmp_path / "o.json")))


def test_real_path_without_delegation_is_descriptive(tmp_path, monkeypatch):
    monkeypatch.setattr("attacker.attacks.common.torch", object())
    with pytest.raises(SystemExit, match="pass --toy"):
        oinv.run(_parse("--output", str(tmp_path / "o.json")))


def test_delegation_happy_path_wraps_the_script_output(tmp_path, monkeypatch):
    calls = _install_delegation_fakes(monkeypatch,
                                      payload={"decode_top1": 97.5,
                                               "engine": "delegated-script"})
    out = tmp_path / "oinv.json"
    args = _parse("--output-inversion-script", "--model M --steps 3",
                  "--output", str(out))
    assert oinv.run(args) == 0
    (cmd,) = calls
    assert cmd[1].endswith(os.path.join("split-training",
                                        "output_inversion.py"))
    assert cmd[2:4] == ["--model", "M"]
    assert cmd[-2:] == ["--output", str(out) + ".oinv.json"]
    art = json.loads(out.read_text())
    assert art["schema"] == "dtraining.attacker.output_inversion.v1"
    assert art["config"]["attack"] == "output_inversion"
    assert art["config"]["delegated"] is True
    assert art["config"]["script_args"] == "--model M --steps 3"
    assert art["results"] == [{"decode_top1": 97.5,
                               "engine": "delegated-script"}]


def test_delegation_respects_script_provided_output(tmp_path, monkeypatch):
    custom = tmp_path / "custom.json"
    calls = _install_delegation_fakes(monkeypatch)
    out = tmp_path / "oinv2.json"
    args = _parse("--output-inversion-script",
                  f"--model M --output {custom}",
                  "--output", str(out))
    assert oinv.run(args) == 0
    (cmd,) = calls
    assert cmd.count("--output") == 1
    assert custom.exists()
    assert json.loads(out.read_text())["results"] == [
        {"decode_top1": 97.5, "engine": "delegated-script"}]


def test_delegation_propagates_script_failure(tmp_path, monkeypatch):
    _install_delegation_fakes(monkeypatch, rc=7)
    out = tmp_path / "oinv_fail.json"
    rc = oinv.run(_parse("--output-inversion-script", "--model M",
                         "--output", str(out)))
    assert rc == 7
    assert not out.exists()
