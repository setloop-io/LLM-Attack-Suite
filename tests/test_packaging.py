"""Packaging tests: pyproject.toml metadata, entry points, LICENSE, and
the console-script target. Pure stdlib."""

import os
import tomllib

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def pyproject():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)


def test_pyproject_parses_and_core_metadata(pyproject):
    project = pyproject["project"]
    assert project["name"] == "llm-attacker"
    assert project["version"] == "1.0.0"
    assert project["requires-python"] == ">=3.12"
    assert project.get("dependencies", []) == []


def test_build_backend_declared(pyproject):
    bs = pyproject["build-system"]
    assert "setuptools" in " ".join(bs["requires"])
    assert bs["build-backend"] == "setuptools.build_meta"


def test_optional_extras(pyproject):
    extras = pyproject["project"]["optional-dependencies"]
    assert extras["torch"] == ["torch"]
    assert extras["gpu"] == ["pynvml"]
    assert extras["dev"] == ["pytest", "numpy"]


def test_console_script_declared(pyproject):
    scripts = pyproject["project"]["scripts"]
    assert scripts["llm-attacker"] == "attacker.__main__:main"


def test_attack_entry_point_group_declared(pyproject):
    groups = pyproject["project"]["entry-points"]
    assert "llm_attacker.attacks" in groups
    assert groups["llm_attacker.attacks"] == {}


def test_console_script_target_importable_and_callable():
    from attacker.__main__ import main
    assert callable(main)
    assert main(["--list-attacks"]) == 0


def test_license_file():
    path = os.path.join(ROOT, "LICENSE")
    assert os.path.isfile(path)
    text = open(path).read()
    assert "MIT License" in text
    assert "Permission is hereby granted, free of charge" in text
    assert "THE SOFTWARE IS PROVIDED" in text


def test_installed_console_script_metadata():
    """After `pip install -e .`, the script is in the installed metadata."""
    md = pytest.importorskip("importlib.metadata")
    try:
        eps = md.entry_points(group="console_scripts")
    except TypeError:
        eps = [ep for ep in md.entry_points().get("console_scripts", [])]
    names = {ep.name for ep in eps}
    if "llm-attacker" not in names:
        pytest.skip("llm-attacker not installed in this environment")
    ep = [ep for ep in eps if ep.name == "llm-attacker"][0]
    assert ep.value == "attacker.__main__:main"
    assert callable(ep.load())
