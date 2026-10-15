"""End-to-end functional validation of the LLM-Attacker framework.

Not another unit test — a whole-framework layer that walks the real user
path and checks the pieces line up:

  1. registry integrity: every registered attack module loads through the
     plugin machinery, satisfies the module contract, and carries valid
     threat metadata (attacker/attacks/__init__.py + attack_metadata.py);
  2. CLI contract: `python -m attacker --list-attacks` / `--self-test`
     exit 0, invoked once each and shared across tests via fixtures;
  3. end-to-end synthetic runs: framework-native agentic modules driven
     through the real dispatcher write the documented artifact fields and
     land on their module contract (undefended worlds fire, defended
     worlds and the null control stay at floor);
  4. report layer: scorecard, coverage manifest, and the markdown
     execution report are all producible from the run record.

Torch-free by construction: nothing here imports torch and no test is
marked `slow`, so the file lands in the torch-free tier
(tests/conftest_plugin.py) and must pass with or without torch
installed. Attack modules are exercised only on their deterministic,
seeded, in-memory synthetic fixtures (the AgentWorld sandbox).
"""

import json
import os
import re
import subprocess
import sys

import pytest

from attacker import attack_metadata, coverage, report, scorecard, threat_model
from attacker.attacks import ATTACKS, load, validate_contract

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The documented artifact skeleton (attacker/artifacts.py docstring).
ARTIFACT_FIELDS = ("schema", "config", "threat_model", "provenance",
                   "results", "summary")

# Framework-native modules with deterministic synthetic fixtures, driven
# end-to-end below. The outcome contract is the module's own doctrine:
# attack arms must fire with every defense off, must be silent when
# defended, and the sanctioned control arm must execute everywhere.
E2E_ATTACKS = {
    "agent-goal-hijack": {
        "attack_arms": ("tool-output-hijack", "retrieved-document-hijack"),
        "control_arm": "benign-task",
        "firing_worlds": ("vulnerable", "none"),
        "floor_worlds": ("defended", "provenance-only"),
        "covered_threat": "ASI01",
    },
    "tool-misuse": {
        "attack_arms": ("dangerous-parameters", "scope-confusion",
                        "side-effect-amplification"),
        "control_arm": "authorized-task",
        "firing_worlds": ("vulnerable",),
        "floor_worlds": ("defended",),
        "covered_threat": "ASI02",
    },
}

SCORECARD_VERDICTS = {"PASS", "FAIL", "INFO", "ERROR",
                      "INSUFFICIENT_EVIDENCE"}


def run_cli(*args, timeout=600):
    """One dispatcher call with the repo root as cwd (the documented
    invocation); the caller asserts on the completed process."""
    return subprocess.run([sys.executable, "-m", "attacker", *args],
                          cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout)


@pytest.fixture(scope="module")
def cli_listing():
    """A single `--list-attacks` invocation shared by the CLI tests."""
    return run_cli("--list-attacks")


@pytest.fixture(scope="module")
def cli_self_test():
    """A single `--self-test` invocation shared by the CLI tests."""
    return run_cli("--self-test")


@pytest.fixture(scope="module")
def synthetic_runs(tmp_path_factory):
    """Both E2E attacks run once each through the real dispatcher
    (`python -m attacker --mode inference --attack NAME --toy --quick`),
    with the report piggyback flag exercising the post-run report hook."""
    runs = {}
    for name in E2E_ATTACKS:
        out = tmp_path_factory.mktemp("e2e") / f"{name}.json"
        r = run_cli("--mode", "inference", "--attack", name,
                    "--toy", "--quick", "--output", str(out), "--report")
        assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
        runs[name] = {"path": str(out),
                      "art": json.loads(out.read_text()),
                      "report_path": str(out) + ".report.md"}
    return runs


# ---------------------------------------------------------------------------
# 1. Registry integrity
# ---------------------------------------------------------------------------

def test_registry_is_non_empty_and_names_unique():
    assert ATTACKS, "the attack registry must not be empty"
    assert len(ATTACKS) == len(set(ATTACKS))
    assert all(isinstance(n, str) and n for n in ATTACKS)


def test_every_registered_module_loads_and_satisfies_the_contract():
    for name in sorted(ATTACKS):
        mod = load(name)  # load() re-validates the contract
        assert validate_contract(mod, name) is mod
        assert set(mod.MODES) <= {"training", "inference"}
        assert mod.DESCRIPTION.strip()
        # --help must work torch-less (repo rule): building the parser
        # and formatting help must not touch torch.
        assert mod.build_parser().format_help()


def test_experiment_ids_are_unique_across_the_registry():
    ids = [load(name).EXPERIMENT_ID for name in sorted(ATTACKS)]
    assert len(ids) == len(set(ids)), \
        f"duplicate EXPERIMENT_IDs: {sorted(i for i in set(ids) if ids.count(i) > 1)}"


def test_every_attack_declares_valid_threat_metadata():
    # Registry-wide: every entry schema-valid, no gaps, no orphans.
    covered = attack_metadata.validate_registry(list(ATTACKS))
    assert set(covered) == set(ATTACKS)
    for name in sorted(ATTACKS):
        md = attack_metadata.metadata_for(name)
        assert attack_metadata.validate_metadata(name, md) is md
        for field in ("actor_intent", "persistence", "surfaces",
                      "security_properties", "lifecycle_stages",
                      "owasp_llm_2026"):
            assert field in md, f"{name}: missing metadata field {field!r}"
        assert all(s in attack_metadata.SURFACES for s in md["surfaces"])


# ---------------------------------------------------------------------------
# 2. CLI contract (one invocation per flag, shared via the fixtures above)
# ---------------------------------------------------------------------------

def test_cli_list_attacks_exits_zero_and_covers_the_registry(cli_listing):
    r = cli_listing
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    # Name lines are `  {name:20s} [EXPERIMENT_ID] ...`; description
    # continuations are indented deeper and carry no bracketed id.
    listed = {m.group(1) for ln in r.stdout.splitlines()
              if (m := re.match(r"^  (\S+) +\[[^\]]+\]", ln))}
    assert listed == set(ATTACKS), \
        f"CLI/registry drift: {sorted(set(ATTACKS) ^ listed)}"


def test_cli_self_test_exits_zero_with_the_pass_marker(cli_self_test):
    r = cli_self_test
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    assert "FRAMEWORK SELF-TEST PASSED" in r.stdout
    assert "[FAIL]" not in r.stdout


# ---------------------------------------------------------------------------
# 3. End-to-end synthetic runs through the real dispatcher
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(E2E_ATTACKS))
def test_e2e_run_writes_the_documented_artifact(name, synthetic_runs):
    mod = load(name)
    run = synthetic_runs[name]
    art = run["art"]
    for field in ARTIFACT_FIELDS:
        assert field in art, f"{name}: artifact missing {field!r}"
    assert art["schema"] == mod.ARTIFACT_SCHEMA
    assert art["config"]["attack"] == mod.EXPERIMENT_ID
    assert art["config"]["toy"] is True
    assert isinstance(art["interpretation"], str) \
        and art["interpretation"].strip()
    threat_model.validate(art["threat_model"])
    for key in ("hostname", "timestamp_utc"):
        assert art["provenance"].get(key), \
            f"{name}: provenance missing {key!r}"
    # The piggyback --report hook produced the markdown execution report.
    assert os.path.isfile(run["report_path"])
    # Crash-safe per-cell journal: one well-formed record per cell.
    journal = run["path"] + ".jsonl"
    lines = (l for l in open(journal).read().splitlines() if l.strip())
    records = [json.loads(l) for l in lines]
    assert records and all(r.get("experiment") == mod.EXPERIMENT_ID
                           for r in records)


@pytest.mark.parametrize("name", sorted(E2E_ATTACKS))
def test_e2e_run_outcomes_match_the_module_contract(name, synthetic_runs):
    mod = load(name)
    spec = E2E_ATTACKS[name]
    art = synthetic_runs[name]["art"]
    cells = {(r.get("arm"), r.get("world")): r for r in art["results"]
             if "error" not in r}
    expected = {(a, w) for a in mod.ARMS
                for w in spec["firing_worlds"] + spec["floor_worlds"]}
    assert cells.keys() == expected, "every arm x world cell must land"
    for arm in spec["attack_arms"]:
        for world in spec["firing_worlds"]:
            cell = cells[(arm, world)]
            assert cell["total_violations"] > 0, \
                f"{name} {arm}/{world}: the undefended effect must fire"
            assert cell["violation_rate"] > 0
        for world in spec["floor_worlds"]:
            cell = cells[(arm, world)]
            assert cell["total_violations"] == 0, \
                f"{name} {arm}/{world}: defended world must be at floor"
            assert cell["false_blocks"] == 0
    # The sanctioned control arm executes in every world, never blocked.
    for world in spec["firing_worlds"] + spec["floor_worlds"]:
        cell = cells[(spec["control_arm"], world)]
        assert cell["total_violations"] == 0
        assert cell["false_blocks"] == 0
    # The summary mirrors the per-cell contract.
    summary = {r["experiment"]: r for r in art["summary"]}[mod.EXPERIMENT_ID]
    assert summary["vulnerable_total_violations"] > 0
    assert summary["defended_total_violations"] == 0
    assert summary["control_false_blocks"] == 0


# ---------------------------------------------------------------------------
# 4. Scorecard / coverage / report layer over the produced run records
# ---------------------------------------------------------------------------

def test_scorecard_aggregates_the_run_records(synthetic_runs, tmp_path):
    paths = [run["path"] for run in synthetic_runs.values()]
    card = scorecard.build_scorecard(paths)
    assert card["schema"] == scorecard.SCHEMA
    assert card["policy"] in scorecard.SCORECARD_POLICIES
    assert card["rows"], "scorecard must produce one row per summary cell"
    assert all(r["verdict"] in SCORECARD_VERDICTS for r in card["rows"])
    # These cells speak no excess-over-control metric -> the gate does not
    # apply (INFO), and nothing failed, so the run aggregates clean.
    assert card["summary"]["overall"] == "INFO"
    assert card["summary"]["n_fail"] == 0
    assert card["not_shown"] == []
    # The same aggregation through the CLI, writing scorecard + report.
    out = tmp_path / "scorecard.json"
    r = run_cli("--scorecard", *paths, "--output", str(out))
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    written = json.loads(out.read_text())
    assert written["schema"] == scorecard.SCHEMA
    assert written["summary"]["n_rows"] == card["summary"]["n_rows"]
    assert os.path.isfile(tmp_path / "scorecard.txt")


@pytest.mark.parametrize("name", sorted(E2E_ATTACKS))
def test_coverage_manifest_from_the_run_record(name, synthetic_runs):
    spec = E2E_ATTACKS[name]
    art = synthetic_runs[name]["art"]
    manifest = coverage.build_manifest([art], applicable_agentic=True)
    assert manifest["schema"] == coverage.SCHEMA
    threat = manifest["threats"][spec["covered_threat"]]
    assert threat["status"] == "COVERED"
    assert threat["measured_attacks"] == [name]
    assert name in manifest["attacks_measured"]
    assert all(t["status"] in coverage.STATUSES
               for t in manifest["threats"].values())


@pytest.mark.parametrize("name", sorted(E2E_ATTACKS))
def test_markdown_execution_report_structure(name, synthetic_runs):
    mod = load(name)
    run = synthetic_runs[name]
    md = report.render_report(run["path"])
    for section in ("# Execution report", "## Configuration",
                    "## Threat model", "## Results", "## Gate verdicts",
                    "## Evidence integrity", "## Reproduction"):
        assert section in md, f"{name}: report missing {section!r}"
    assert mod.EXPERIMENT_ID in md
    # The piggybacked report from the run itself carries the same shape.
    on_disk = open(run["report_path"]).read()
    assert "# Execution report" in on_disk and mod.EXPERIMENT_ID in on_disk
