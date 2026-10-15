"""End-to-end integration tripwire for the 15 threat-model-review attacks:
CLI dispatch (--mode/--attack), --toy --quick run, artifact + journal +
evidence chain + piggyback --report, and v2 threat-model validation."""

import json
import subprocess
import sys

import pytest

from attacker.evidence import verify_evidence
from attacker.threat_model import validate as validate_threat_model

NEW_ATTACKS = {
    "protocol-integrity": "inference",
    "multi-tenant-isolation": "inference",
    "artifact-telemetry-leak": "inference",
    "key-session-abuse": "inference",
    "resource-exhaustion": "inference",
    "poisoning-backdoor": "training",
    "malicious-client": "training",
    "supply-chain": "inference",
    "router-side-channel": "inference",
    "privacy-composition": "training",
    "prompt-injection": "inference",
    "hidden-context-exposure": "inference",
    "rag-vector-abuse": "inference",
    "output-sink-injection": "inference",
    "misinformation-integrity": "inference",
}


@pytest.mark.parametrize("attack,mode", sorted(NEW_ATTACKS.items()),
                         ids=sorted(NEW_ATTACKS))
def test_new_attack_end_to_end(tmp_path, attack, mode):
    out = tmp_path / f"{attack}.json"
    r = subprocess.run(
        [sys.executable, "-m", "attacker", "--mode", mode,
         "--attack", attack, "--toy", "--quick", "--output", str(out),
         "--evidence", "--report"],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"{attack} rc={r.returncode}: {r.stderr[-500:]}"
    art = json.loads(out.read_text())
    assert art["config"]["attack"]
    assert art["results"], "no cells ran"
    assert all("experiment" in rec for rec in art["results"])
    validate_threat_model(art["threat_model"])
    journal = out.with_suffix(out.suffix + ".jsonl")
    assert journal.exists() and journal.read_text().strip()
    assert (tmp_path / f"{attack}.json.report.md").exists()
    ok, problems = verify_evidence(str(out) + ".evidence.json")
    assert ok and not problems


def test_new_attacks_are_registered_with_metadata():
    from attacker.attacks import ATTACKS
    from attacker.attack_metadata import metadata_for
    for name in NEW_ATTACKS:
        assert name in ATTACKS
        metadata_for(name)
