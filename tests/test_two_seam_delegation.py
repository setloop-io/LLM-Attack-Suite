"""two-seam-delegation: torch-free toy tier (chained inversion across both
seams of a two-segment delegation world) — contract, registry/metadata
presence, end-to-end artifact+journal, the planted-world error ordering
(chained beats the compounded direct decode), the identity-middle null arm,
and the descriptive real-tier error."""

import json

import pytest

from attacker import artifacts
from attacker.attacks import two_seam_delegation as tsd


def parse(*argv):
    args = tsd.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


def test_module_contract_and_registry():
    from attacker import attack_metadata, attacks
    attacks.validate_contract(tsd, "two-seam-delegation")
    assert tsd.EXPERIMENT_ID == "two_seam_delegation"
    assert set(tsd.MODES) == {"training", "inference"}
    assert tsd.REQUIRES_LABELS is False
    assert tsd.SCHEMA == "dtraining.attacker.two_seam_delegation.v1"
    assert attacks.ATTACKS["two-seam-delegation"] == \
        "attacker.attacks.two_seam_delegation"
    assert attacks.load("two-seam-delegation") is tsd
    assert "two-seam-delegation" in attacks.for_mode("training")
    assert "two-seam-delegation" in attacks.for_mode("inference")
    md = attack_metadata.metadata_for("two-seam-delegation")
    attack_metadata.validate_metadata("two-seam-delegation", md)
    attack_metadata.validate_registry(list(attacks.ATTACKS))


def test_parser_builds_torchless():
    args = tsd.build_parser().parse_args([])
    assert args.middle == "both"
    assert args.evidence is False
    with pytest.raises(SystemExit):
        tsd.build_parser().parse_args(["--help"])
    with pytest.raises(SystemExit):
        tsd.build_parser().parse_args(["--middle", "nope"])


def test_toy_world_deterministic_and_shaped():
    w1 = tsd.toy_world(0, vocab=20, dim=6, n_seq=4, seq_len=5, sigma=0.1)
    w2 = tsd.toy_world(0, vocab=20, dim=6, n_seq=4, seq_len=5, sigma=0.1)
    assert w1["victim"] == w2["victim"]
    assert w1["seam1"] == w2["seam1"] and w1["seam2"] == w2["seam2"]
    assert len(w1["seam1"]) == 4 and len(w1["seam1"][0]) == 5
    assert len(w1["seam1"][0][0]) == 6 and len(w1["seam2"][0][0]) == 6
    assert len(w1["codebook"]) == 20 and len(w1["codebook"][0]) == 6
    wi = tsd.toy_world(0, vocab=20, dim=6, n_seq=2, seq_len=3,
                       middle="identity")
    assert wi["codebook"] == tsd.si._matmul(wi["E"], wi["H1"])


def test_chained_beats_compounded_direct_on_planted_world():
    args = parse("--toy")
    cells = tsd.toy_cell(args, 0.0, 0, "planted")
    direct = cells["direct-seam2"]["seq_accuracy"]
    chained = cells["chained"]["seq_accuracy"]
    control = cells["control-seam1"]["seq_accuracy"]
    assert chained >= direct
    assert chained > direct
    assert control >= chained
    for cell in cells.values():
        assert cell["gate"]["verdict"] in ("PASS", "FAIL")
        assert cell["embedding_mse"] >= 0.0


def test_identity_middle_null_arm_equality():
    args = parse("--toy")
    cells = tsd.toy_cell(args, 0.0, 0, "identity")
    chained = cells["chained"]
    control = cells["control-seam1"]
    assert chained["null_arm_ok"] is True
    assert abs(chained["seq_accuracy"] - control["seq_accuracy"]) \
        <= tsd.MNULL_TOL
    assert chained["embedding_mse"] == pytest.approx(
        control["embedding_mse"], abs=1e-6)
    assert cells["direct-seam2"]["seq_accuracy"] > 0.9


def test_no_args_descriptive_error():
    args = parse()
    with pytest.raises(SystemExit, match="--capture-dir"):
        tsd.run(args)


def test_real_tier_descriptive_error():
    args = parse("--capture-dir", "/nonexistent")
    with pytest.raises(SystemExit) as exc:
        tsd.run(args)
    msg = str(exc.value)
    assert "requires torch" in msg or "dual-seam" in msg


def test_toy_end_to_end_artifact_and_journal(tmp_path):
    out = tmp_path / "tsd.json"
    args = parse("--toy", "--quick", "--output", str(out))
    assert tsd.run(args) == 0
    art = json.loads(out.read_text())
    assert art["schema"] == tsd.SCHEMA
    assert len(art["results"]) == 12
    for cell in art["results"]:
        assert cell["experiment"] == tsd.EXPERIMENT_ID
        assert cell["arm"] in ("direct-seam2", "chained", "control-seam1")
        assert cell["gate"]["verdict"] in ("PASS", "FAIL")
    assert art["summary"]
    journal = out.with_suffix(".json.jsonl")
    lines = [json.loads(l) for l in journal.read_text().splitlines() if l]
    assert len(lines) == 12
    assert all(l["experiment"] == tsd.EXPERIMENT_ID for l in lines)


def test_self_test_passes():
    assert tsd.self_test() == 0
