"""wire-shape (passive traffic-shape analysis): provider registration +
schema, toy end-to-end (length-leak MAE, phase-detect F1), journal input
path, evidence chain, dispatch. All torch-free.

NOTE: importing attacker.attacks.wire_shape registers the "wire-shape"
surface provider at import (by design). To keep the surface registry
pristine for the rest of the session, this module imports lazily inside a
fixture and pops the registration it caused on teardown."""

import importlib
import json
import os
import sys

import pytest

from attacker import surfaces
from attacker.evidence import verify_evidence


@pytest.fixture
def wire_shape_mod():
    fresh = "attacker.attacks.wire_shape" not in sys.modules
    mod = importlib.import_module("attacker.attacks.wire_shape")
    present = "wire-shape" in surfaces.list_surfaces()
    if not present:
        surfaces.register_surface(mod.WireShapeProvider())
    cleanup = fresh or not present
    yield mod
    if cleanup:
        surfaces._REGISTRY.pop("wire-shape", None)


@pytest.fixture
def registered(wire_shape_mod):
    """Dispatch through the real registry without editing ATTACKS."""
    from attacker import attacks
    registered_here = "wire-shape" not in attacks.ATTACKS
    if registered_here:
        attacks.register_attack("wire-shape", wire_shape_mod)
    yield wire_shape_mod
    if registered_here:
        attacks.ATTACKS.pop("wire-shape", None)


def _run_toy(mod, tmp_path, *extra, mode="inference", name="ws.json"):
    args = mod.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(tmp_path / name), *extra])
    args.mode = mode
    assert mod.run(args) == 0
    art = json.loads((tmp_path / name).read_text())
    journal = [json.loads(l) for l in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def test_contract_and_torchless_help(wire_shape_mod):
    from attacker.attacks import validate_contract
    assert validate_contract(wire_shape_mod) is wire_shape_mod
    assert wire_shape_mod.MODES == ("training", "inference")
    assert wire_shape_mod.REQUIRES_LABELS is False
    ap = wire_shape_mod.build_parser()
    with pytest.raises(SystemExit) as exc:
        ap.parse_args(["--help"])
    assert exc.value.code == 0


def test_self_test_passes(wire_shape_mod):
    assert wire_shape_mod.self_test() == 0


def test_surface_provider_registered_and_validates(wire_shape_mod):
    prov = surfaces.get_surface("wire-shape")
    assert isinstance(prov, wire_shape_mod.WireShapeProvider)
    assert set(prov.schema) == {"ts", "bytes", "dir", "session_id"}
    assert prov.modes == ("training", "inference")
    assert prov.validate({"ts": 1, "bytes": 4, "dir": "c2s"}) == {
        "ts": 1.0, "bytes": 4, "dir": "c2s", "session_id": None}
    assert prov.validate({"ts": 1.5, "bytes": 4, "dir": "s2c",
                          "session_id": "s"})["session_id"] == "s"
    with pytest.raises(ValueError, match="dir"):
        prov.validate({"ts": 1.0, "bytes": 4, "dir": "sideways"})
    with pytest.raises(ValueError, match="schema"):
        prov.validate({"ts": 1.0, "bytes": 4})
    with pytest.raises(ValueError, match="bytes"):
        prov.validate({"ts": 1.0, "bytes": -1, "dir": "c2s"})
    with pytest.raises(ValueError, match="ts"):
        prov.validate({"ts": "now", "bytes": 4, "dir": "c2s"})
    with pytest.raises(ValueError, match="mode"):
        prov.rows(mode="sideways", source="x")


def test_surface_rows_over_journal(wire_shape_mod, tmp_path):
    jf = tmp_path / "wire.jsonl"
    recs = [{"ts": 100.0 + i * 0.01, "bytes": 4, "dir": "s2c",
             "session_id": "s0"} for i in range(5)]
    recs.append({"ts": 100.0, "bytes": 128, "dir": "c2s"})
    jf.write_text("".join(json.dumps(r) + "\n" for r in recs))
    prov = surfaces.get_surface("wire-shape")
    rows = list(prov.rows(mode="inference", source=str(jf)))
    assert len(rows) == 6
    assert rows[0][0] == [100.0, 4]
    assert rows[0][1]["session_id"] == "s0"
    assert rows[-1][1]["session_id"] is None
    with pytest.raises(ValueError, match="source"):
        list(prov.rows(mode="training"))


def test_toy_length_leak_mae_and_rank_corr(wire_shape_mod, tmp_path):
    art, journal = _run_toy(wire_shape_mod, tmp_path, "--arm", "length-leak")
    recs = [r for r in art["results"] if r["arm"] == "length-leak"]
    assert len(recs) == len(journal) == 2
    for r in recs:
        assert r["session_type"] == "inference"
        assert r["bytes_per_token_learned"] is True
        assert r["mae_tokens"] < 2.5
        assert r["rank_corr"] >= 0.9
        assert r["n_messages"] >= 4
        assert len(r["est_tokens"]) == len(r["true_tokens"])
    summ = [s for s in art["summary"] if s["arm"] == "length-leak"][0]
    assert summ["mae_tokens_mean"] < 2.5
    assert summ["rank_corr_mean"] >= 0.9


def test_toy_length_leak_supplied_model(wire_shape_mod, tmp_path):
    art, _ = _run_toy(wire_shape_mod, tmp_path, "--arm", "length-leak",
                      "--bytes-per-token", "4.0")
    for r in art["results"]:
        assert r["bytes_per_token"] == 4.0
        assert r["bytes_per_token_learned"] is False
        assert r["mae_tokens"] < 2.0


def test_toy_phase_detect_f1(wire_shape_mod, tmp_path):
    art, _ = _run_toy(wire_shape_mod, tmp_path, "--arm", "phase-detect")
    recs = [r for r in art["results"] if r["arm"] == "phase-detect"]
    for r in recs:
        assert r["f1"] >= 0.9
        assert r["n_true"] == r["n_pred"]
        assert r["label_accuracy"] == 1.0
        assert set(r["segment_labels"]) == {"prefill", "decode"}
    summ = [s for s in art["summary"] if s["arm"] == "phase-detect"][0]
    assert summ["f1_mean"] >= 0.9


def test_toy_training_session(wire_shape_mod, tmp_path):
    art, _ = _run_toy(wire_shape_mod, tmp_path, mode="training")
    assert all(r["session_type"] == "training" for r in art["results"])
    phase = [r for r in art["results"] if r["arm"] == "phase-detect"]
    assert all(r["f1"] >= 0.9 for r in phase)
    assert all(set(r["segment_labels"]) == {"training"} for r in phase)
    leak = [r for r in art["results"] if r["arm"] == "length-leak"]
    assert all(r["mae_tokens"] < 4.0 for r in leak)


def test_dispatch_toy_end_to_end(registered, tmp_path):
    from attacker.__main__ import main
    out = tmp_path / "disp.json"
    rc = main(["--mode", "inference", "--attack", "wire-shape",
               "--toy", "--quick", "--output", str(out)])
    assert rc == 0
    art = json.loads(out.read_text())
    assert art["schema"] == "dtraining.attacker.wire_shape.v1"
    assert art["config"]["journal_schema"] == "llm_attacker.wire_shape.v1"
    assert len(art["summary"]) == 2
    journal = (tmp_path / "disp.json.jsonl").read_text().strip().splitlines()
    assert len(journal) == len(art["results"])


def test_journal_input_path(wire_shape_mod, tmp_path):
    """A wire-shape journal (the MITM proxy's schema) analyzes per session;
    without ground truth the cells carry estimates, not MAE/F1."""
    mod = wire_shape_mod
    records = []
    for seed in (1, 2):
        recs, _ = mod.make_toy_session("inference", turns=3, seed=seed)
        for r in recs:
            r["session_id"] = f"session-{seed}"
        records.extend(recs)
    jf = tmp_path / "mitm.jsonl"
    jf.write_text("".join(json.dumps(r) + "\n" for r in records))
    args = mod.build_parser().parse_args(
        ["--journal", str(jf), "--bytes-per-token", "4.0",
         "--output", str(tmp_path / "j.json")])
    args.mode = "inference"
    assert mod.run(args) == 0
    art = json.loads((tmp_path / "j.json").read_text())
    assert art["config"]["source"] == "journal"
    sessions = {r["session_id"] for r in art["results"]}
    assert sessions == {"session-1", "session-2"}
    leak = [r for r in art["results"] if r["arm"] == "length-leak"]
    assert all("est_tokens" in r and "mae_tokens" not in r for r in leak)
    phase = [r for r in art["results"] if r["arm"] == "phase-detect"]
    assert all("f1" not in r and r["n_boundaries_pred"] > 0 for r in phase)


def test_journal_schema_drift_is_descriptive(wire_shape_mod, tmp_path):
    jf = tmp_path / "bad.jsonl"
    jf.write_text(json.dumps({"ts": 1.0, "bytes": 4, "dir": "up"}) + "\n")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        wire_shape_mod.read_journal(str(jf))


def test_evidence_chain(wire_shape_mod, tmp_path):
    out = tmp_path / "ev.json"
    args = wire_shape_mod.build_parser().parse_args(
        ["--toy", "--quick", "--evidence", "--output", str(out)])
    args.mode = "inference"
    assert wire_shape_mod.run(args) == 0
    manifest = str(out) + ".evidence.json"
    assert os.path.isfile(manifest)
    assert os.path.isfile(str(out) + ".evidence.jsonl")
    ok, problems = verify_evidence(manifest)
    assert ok and problems == []


def test_no_input_path_is_descriptive(wire_shape_mod):
    args = wire_shape_mod.build_parser().parse_args([])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="--journal"):
        wire_shape_mod.run(args)


def test_surface_registry_left_pristine(wire_shape_mod):
    assert "wire-shape" in surfaces.list_surfaces()
