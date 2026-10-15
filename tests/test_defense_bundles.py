import copy
import hashlib
import json
from pathlib import Path

import pytest

from attacker import defense_bundles as db
from attacker.attacks import replica_integrity as ri
from attacker.attacks import representation_privacy as rp
from attacker.attacks import view_privacy as vp

FIXTURE_DIR = Path(__file__).parent / "data" / "defense_bundle_v1"
BUNDLE_PATH = FIXTURE_DIR / "bundle.json"


def _load_raw():
    return json.loads(BUNDLE_PATH.read_text(), object_pairs_hook=db._strict_object)


def _write_bundle(tmp_path, bundle):
    for name in ("captures.json", "truth.json"):
        (tmp_path / name).write_text((FIXTURE_DIR / name).read_text())
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    return path


def test_golden_bundle_validates():
    bundle = db.load_bundle(BUNDLE_PATH)
    assert bundle["schema"] == db.BUNDLE_SCHEMA
    ids = [c["id"] for c in bundle["components"]]
    assert ids == ["rep", "views", "captures", "truth", "phase", "kdf", "decoys", "layout"]
    assert bundle["provenance"]["framework"] == "golden-fixture-framework"


def test_sha256_pinning_verified(tmp_path):
    raw = _load_raw()
    for component in raw["components"]:
        if component["id"] == "captures":
            component["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256 mismatch"):
        db.load_bundle(_write_bundle(tmp_path, raw))
    path = _write_bundle(tmp_path, _load_raw())
    (tmp_path / "captures.json").write_text(json.dumps(
        {"schema": "llm_attacker.replica_observations.v1",
         "samples": [{"id": "y", "replicas": [[1.0], [1.0]], "key_ids": ["a", "b"]}]}))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        db.load_bundle(path)


@pytest.mark.parametrize("mutation,match", [
    ("bad-schema", "unsupported defense bundle schema"),
    ("no-provenance", "provenance"),
    ("provenance-field", "provenance.framework"),
    ("unknown-kind", "unknown component kind"),
    ("duplicate-id", "duplicate component id"),
    ("no-payload", "exactly one of inline or path"),
    ("both-payloads", "exactly one of inline or path"),
    ("missing-file", "payload file not found"),
    ("escape", "inside the bundle directory"),
    ("unknown-field", "unknown fields"),
])
def test_envelope_mutation_matrix(tmp_path, mutation, match):
    raw = _load_raw()
    if mutation == "bad-schema":
        raw["schema"] = "llm_attacker.defense_bundle.v0"
    elif mutation == "no-provenance":
        del raw["provenance"]
    elif mutation == "provenance-field":
        raw["provenance"]["framework"] = ""
    elif mutation == "unknown-kind":
        raw["components"][0]["kind"] = "mystery.v1"
    elif mutation == "duplicate-id":
        raw["components"].append(copy.deepcopy(raw["components"][0]))
    elif mutation == "no-payload":
        raw["components"][0].pop("inline")
    elif mutation == "both-payloads":
        raw["components"][0]["path"] = "captures.json"
    elif mutation == "missing-file":
        raw["components"] = [c for c in raw["components"] if "path" not in c]
        raw["components"].append({"id": "gone", "kind": "kdf_metadata.v1", "path": "nope.json"})
    elif mutation == "escape":
        raw["components"] = [c for c in raw["components"] if "path" not in c]
        raw["components"].append({"id": "esc", "kind": "kdf_metadata.v1", "path": "../escape.json"})
    else:
        raw["components"][0]["bogus"] = 1
    with pytest.raises(ValueError, match=match):
        db.load_bundle(_write_bundle(tmp_path, raw))


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text(BUNDLE_PATH.read_text().replace(
        '"framework": "golden-fixture-framework"',
        '"framework": "a", "framework": "b"'))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        db.load_bundle(path)


@pytest.mark.parametrize("kind,payload,match", [
    ("kdf_metadata.v1", {"schema": "kdf_metadata.v1", "algorithm": "a", "domain": "d", "master_bits": 0},
     "master_bits"),
    ("kdf_metadata.v1", {"schema": "kdf_metadata.v1", "algorithm": "a", "domain": "d", "master_bits": 128, "epoch": -1},
     "epoch"),
    ("decoy_pool.v1", {"schema": "decoy_pool.v1", "pool": [[1.0]], "honest_labels": [0, 1]},
     "honest_labels"),
    ("decoy_pool.v1", {"schema": "decoy_pool.v1", "pool": [], "honest_labels": []}, "pool"),
    ("view_layout.v1", {"schema": "view_layout.v1", "n_views": 2, "coords": [[0]]}, "coords"),
    ("view_layout.v1", {"schema": "view_layout.v1", "n_views": 1, "coords": [[0, 0]]}, "coords"),
    ("phase_manifest.v1", {"schema": "phase_manifest.v1", "phase_provenance": {}, "phase_records": []},
     "phase"),
    ("mystery.v1", {}, "unknown component kind"),
])
def test_component_validator_mutations(kind, payload, match):
    with pytest.raises(ValueError, match=match):
        db.validate_component(kind, payload)


def test_materialize_round_trip(tmp_path):
    manifest = db.materialize(BUNDLE_PATH, tmp_path / "out")
    assert manifest["schema"] == db.MANIFEST_SCHEMA
    components = manifest["components"]
    assert components["rep"]["target"] == "representation-privacy"
    assert components["rep"]["flags"][0] == "--bundle"
    assert components["views"]["target"] == "view-privacy"
    assert components["captures"]["flags"][0] == "--captures"
    assert components["truth"]["flags"][0] == "--truth"
    assert components["kdf"]["target"] == "metadata"
    assert (tmp_path / "out" / "manifest.json").is_file()
    for entry in components.values():
        assert db.file_sha256(entry["file"]) == entry["sha256"]

    rp_args = rp.build_parser().parse_args(["--bundle", components["rep"]["file"],
                                            "--arms", "bottleneck"])
    assert rp.run(rp_args) == 0
    samples, targets = ri.load_inputs(components["captures"]["file"],
                                      components["truth"]["file"])
    assert samples[0]["id"] == "x" and targets == {"x": [0.0]}
    rows = vp.evaluate_bundle(json.loads(Path(components["views"]["file"]).read_text()))
    assert rows and all(r["case"] == "views" for r in rows)


def test_cli_round_trip(tmp_path, capsys):
    assert db.main(["validate", str(BUNDLE_PATH)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == db.BUNDLE_SCHEMA and "rep" in out["components"]
    assert db.main(["materialize", str(BUNDLE_PATH), "--out", str(tmp_path / "cli")]) == 0
    targets = json.loads(capsys.readouterr().out)
    assert targets["rep"] == "representation-privacy"
    assert (tmp_path / "cli" / "manifest.json").is_file()


def _obs_truth():
    obs = {"schema": "llm_attacker.replica_observations.v1", "samples": [
        {"id": "x", "replicas": [[0.0], [0.0], [0.0]], "key_ids": ["a", "a", "b"]}]}
    truth = {"schema": "llm_attacker.replica_truth.v1", "targets": {"x": [0.0]}}
    return obs, truth


_MUTATIONS = ["wrong-schema", "extra-key", "empty-samples", "not-dict",
              "truth-wrong-schema", "truth-extra-key", "truth-not-dict"]


def _mutate(obs, truth, mutation):
    if mutation == "wrong-schema":
        obs["schema"] = "v0"
    elif mutation == "extra-key":
        obs["extra"] = 1
    elif mutation == "empty-samples":
        obs["samples"] = []
    elif mutation == "not-dict":
        obs = [1]
    elif mutation == "truth-wrong-schema":
        truth["schema"] = "v0"
    elif mutation == "truth-extra-key":
        truth["extra"] = 1
    else:
        truth = "nope"
    return obs, truth


@pytest.mark.parametrize("mutation", _MUTATIONS)
def test_refactored_validators_reject_identically(tmp_path, mutation):
    obs, truth = _mutate(*_obs_truth(), mutation)
    cp, tp = tmp_path / "captures.json", tmp_path / "truth.json"
    cp.write_text(json.dumps(obs))
    tp.write_text(json.dumps(truth))
    with pytest.raises(ValueError) as via_load:
        ri.load_inputs(cp, tp)
    with pytest.raises(ValueError) as via_validate:
        if mutation.startswith("truth"):
            ri.validate_truth(truth)
        else:
            ri.validate_observations(obs)
    assert str(via_load.value) == str(via_validate.value)


def test_refactored_validators_accept_valid_payloads():
    obs, truth = _obs_truth()
    assert ri.validate_observations(obs) is obs
    assert ri.validate_truth(truth) is truth
