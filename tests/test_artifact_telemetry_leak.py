"""artifact-telemetry-leak: canary-based persistent/operational store
scanner — toy fixture pair (leaking vs redacted), the negative control,
roots safety, canary-file errors, and determinism. All torch-free."""

import json
import os

import pytest

from attacker import threat_model
from attacker.attacks import validate_contract
from attacker.attacks import artifact_telemetry_leak as atl


def parse(*argv):
    return atl.build_parser().parse_args(list(argv))


def test_contract():
    assert validate_contract(atl) is atl
    assert atl.EXPERIMENT_ID == "artifact_telemetry_leak"
    assert atl.MODES == ("training", "inference")
    assert atl.REQUIRES_LABELS is False
    assert isinstance(atl.DESCRIPTION, str) and atl.DESCRIPTION


def test_help_works_torchless():
    with pytest.raises(SystemExit) as exc:
        parse("--help")
    assert exc.value.code == 0


def test_parser_flags():
    args = parse("--toy", "--quick", "--deletion-deadline", "12",
                 "--roots", "a", "b", "--canary-file", "c.json",
                 "--work-dir", "w")
    assert args.toy and args.quick
    assert args.deletion_deadline == 12.0
    assert args.roots == ["a", "b"]
    assert args.canary_file == "c.json"
    assert args.work_dir == "w"
    assert hasattr(args, "output") and hasattr(args, "seed") \
        and hasattr(args, "seeds") and hasattr(args, "evidence")


@pytest.fixture
def toy_run(tmp_path):
    output = str(tmp_path / "atl.json")
    args = parse("--toy", "--quick", "--output", output,
                 "--work-dir", str(tmp_path / "work"))
    assert atl.run(args) == 0
    with open(output) as f:
        return output, json.load(f)


def test_toy_artifact_layout(toy_run):
    output, art = toy_run
    for key in ("schema", "config", "threat_model", "provenance",
                "results", "summary"):
        assert key in art
    assert art["schema"] == "dtraining.attacker.artifact_telemetry_leak.v1"
    assert art["config"]["attack"] == atl.EXPERIMENT_ID
    assert art["summary"]
    threat_model.validate(art["threat_model"])
    assert art["threat_model"]["schema"] == threat_model.SCHEMA_V2
    assert art["threat_model"]["actor"]["intent"] == "passive"
    assert art["threat_model"]["mappings"]["owasp_llm_2026"] == \
        ["LLM02", "LLM08"]
    with open(output + ".jsonl") as f:
        journal = f.read().strip().splitlines()
    assert len(journal) == len(art["results"])
    assert all(json.loads(l)["experiment"] == atl.EXPERIMENT_ID
               for l in journal)


def test_every_arm_and_store_recorded(toy_run):
    _, art = toy_run
    assert art["results"]
    assert all(r["experiment"] == atl.EXPERIMENT_ID for r in art["results"])
    arms = {r["arm"] for r in art["results"]}
    assert {"store-scan", "redaction-check"} <= arms
    stores = {r["store"] for r in art["results"] if r["arm"] == "store-scan"}
    assert stores == {"toy-leaking", "toy-redacted"}


def test_negative_control_redacted_world_is_clean(toy_run):
    _, art = toy_run
    red = next(r for r in art["results"] if r.get("store") == "toy-redacted")
    assert red["role"] == "negative-control"
    assert red["n_hits"] == 0
    assert red["unauthorized_readability_count"] == 0
    assert red["retention_beyond_deadline_count"] == 0
    assert red["exposure_by_class"]
    assert all(v["exposed"] == 0
               for v in red["exposure_by_class"].values())
    pair = next(r for r in art["results"] if r["arm"] == "redaction-check")
    assert pair["redacted_hits"] == 0
    assert pair["redaction_precision"] == 1.0
    assert pair["redaction_recall"] == 1.0


def test_vulnerable_world_leaks(toy_run):
    """The leaking world must show the attack working — otherwise the
    negative control above is vacuous."""
    _, art = toy_run
    leak = next(r for r in art["results"] if r.get("store") == "toy-leaking")
    assert leak["role"] == "vulnerable-world"
    assert leak["n_hits"] >= 5
    assert sorted(leak["classes_exposed"]) == sorted(atl.DATA_CLASSES)
    assert all(v["rate"] == 1.0
               for v in leak["exposure_by_class"].values())
    variants = {v for h in leak["hits"] for v in h["variants"]}
    assert {"exact", "base64", "truncated", "token_split"} <= variants
    assert leak["unauthorized_readability_count"] >= 1
    assert "captures/labels.jsonl" in leak["world_readable_files"]
    assert leak["retention_beyond_deadline_count"] == 1
    assert leak["retention_beyond_deadline_files"] == \
        ["backups/run_0042.bak.jsonl"]
    assert leak["copies_per_canary"]["canary-labels"] == 2
    pair = next(r for r in art["results"] if r["arm"] == "redaction-check")
    assert pair["scanner_fixture_recall"] == 1.0
    values = [c["value"] for c in atl.make_canaries(1234)]
    for h in leak["hits"]:
        assert "sha256" in h and "offsets" in h and "value" not in h
        assert not any(v in json.dumps(h) for v in values)


def test_determinism(tmp_path):
    runs = []
    for i in (0, 1):
        output = tmp_path / f"d{i}.json"
        args = parse("--toy", "--quick", "--output", str(output),
                     "--work-dir", str(tmp_path / f"w{i}"))
        assert atl.run(args) == 0
        with open(output) as f:
            art = json.load(f)
        runs.append((art["results"], art["summary"]))
    assert runs[0] == runs[1]


def test_roots_mode_scans_authorized_store(tmp_path):
    store = tmp_path / "leaking"
    atl.make_fixture_store(str(store), 5, redacted=False,
                           canaries=atl.make_canaries(1234))
    output = tmp_path / "roots.json"
    args = parse("--roots", str(store), "--output", str(output))
    assert atl.run(args) == 0
    with open(output) as f:
        art = json.load(f)
    cell = art["results"][0]
    assert cell["arm"] == "store-scan"
    assert cell["role"] == "evaluator-authorized-store"
    assert cell["root"] == os.path.realpath(str(store))
    assert cell["n_hits"] >= 5
    assert sorted(cell["classes_exposed"]) == sorted(atl.DATA_CLASSES)
    assert art["summary"][0]["n_hits_total"] == cell["n_hits"]


def test_roots_mode_with_canary_file(tmp_path):
    store = tmp_path / "leaking"
    canaries = [{"id": "c-cred", "data_class": "credentials",
                 "value": "CANARY-credentials-deadbeef"}]
    atl.make_fixture_store(str(store), 5, redacted=False, canaries=[
        {"id": c["id"], "data_class": c["data_class"], "value": c["value"]}
        for c in canaries])
    cfile = tmp_path / "canaries.json"
    cfile.write_text(json.dumps({"schema": atl.CANARY_FILE_SCHEMA,
                                 "canaries": canaries}))
    output = tmp_path / "roots.json"
    args = parse("--roots", str(store), "--canary-file", str(cfile),
                 "--output", str(output))
    assert atl.run(args) == 0
    with open(output) as f:
        art = json.load(f)
    assert art["config"]["canary_file"] == str(cfile)
    hits = art["results"][0]["hits"]
    assert {h["canary_id"] for h in hits} == {"c-cred"}
    assert hits[0]["variants"] == ["base64"]


def test_roots_safety_and_missing_paths(tmp_path):
    with pytest.raises(ValueError, match="refusing"):
        atl.validate_roots(["/"])
    with pytest.raises(ValueError, match="refusing"):
        atl.validate_roots([os.path.expanduser("~")])
    with pytest.raises(ValueError, match="no-such-dir"):
        atl.validate_roots([str(tmp_path / "no-such-dir")])
    with pytest.raises(SystemExit, match="refusing"):
        atl.run(parse("--roots", "/"))
    with pytest.raises(SystemExit, match="no-such-dir"):
        atl.run(parse("--roots", str(tmp_path / "no-such-dir")))


def test_malformed_canary_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="bad.json"):
        atl.load_canaries(str(bad))
    with pytest.raises(ValueError, match="missing.json"):
        atl.load_canaries(str(tmp_path / "missing.json"))
    notalist = tmp_path / "notalist.json"
    notalist.write_text(json.dumps({"canaries": {"value": 1}}))
    with pytest.raises(ValueError, match="notalist.json"):
        atl.load_canaries(str(notalist))
    novalue = tmp_path / "novalue.json"
    novalue.write_text(json.dumps([{"id": "x"}]))
    with pytest.raises(ValueError, match="novalue.json"):
        atl.load_canaries(str(novalue))
    store = tmp_path / "leaking"
    atl.make_fixture_store(str(store), 5, redacted=False)
    with pytest.raises(SystemExit, match="bad.json"):
        atl.run(parse("--roots", str(store), "--canary-file", str(bad)))


def test_no_input_path_is_a_descriptive_error():
    with pytest.raises(SystemExit, match="pick an input path"):
        atl.run(parse())


def test_self_test_passes():
    assert atl.self_test() == 0
