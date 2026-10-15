"""artifacts (attacker/artifacts.py): the standard artifact skeleton,
crash-safe JSONL journal (append/resume semantics), atomic-ish final write,
provenance never-raises contract, and the optional evidence-chain hook."""

import json
import os
import runpy
import socket
import subprocess
import warnings

import pytest

from attacker import artifacts
from attacker.artifacts import (append_jsonl, light_provenance, make_artifact,
                                write_artifact)


def test_light_provenance_fields_and_format():
    prov = light_provenance()
    assert set(prov) == {"hostname", "container_image", "timestamp_utc",
                         "dtraining_commit"}
    assert prov["hostname"] == socket.gethostname()
    assert prov["timestamp_utc"].endswith("Z")
    assert len(prov["timestamp_utc"]) == 20
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    assert prov["dtraining_commit"] in (head, "unknown")


def test_light_provenance_reads_container_image_env(monkeypatch):
    monkeypatch.setenv("CONTAINER_IMAGE", "dtraining:test-tag")
    assert light_provenance()["container_image"] == "dtraining:test-tag"
    monkeypatch.delenv("CONTAINER_IMAGE")
    assert light_provenance()["container_image"] == "unknown"


def test_light_provenance_never_raises_on_git_failure(monkeypatch):
    def boom(*a, **k):
        raise OSError("no git on this host")
    monkeypatch.setattr(artifacts.subprocess, "run", boom)
    prov = light_provenance()
    assert prov["dtraining_commit"] == "unknown"


def test_light_provenance_empty_git_output_is_unknown(monkeypatch):
    monkeypatch.setattr(artifacts.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(
                            a[0], 0, stdout="", stderr=""))
    assert light_provenance()["dtraining_commit"] == "unknown"


def test_make_artifact_standard_skeleton():
    out = make_artifact("dtraining.attacker.test.v1", {"x": 1},
                        "test threat model")
    assert out["schema"] == "dtraining.attacker.test.v1"
    assert out["config"] == {"x": 1}
    assert out["threat_model"] == "test threat model"
    assert out["results"] == [] and out["summary"] == []
    assert set(out["provenance"]) == {"hostname", "container_image",
                                      "timestamp_utc", "dtraining_commit"}
    assert "interpretation" not in out
    assert json.loads(json.dumps(out)) == out


def test_make_artifact_interpretation_included_when_given():
    out = make_artifact("s", {}, "tm", interpretation="read as a floor")
    assert out["interpretation"] == "read as a floor"


def test_append_jsonl_appends_one_record_per_line(tmp_path):
    out = str(tmp_path / "art.json")
    append_jsonl(out, {"cell": 1, "top1": 3.2})
    append_jsonl(out, {"cell": 2, "error": "error: boom", "top1": None})
    lines = (tmp_path / "art.json.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"cell": 1, "top1": 3.2}
    assert json.loads(lines[1])["error"] == "error: boom"


def test_append_jsonl_resumes_an_existing_journal(tmp_path):
    out = str(tmp_path / "art.json")
    (tmp_path / "art.json.jsonl").write_text(
        json.dumps({"cell": 0, "note": "from an earlier crashed run"}) + "\n")
    append_jsonl(out, {"cell": 1})
    cells = [json.loads(ln)["cell"]
             for ln in (tmp_path / "art.json.jsonl").read_text().splitlines()]
    assert cells == [0, 1]


def test_append_jsonl_no_output_is_a_noop(tmp_path):
    append_jsonl(None, {"cell": 1})
    append_jsonl("", {"cell": 1})
    assert list(tmp_path.iterdir()) == []


def test_append_jsonl_unserializable_record_raises(tmp_path):
    out = str(tmp_path / "art.json")
    with pytest.raises(TypeError):
        append_jsonl(out, {"cell": 1, "bad": object()})


def test_write_artifact_round_trip_and_no_tmp_left(tmp_path):
    out = str(tmp_path / "art.json")
    art = make_artifact("dtraining.attacker.test.v1", {"x": 1}, "tm")
    write_artifact(out, art)
    assert json.loads((tmp_path / "art.json").read_text()) == art
    assert not (tmp_path / "art.json.tmp").exists()
    assert "\n  " in (tmp_path / "art.json").read_text()


def test_write_artifact_overwrites_atomically(tmp_path):
    out = str(tmp_path / "art.json")
    write_artifact(out, {"version": 1})
    write_artifact(out, {"version": 2})
    assert json.loads((tmp_path / "art.json").read_text()) == {"version": 2}
    assert not (tmp_path / "art.json.tmp").exists()


def test_write_artifact_no_output_is_a_noop(tmp_path, capsys):
    write_artifact(None, {"x": 1})
    write_artifact("", {"x": 1})
    assert list(tmp_path.iterdir()) == []
    assert "[artifact] wrote" not in capsys.readouterr().out


def test_write_artifact_announces_the_write(tmp_path, capsys):
    out = str(tmp_path / "art.json")
    write_artifact(out, {"x": 1})
    assert f"[artifact] wrote {out}" in capsys.readouterr().out


class _RecordingEvidence:
    """Duck-typed EvidenceLogger double recording the hook calls."""

    def __init__(self):
        self.events = []
        self.finalized = []

    def event(self, kind, payload=None):
        self.events.append((kind, payload))

    def finalize(self, artifact_path=None):
        self.finalized.append(artifact_path)


def test_make_artifact_records_creation_event():
    ev = _RecordingEvidence()
    make_artifact("dtraining.attacker.test.v1", {}, "tm", evidence=ev)
    assert ev.events == [("artifact_created",
                          {"schema": "dtraining.attacker.test.v1"})]


def test_write_artifact_records_and_finalizes_evidence(tmp_path):
    ev = _RecordingEvidence()
    out = str(tmp_path / "art.json")
    write_artifact(out, {"x": 1}, evidence=ev)
    assert [k for k, _ in ev.events] == ["artifact_write"]
    payload = ev.events[0][1]
    assert payload["path"] == out
    assert payload["size_bytes"] == os.path.getsize(out)
    from attacker.evidence import file_sha256
    assert payload["sha256"] == file_sha256(out)
    assert ev.finalized == [out]


def test_no_evidence_sidecars_without_a_logger(tmp_path):
    out = str(tmp_path / "art.json")
    write_artifact(out, make_artifact("s", {}, "tm"))
    assert not (tmp_path / "art.json.evidence.json").exists()
    assert not (tmp_path / "art.json.evidence.jsonl").exists()


def test_real_evidence_logger_chain_verifies(tmp_path):
    from attacker.evidence import EvidenceLogger, verify_evidence
    out = str(tmp_path / "art.json")
    ev = EvidenceLogger(out, node_role="pytest/cpu")
    art = make_artifact("dtraining.attacker.test.v1", {"x": 1}, "tm",
                        evidence=ev)
    write_artifact(out, art, evidence=ev)
    assert (tmp_path / "art.json.evidence.json").is_file()
    assert (tmp_path / "art.json.evidence.jsonl").is_file()
    ok, problems = verify_evidence(str(tmp_path / "art.json.evidence.json"))
    assert ok and problems == []


def test_self_test_passes():
    assert artifacts.self_test() == 0


def test_module_runs_standalone_torchless():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("attacker.artifacts", run_name="__main__")
    assert exc.value.code == 0
