"""Tests for attacker.evidence (pure stdlib, torch-free) and its hook into
attacker.artifacts."""

import json

import pytest

from attacker import artifacts
from attacker.evidence import (ENV_WHITELIST, GENESIS_HASH, EvidenceLogger,
                               file_sha256, verify_evidence)


def _write_artifact_file(path):
    with open(path, "w") as f:
        json.dump({"schema": "test.v1", "results": []}, f)


def _make_run(tmp_path, events=3):
    art = tmp_path / "run.json"
    _write_artifact_file(str(art))
    ev = EvidenceLogger(str(art), node_role="toula/gpu",
                        extra={"experiment": "pytest"})
    for i in range(events):
        ev.event("cell_result", {"cell": i, "top1": float(i)})
    ev.event("artifact_write", {"path": str(art)})
    manifest_path = ev.finalize(str(art))
    return ev, art, manifest_path


def test_manifest_fields(tmp_path):
    ev, art, manifest_path = _make_run(tmp_path)
    with open(manifest_path) as f:
        m = json.load(f)

    assert m["schema"] == "llm_attacker.evidence.v2"
    assert m["run_id"] == ev.run_id
    assert m["node_role"] == "toula/gpu"
    assert m["extra"] == {"experiment": "pytest"}
    for field in ("start_utc", "end_utc", "argv", "cwd", "hostname",
                  "platform", "python_version", "cpu_count", "packages",
                  "git", "env", "chain_hash", "event_count", "events_path"):
        assert field in m, field
    assert isinstance(m["argv"], list)
    assert isinstance(m["packages"], dict)
    assert set(m["git"]) == {"commit", "dirty", "source_sha256"}
    assert m["event_count"] == 6
    assert m["artifact"]["path"] == str(art)
    assert m["artifact"]["sha256"] == file_sha256(str(art))
    assert m["artifact"]["size_bytes"] == art.stat().st_size


def test_manifest_run_id_is_uuid4(tmp_path):
    import uuid
    ev, _, manifest_path = _make_run(tmp_path)
    m = json.load(open(manifest_path))
    parsed = uuid.UUID(m["run_id"])
    assert parsed.version == 4
    ev2 = EvidenceLogger(str(tmp_path / "other.json"))
    assert ev2.run_id != ev.run_id
    ev2.finalize()


def test_env_capture_is_whitelisted(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("A_SECRET_THAT_MUST_NOT_LEAK", "hunter2")
    ev, _, manifest_path = _make_run(tmp_path)
    m = json.load(open(manifest_path))
    assert m["env"] == {"CUDA_VISIBLE_DEVICES": "0,1"}
    assert all(k in ENV_WHITELIST for k in m["env"])


def test_chain_structure(tmp_path):
    _, art, _ = _make_run(tmp_path, events=2)
    lines = (tmp_path / "run.json.evidence.jsonl").read_text().splitlines()
    events = [json.loads(l) for l in lines]
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert events[0]["kind"] == "run_start"
    assert events[0]["prev_hash"] == GENESIS_HASH
    assert events[-1]["kind"] == "run_end"
    for prev, cur in zip(events, events[1:]):
        assert cur["prev_hash"] == prev["hash"]
    assert all(set(e) == {"seq", "ts_utc", "run_id", "kind", "payload",
                          "prev_hash", "hash"} for e in events)


def test_verify_clean_run(tmp_path):
    _, _, manifest_path = _make_run(tmp_path)
    ok, problems = verify_evidence(manifest_path)
    assert ok, problems
    assert problems == []


def test_tamper_event_line_detected(tmp_path):
    _, art, manifest_path = _make_run(tmp_path)
    events_path = tmp_path / "run.json.evidence.jsonl"
    lines = events_path.read_text().splitlines()
    victim = json.loads(lines[1])
    victim["payload"]["top1"] = 99.9
    lines[1] = json.dumps(victim)
    events_path.write_text("\n".join(lines) + "\n")
    ok, problems = verify_evidence(manifest_path)
    assert not ok
    assert any("event 1: hash mismatch" in p for p in problems)


def test_tamper_event_deleted_detected(tmp_path):
    _, _, manifest_path = _make_run(tmp_path)
    events_path = tmp_path / "run.json.evidence.jsonl"
    lines = events_path.read_text().splitlines()
    del lines[2]
    events_path.write_text("\n".join(lines) + "\n")
    ok, problems = verify_evidence(manifest_path)
    assert not ok
    assert problems


def test_tamper_manifest_chain_hash_detected(tmp_path):
    _, _, manifest_path = _make_run(tmp_path)
    m = json.load(open(manifest_path))
    m["chain_hash"] = GENESIS_HASH
    with open(manifest_path, "w") as f:
        json.dump(m, f)
    ok, problems = verify_evidence(manifest_path)
    assert not ok
    assert any("hash mismatch" in p for p in problems)


def test_tamper_artifact_detected(tmp_path):
    _, art, manifest_path = _make_run(tmp_path)
    with open(art, "a") as f:
        f.write(" ")
    ok, problems = verify_evidence(manifest_path)
    assert not ok
    assert any("artifact sha256 mismatch" in p for p in problems)


def test_verify_missing_manifest():
    ok, problems = verify_evidence("/nonexistent/evidence.json")
    assert not ok
    assert any("unreadable" in p for p in problems)


def test_event_after_finalize_raises(tmp_path):
    ev = EvidenceLogger(str(tmp_path / "x.json"))
    ev.finalize()
    with pytest.raises(ValueError):
        ev.event("late", {})
    assert ev.finalize() == ev.manifest_path


def test_context_manager_seals_on_success(tmp_path):
    art = tmp_path / "cm.json"
    _write_artifact_file(str(art))
    with EvidenceLogger(str(art), node_role="cm/test") as ev:
        ev.event("cell_result", {"cell": 1})
    ok, problems = verify_evidence(str(art) + ".evidence.json")
    assert ok, problems


def test_context_manager_records_error(tmp_path):
    art = tmp_path / "cmerr.json"
    with pytest.raises(RuntimeError):
        with EvidenceLogger(str(art)) as ev:
            ev.event("cell_result", {"cell": 1})
            raise RuntimeError("boom")
    events = [json.loads(l) for l in
              (tmp_path / "cmerr.json.evidence.jsonl").read_text()
              .splitlines()]
    assert any(e["kind"] == "error" and "boom" in e["payload"]["error"]
               for e in events)


def test_unserializable_payload_does_not_raise(tmp_path):
    ev = EvidenceLogger(str(tmp_path / "u.json"))
    rec = ev.event("weird", {"obj": object()})
    assert "_unserializable" in rec["payload"]
    ev.finalize()
    ok, problems = verify_evidence(ev.manifest_path)
    assert not ok
    assert problems == ["artifact was missing at finalize time"]


def test_write_artifact_with_evidence(tmp_path):
    out_path = str(tmp_path / "art.json")
    ev = EvidenceLogger(out_path, node_role="toula/gpu")
    art = artifacts.make_artifact("test.schema.v1", {"x": 1},
                                  "pytest threat model", evidence=ev)
    artifacts.write_artifact(out_path, art, evidence=ev)

    manifest_path = out_path + ".evidence.json"
    events_path = out_path + ".evidence.jsonl"
    assert (tmp_path / "art.json").exists()
    assert (tmp_path / "art.json.evidence.json").exists()
    assert (tmp_path / "art.json.evidence.jsonl").exists()

    m = json.load(open(manifest_path))
    assert m["artifact"]["sha256"] == file_sha256(out_path)
    kinds = [json.loads(l)["kind"] for l in open(events_path)]
    assert kinds == ["run_start", "artifact_created", "artifact_write",
                     "run_end"]
    write_ev = [json.loads(l) for l in open(events_path)
                if json.loads(l)["kind"] == "artifact_write"][0]
    assert write_ev["payload"]["sha256"] == m["artifact"]["sha256"]

    ok, problems = verify_evidence(manifest_path)
    assert ok, problems


def test_write_artifact_default_none_writes_no_sidecars(tmp_path):
    out_path = str(tmp_path / "plain.json")
    art = artifacts.make_artifact("test.schema.v1", {}, "tm")
    artifacts.write_artifact(out_path, art)
    assert (tmp_path / "plain.json").exists()
    assert not (tmp_path / "plain.json.evidence.json").exists()
    assert not (tmp_path / "plain.json.evidence.jsonl").exists()


def test_write_artifact_atomic_pattern_intact_with_evidence(tmp_path):
    out_path = str(tmp_path / "atomic.json")
    ev = EvidenceLogger(out_path)
    artifacts.write_artifact(out_path,
                             artifacts.make_artifact("s", {}, "tm"),
                             evidence=ev)
    assert not (tmp_path / "atomic.json.tmp").exists()
    assert json.load(open(out_path))["schema"] == "s"


def test_verify_evidence_after_relocation(tmp_path):
    """Regression (toula remote-fetch): the manifest records absolute paths
    from the writer host; after the whole set is moved, verify_evidence must
    resolve events log AND artifact as manifest siblings — not mis-parse the
    artifact as the events log."""
    import shutil

    src = tmp_path / "src"
    src.mkdir()
    out_path = str(src / "run.json")
    ev = EvidenceLogger(out_path, node_role="test")
    artifacts.write_artifact(out_path,
                             artifacts.make_artifact("s", {}, "tm"),
                             evidence=ev)

    dst = tmp_path / "fetched" / "node" / "run"
    dst.mkdir(parents=True)
    for f in src.iterdir():
        shutil.copy2(f, dst / f.name)
    shutil.rmtree(src)

    ok, problems = verify_evidence(str(dst / "run.json.evidence.json"))
    assert ok, problems


def test_verify_evidence_relocation_detects_tamper(tmp_path):
    """The relocated fallback must not weaken tamper detection."""
    import shutil

    src = tmp_path / "src"
    src.mkdir()
    out_path = str(src / "run.json")
    ev = EvidenceLogger(out_path)
    artifacts.write_artifact(out_path,
                             artifacts.make_artifact("s", {}, "tm"),
                             evidence=ev)

    dst = tmp_path / "dst"
    dst.mkdir()
    for f in src.iterdir():
        shutil.copy2(f, dst / f.name)
    shutil.rmtree(src)

    p = dst / "run.json"
    data = p.read_bytes()
    p.write_bytes(b"X" + data[1:])
    ok, problems = verify_evidence(str(dst / "run.json.evidence.json"))
    assert not ok
    assert any("sha256" in pr for pr in problems)


def test_logger_creates_missing_parent_dirs(tmp_path):
    """Regression (toula attack runs): EvidenceLogger must create the
    output's parent dirs instead of FileNotFoundError."""
    out = tmp_path / "fresh" / "deep" / "cell.json"
    ev = EvidenceLogger(str(out))
    artifacts.write_artifact(str(out),
                             artifacts.make_artifact("s", {}, "tm"),
                             evidence=ev)
    ok, problems = verify_evidence(str(out) + ".evidence.json")
    assert ok, problems


def test_package_inventory_preserves_first_distribution_precedence(monkeypatch):
    from importlib import metadata
    from types import SimpleNamespace
    from attacker.evidence import _installed_packages
    distributions = [
        SimpleNamespace(metadata={'Name': 'transformers'}, version='4.57.6'),
        SimpleNamespace(metadata={'Name': 'torch'}, version='2.11.0'),
        SimpleNamespace(metadata={'Name': 'Transformers'}, version='5.15.0'),
    ]
    monkeypatch.setattr(metadata, 'distributions', lambda: distributions)
    assert _installed_packages() == {'torch': '2.11.0', 'transformers': '4.57.6'}
